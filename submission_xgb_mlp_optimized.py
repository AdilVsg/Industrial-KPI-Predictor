import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb # Used only for fast feature selection
from scipy.optimize import minimize
import os
import gc
import warnings

warnings.filterwarnings('ignore')

# 1) CONFIGURATION
DATA_DIR = r"C:\Users\adilc\OneDrive\Bureau\Hackathon\data" # Update this path
N_FOLDS = 5
SEED = 42

# DIFFERENTIATED SELECTION
FEATS_XGB = 500
FEATS_MLP = 150

print("Loading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
test_ids = test_df['id']
targets = ['wip', 'investissement', 'satisfaction']

def optimize_floats(df):
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    if floats:
        df[floats] = df[floats].astype('float32')
    return df

train_df = optimize_floats(train_df)
test_df = optimize_floats(test_df)

y = train_df[targets].values
X = train_df.drop(columns=targets + ['id'])
X_test = test_df.drop(columns=['id'])

del train_df, test_df
gc.collect()

# 2) DATA PREP
print("Data Prep (Encoding & Cleaning)...")

# A. Ordinal Encoding (Fast and efficient for trees)
cat_cols = X.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[cat_cols] = encoder.fit_transform(X[cat_cols])
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# B. Filling NaN values
X = X.fillna(0)
X_test = X_test.fillna(0)

# C. Dropping useless columns (Quasi-zero variance)
selector_var = VarianceThreshold(threshold=0.01)
X = pd.DataFrame(selector_var.fit_transform(X))
X_test = pd.DataFrame(selector_var.transform(X_test))
print(f"   Features after variance cleaning: {X.shape[1]}")

# 3) FEATURE SELECTION (LGBM IMPORTANCE)
print("Calculating feature importance...")

# Using a fast LGBM to rank columns
lgb_sel = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=SEED, n_jobs=-1, verbose=-1)
lgb_sel.fit(X, y[:, 2]) # Basing importance on 'Satisfaction' target

# Creating the ranking
feature_imp = pd.DataFrame(sorted(zip(lgb_sel.feature_importances_, X.columns)), columns=['Value','Feature'])
sorted_features = feature_imp.sort_values(by="Value", ascending=False)['Feature'].tolist()

# Creating two distinct datasets
# 1. Dataset for XGBoost (Top 500)
top_xgb = sorted_features[:FEATS_XGB]
X_for_xgb = X[top_xgb].values
X_test_for_xgb = X_test[top_xgb].values

# 2. Dataset for MLP (Top 150)
top_mlp = sorted_features[:FEATS_MLP]
X_for_mlp = X[top_mlp].values
X_test_for_mlp = X_test[top_mlp].values

print(f"   Datasets ready: XGB ({X_for_xgb.shape[1]} cols) | MLP ({X_for_mlp.shape[1]} cols)")
del X, X_test
gc.collect()

# 4) MLP SPECIFIC PREP (SCALING)
print("Scaling for MLP...")
# MLP strictly requires data between -1 and 1 (or 0 and 1) to converge
scaler_x = StandardScaler()
X_for_mlp = scaler_x.fit_transform(X_for_mlp)
X_test_for_mlp = scaler_x.transform(X_test_for_mlp)

# 5) DUAL TRAINING (XGB + MLP)

oof_xgb = np.zeros(y.shape)
test_xgb = np.zeros((X_test_for_xgb.shape[0], 3))

oof_mlp = np.zeros(y.shape)
test_mlp = np.zeros((X_test_for_mlp.shape[0], 3))

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

print("Starting Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_for_xgb, y)):
    print(f"   Fold {fold+1}/{N_FOLDS}...")
    
    # MODEL A: XGBOOST
    xgb_model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.7,
        objective='reg:absoluteerror',
        tree_method='hist',
        n_jobs=-1,
        random_state=SEED
    )
    chain_xgb = RegressorChain(xgb_model, order=[0, 1, 2])
    chain_xgb.fit(X_for_xgb[train_idx], y[train_idx])
    
    oof_xgb[val_idx] = chain_xgb.predict(X_for_xgb[val_idx])
    test_xgb += chain_xgb.predict(X_test_for_xgb) / N_FOLDS
    
    # MODEL B: MLP (NEURAL NET)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y[train_idx])
    
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.01,
        batch_size=128,
        learning_rate_init=0.005,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED
    )
    # MLP handles multi-output natively, no need for Chain (often better this way)
    mlp_model.fit(X_for_mlp[train_idx], y_train_scaled)
    
    # Prediction and Inverse Scaling
    pred_scaled = mlp_model.predict(X_for_mlp[val_idx])
    oof_mlp[val_idx] = scaler_y.inverse_transform(pred_scaled)
    
    pred_test_scaled = mlp_model.predict(X_test_for_mlp)
    test_mlp += scaler_y.inverse_transform(pred_test_scaled) / N_FOLDS
    
    # INTERMEDIATE SCORES
    score_xgb = np.mean(np.abs(y[val_idx][:, 2] - oof_xgb[val_idx][:, 2]) < 0.05)
    score_mlp = np.mean(np.abs(y[val_idx][:, 2] - oof_mlp[val_idx][:, 2]) < 0.05)
    print(f"      Score XGB: {score_xgb:.4f} | Score MLP: {score_mlp:.4f}")

# 6) BLEND OPTIMIZATION
print("Optimizing the blend...")

y_true = y[:, 2] # Real Satisfaction
p_xgb = oof_xgb[:, 2]
p_mlp = oof_mlp[:, 2]

def objective_func(weights):
    w1, w2 = weights
    total = w1 + w2
    if total == 0: return 1
    
    final_pred = (w1 * p_xgb + w2 * p_mlp) / total
    residuals = np.abs(y_true - final_pred)
    return -np.mean(residuals < 0.05) # Maximizing the score < 0.05

res = minimize(objective_func, [0.7, 0.3], bounds=[(0, 1), (0, 1)], method='Nelder-Mead')
best_w = res.x / np.sum(res.x)

print(f"OPTIMAL WEIGHTS: XGB={best_w[0]:.3f}, MLP={best_w[1]:.3f}")
print(f"ESTIMATED FINAL CV SCORE: {-res.fun:.5f}")

# 7) SUBMISSION
print("Generating submission file...")

final_wip = best_w[0]*test_xgb[:,0] + best_w[1]*test_mlp[:,0]
final_inv = best_w[0]*test_xgb[:,1] + best_w[1]*test_mlp[:,1]
final_sat = best_w[0]*test_xgb[:,2] + best_w[1]*test_mlp[:,2]

submission = pd.DataFrame({
    'id': test_ids,
    'wip': final_wip,
    'investissement': final_inv,
    'satisfaction': final_sat.clip(0, 1)
})

output_path = os.path.join(DATA_DIR, 'submission_xgb_mlp_optimized.csv')
submission.to_csv(output_path, index=False)
print(f"Done! File saved at: {output_path}")
