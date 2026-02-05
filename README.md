# Hybrid XGBoost & MLP for Industrial KPI Prediction

![Author](https://img.shields.io/badge/Author-Adil%20CHOUKAIRE-blue)
![Team](https://img.shields.io/badge/Team-Team%204-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)


Machine Learning solution designed to replace physical industrial simulations by predicting key performance indicators (KPIs) for aerospace workshops. This project implements a **Hybrid Ensemble Strategy** combining Gradient Boosting and Neural Networks.

## 📁 Project Structure
```bash
INDUSTRIAL-KPI-PREDICTOR/
├── 📁 data/
│   ├── train.csv
│   └── test.csv
├── submission_xgb_mlp_optimized.py
├── requirements.txt
└── README.md
```

## 🛠️ Script Description

### `submission_xgb_mlp_optimized.py`

This is the core script containing the entire pipeline. It implements a **Dual-Pipeline Strategy** to handle the specificities of both Tree-based models and Neural Networks.

#### 1. Data Preparation & Engineering
- **Optimization:** Casts `float64` to `float32` for memory efficiency.
- **Cleaning:** Removes columns with quasi-zero variance using `VarianceThreshold(0.01)`.
- **Differentiated Feature Selection:**
  - Uses a lightweight **LightGBM** to rank feature importance.
  - Selects **Top 500** features for XGBoost (Robust to noise).
  - Selects **Top 150** features for MLP (Requires high signal-to-noise ratio).

#### 2. Model A: The Structural Pillar (XGBoost)
- **Architecture:** Uses a `RegressorChain` to capture dependencies (`WIP` -> `Investment` -> `Satisfaction`).
- **Encoding:** Ordinal Encoding for categorical variables.
- **Parameters:** Tuned with `tree_method='hist'` for speed and `reg:absoluteerror` (MAE) to match the evaluation metric.

#### 3. Model B: The Non-Linear Specialist (MLP)
- **Architecture:** Multi-Layer Perceptron (Dense 128 -> 64).
- **Critical Preprocessing:**
  - **StandardScaler** applied to Inputs.
  - **Target Scaling:** Applies `StandardScaler` to the target variables ($Y$). This prevents the "Investment" values (millions) from crushing the gradients of the "Satisfaction" score (0-1).

#### 4. Ensemble Optimization
- **Strategy:** Weighted Blending.
- **Algorithm:** Uses **Nelder-Mead** optimization on Out-Of-Fold (OOF) predictions to dynamically find the best weight distribution between XGBoost and MLP.

## 🎯 Objective

The goal is to predict three industrial targets based on workshop configuration:
1.  **WIP:** Work In Progress.
2.  **Investment:** Cost in millions.
3.  **Satisfaction:** A normalized score [0-1].

**Challenge:** The project is evaluated strictly on the **Satisfaction** accuracy with a tolerance of `0.05`.

## ⚠️ Notes

- **Performance:** The model achieves a Cross-Validation score of ~0.757.
- **Resource Usage:** The script is optimized for CPU execution (`n_jobs=-1`) and completes in under 15 minutes.
- **Configuration:** Ensure the `DATA_DIR` variable in the script points to your local data folder.
