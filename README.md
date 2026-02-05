# INDUSTRIAL-KPI-PREDICTOR

![Author](https://img.shields.io/badge/Author-Adil%20CHOUKAIRE-blue)
![Client](https://img.shields.io/badge/Client-Airbus-00205B)
![Status](https://img.shields.io/badge/Status-Completed-success)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)


Machine Learning solution designed for **Airbus** to replace physical industrial simulations by predicting key performance indicators (KPIs) for aerospace workshops. This project implements a **Hybrid Ensemble Strategy** combining Gradient Boosting and Neural Networks.

## 📁 Project Structure
```bash
INDUSTRIAL-KPI-PREDICTOR/
├── 📁 data/
│   ├── train.csv  # (Not included - Confidential)
│   └── test.csv   # (Not included - Confidential)
├── submission_xgb_mlp_optimized.py
├── requirements.txt
└── README.md
```
> **⚠️ Data Confidentiality:** The `train.csv` and `test.csv` files contain proprietary industrial data from **Airbus**. For confidentiality reasons, they are **not included** in this repository. To run the script, please place the authorized dataset in the `data/` directory.

## 🎯 Objective

The goal is to predict three industrial targets based on workshop configuration:
1.  **WIP:** Work In Progress.
2.  **Investment:** Cost in millions.
3.  **Satisfaction:** A normalized score [0-1].

**Challenge:** The project is evaluated strictly on the **Satisfaction** accuracy with a tolerance of `0.05`.

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

#### 5. Output & Submission
The script generates a final output file named **`submission_xgb_mlp_optimized.csv`**.
- This CSV contains the predicted KPIs for the test set.
- The file is intended to be committed to the evaluation platform, where an external software calculates the final score against the ground truth.

## ⚠️ Notes

- **Performance:** The model achieves a Cross-Validation score of ~0.757.
- **Resource Usage:** The script is optimized for CPU execution (`n_jobs=-1`) and completes in under 15 minutes.
- **Configuration:** Ensure the `DATA_DIR` variable in the script points to your local data folder.
