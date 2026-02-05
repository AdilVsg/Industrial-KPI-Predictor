# Hybrid XGBoost & MLP for Industrial KPI Prediction

![Author](https://img.shields.io/badge/author-Adil%20CHOUKAIRE-blue)

This repository contains the source code for a machine learning solution designed to predict key performance indicators (KPIs) for industrial aerospace workshops. The solution was developed during a hackathon to replace time-consuming physical simulations with instant AI inference.

## 📌 Project Overview

The objective is to predict three specific targets based on roughly 76,000 anonymized workshop configuration features:

1.  **WIP (Work In Progress):** Current workload in the workshop.
2.  **Investment:** Required financial injection (millions of Euros).
3.  **Satisfaction:** A normalized score [0-1] representing customer satisfaction.

**Challenge:** The evaluation metric is strict accuracy on the *Satisfaction* target only ($|y_{pred} - y_{true}| < 0.05$). However, predicting all three targets jointly is essential to capture the physical dependencies of the factory.

## 🚀 The Solution: Hybrid Ensemble Strategy

We implemented a **weighted ensemble model** combining the robustness of Gradient Boosting with the non-linear capabilities of Neural Networks.



### 1. Differentiated Feature Selection
The dataset suffers from the curse of dimensionality (7k rows vs 76k columns). We applied a "Less is More" strategy with specific pipelines for each model:
* **For XGBoost:** Top **500 features** selected via LightGBM importance. Trees handle noise relatively well.
* **For MLP:** Top **150 features** only. Neural networks are sensitive to noise, so we restricted the input to the most critical signals.

### 2. Model Architecture
* **Model A: XGBoost (RegressorChain)**
    * Uses a `RegressorChain` to model causal dependencies: `WIP -> Investment -> Satisfaction`.
    * Objective function: Mean Absolute Error (MAE) to align with the tolerance-based metric.
    * Optimized with `tree_method='hist'` for speed.

* **Model B: Multi-Layer Perceptron (MLP)**
    * Dense architecture `(128, 64)` with ReLU activation.
    * **Critical Step:** Both inputs and targets are scaled using `StandardScaler`. This prevents the "Investment" target (millions) from crushing the gradients of the "Satisfaction" target (0-1).

### 3. Optimization
The final prediction is a weighted average of both models. The weights are dynamically optimized using the **Nelder-Mead** algorithm on Out-Of-Fold (OOF) predictions to maximize the validation score.

## 📂 Project Structure

```bash
.
├── data/                   # Dataset folder (train.csv, test.csv)
├── submission_xgb_mlp.py   # Main script (Data prep, Training, Inference)
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
