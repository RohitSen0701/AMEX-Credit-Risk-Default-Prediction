# AMEX Credit Risk Default Prediction
 
Predicting credit card default risk from transactional & behavioral customer data using the American Express (AMEX) Kaggle dataset — 
from raw statement-level records to a tuned, interpretable XGBoost model.

---
 
## Description
 
Credit card issuers lose billions of dollars every year to customer default. This project builds an end-to-end machine learning pipeline that predicts the probability of a customer defaulting on their credit card balance, using **5.5M+ monthly statement records for 458K+ customers** from the American Express Kaggle competition dataset.
 
The project goes beyond just training a model — it covers realistic industry constraints: severe class imbalance, heavy missingness, longitudinal (time-series-per-customer) feature engineering, correlation-based feature selection, business-driven threshold tuning, and SHAP-based interpretability so the model's decisions can be explained to non-technical stakeholders.
 
**Final model: Tuned XGBoost — ROC-AUC 0.943 | PR-AUC 0.847 | Recall 93.1% at a business-chosen decision threshold.**
 
---
 
## Table of Contents
 
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Objective](#objective)
- [Tech Stack](#tech-stack)
- [Results](#results)
- [Best Model](#best-model)
- [Installation and Usage](#installation-and-usage)
- [Future Improvements](#future-improvements)
- [Author](#author)
---
 
## Project Overview
 
Each customer in the dataset has up to 13 monthly credit card statements, each with ~190 anonymized financial, behavioral, and risk features (spend, balance, payment, delinquency, and risk variables). The label (`target`) marks whether the customer defaulted within 120 days of their last statement.
 
This is fundamentally a **longitudinal classification problem** — the raw data is one row per customer per month, but the prediction needs to happen at the customer level. The bulk of the engineering effort in this project goes into collapsing each customer's statement history into a single, information-rich feature vector before any model ever sees the data.
 
## Business Problem
 
> *"Which customers are likely to default on their next payment, and can we flag them early enough to intervene — without flooding the risk team with false alarms?"*
 
For a credit issuer, this is a cost-asymmetric problem:
- **Missing an actual defaulter (false negative)** → direct financial loss (unpaid balance, write-off).
- **Flagging a good customer as high-risk (false positive)** → unnecessary friction, credit-limit cuts, or lost customer goodwill.
Because false negatives are far costlier than false positives in this setting, the project deliberately optimizes for **high recall on the default class**, then tunes the decision threshold to keep precision as high as possible while still catching the large majority of true defaulters — a workflow that mirrors how a real risk team would calibrate a scorecard.
 
## Dataset
 
- **Source:** [American Express – Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction) (Kaggle competition dataset, parquet format)
- **Raw size:** 5,531,451 statement rows × 192 columns, 458,913 unique customers, 396 unique statement dates
- **Structure:** Monthly statements per customer (up to 13 months of history), each row = one customer-month
- **Feature groups** (anonymized for confidentiality by AMEX):
  - `D_*` — Delinquency variables
  - `S_*` — Spend variables
  - `P_*` — Payment variables
  - `B_*` — Balance variables
  - `R_*` — Risk variables
- **Target:** Binary flag — 1 if the customer defaulted, 0 otherwise
- **Class balance (customer-level):** ~74.6% non-default vs. ~25.4% default — moderately imbalanced
**Development sample used in this project:** To keep iteration fast on limited compute, a stratified random sample of **50,000 customers** (~603K statement rows) was drawn from the full dataset and used for all EDA, feature engineering, and modeling. Customer-level target proportions in the sample (74.6% / 25.4%) closely match the full population, confirming the sample is representative.
 
## Objective
 
Build a customer-level binary classifier that:
1. Accurately ranks customers by default risk (maximize ROC-AUC / PR-AUC)
2. Meets a minimum recall target (≥93%) so very few true defaulters are missed
3. Remains interpretable enough to explain individual risk scores to a non-technical audience (SHAP)
4. Follows a leakage-free, production-realistic workflow (train-only fitting for imputers/scalers, customer-level splits, no target leakage across statements)


## Tech Stack
 
| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data handling | pandas, NumPy, PyArrow (parquet, batched streaming reads) |
| Modeling | scikit-learn, XGBoost, LightGBM |
| Model tuning | scikit-learn `RandomizedSearchCV` |
| Interpretability | SHAP |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook / Kaggle Kernels |

## 🔧 Methodology

### 1. Data Understanding & EDA

The analysis included:

- Dataset structure and data-type analysis
- Target distribution analysis
- Missing-value analysis
- Customer-level observation analysis
- Distribution and outlier analysis
- Temporal/transaction-history exploration
- Feature-level statistical exploration

---

### 2. Data Preprocessing

The preprocessing pipeline included:

- Missing-value treatment
- Categorical/date/identifier handling
- Customer-level aggregation
- Feature validation
- Removal of unsuitable features

Customer identifiers and date-related fields were not directly used as predictive numerical features.

---

### 3. Feature Engineering

Longitudinal customer information was transformed into customer-level predictive features, including:

- Latest values
- Historical mean
- Historical median
- Historical minimum and maximum
- Historical standard deviation
- Recent-period averages
- Change from first to latest observation
- Recent vs. historical behavior
- Missingness indicators
- Valid observation counts

This allowed the models to capture both the customer's current financial state and their historical behavior.

---

### 4. Feature Selection

Feature selection was performed to reduce redundancy and improve model efficiency.

The process included:

- Removing constant features
- Removing near-constant features
- Correlation-based filtering

Highly correlated feature pairs were reviewed using a **0.95 correlation threshold**.

After feature selection:

> **70 features remained for model development.**

---

### 5. Train–Validation Split

The customer-level dataset was divided into:

- **Training set:** 40,000 customers
- **Validation set:** 10,000 customers

The split was performed at the customer level to avoid using the same customer's observations across training and validation.

---

### 6. Class Imbalance

The training data contained:

- Class 0: **74.58%**
- Class 1: **25.42%**

Balanced class weighting / positive-class weighting was incorporated into the model development process to account for the imbalance.

---

 
## Results
 
All four baseline models were evaluated on the same held-out 10,000-customer validation set, using the same selected feature set and a 0.50 classification threshold:
 
| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.9385 | 0.8328 | 0.8922 | 0.6663 | 0.7629 |
| Random Forest | 0.9402 | 0.8361 | 0.8446 | 0.7239 | 0.7796 |
| XGBoost (baseline) | 0.9430 | 0.8469 | 0.8910 | 0.6826 | 0.7730 |
| **LightGBM** (baseline) | 0.9427 | **0.8466** | **0.8985** | 0.6790 | 0.7735 |
 
Gradient-boosted trees (XGBoost, LightGBM) consistently outperformed the linear and bagged-tree baselines, confirming non-linear feature interactions are important for this problem.

---

## 🔍 Model Interpretation

### XGBoost Feature Importance

The most important features included:

1. `P_2_latest`
2. `B_1_latest`
3. `P_2_hist_mean`
4. `B_2_latest`
5. `B_9_latest`
6. `D_48_latest`
7. `D_44_hist_mean`
8. `P_2_hist_valid_count`

`P_2_latest` was the dominant feature in the XGBoost model.

---

### SHAP Analysis

SHAP was used to understand both feature importance and the direction of each feature's contribution to model predictions.

The analysis indicated that:

- Lower `P_2_latest` values strongly pushed predictions toward higher default probability.
- Lower historical `P_2` values also tended to increase predicted default risk.
- Several recent and historical behavioral variables contributed to the model's predictions.
- The model incorporated both current customer state and historical behavioral patterns.

SHAP explanations represent **model associations and contributions, not causal relationships**.

---

 
## Best Result
 
**Tuned XGBoost**, selected via `RandomizedSearchCV` (optimizing PR-AUC), with the decision threshold recalibrated to hit a ≥93% recall target:
 
| Metric | Score |
|---|---|
| ROC-AUC | **0.9434** |
| PR-AUC | **0.8473** |
| Recall (default class) | **93.15%** |
| Precision (default class) | 64.07% |
| F1-score (default class) | 0.7592 |
| Decision threshold | 0.40 |
 
At this threshold, the model correctly identifies **2,368 of 2,542** true defaulters in the validation set (only 174 missed), while flagging 1,328 non-defaulters for extra review — a deliberate, business-driven trade-off that prioritizes catching risk over minimizing false alarms.
 
**Top predictive drivers** (by XGBoost gain importance & confirmed by SHAP): the customer's most recent `P_2` risk score, historical average `P_2`, latest `B_1`/`B_2`/`B_9` balance features, and `D_48`/`D_44` delinquency indicators — consistent with how a real underwriting model would weight recency and risk-score trend most heavily.

## 💡 Key Findings

- Customer-level feature engineering substantially transformed the longitudinal transaction data into a modeling-ready dataset.
- Current and historical `P_2` behavior was particularly influential in the final model.
- XGBoost provided strong predictive performance compared with the baseline models.
- Threshold optimization increased recall from **90.28% at the default 0.50 threshold to 93.15% at 0.40**.
- The final model prioritizes identifying potential defaulters while accepting a higher number of false positives.

---

## ⚠️ Limitations

- Model development was performed using a **50,000-customer sample** rather than the complete customer population.
- The reported validation performance comes from the 10,000-customer validation set within this development sample.
- The final threshold was selected using validation performance; therefore, an additional untouched test set would provide a stronger final generalization estimate.
- LightGBM was evaluated as a baseline but was not further tuned due to computational cost.

---
 
## Installation and Usage
 
**1. Clone the repository**
```bash
git clone https://github.com/RohitSen0701/AMEX-Credit-Risk-Default-Prediction.git
cd AMEX-Credit-Risk-Default-Prediction
```
 
**2. Create an environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
 
`requirements.txt`
```
pandas
numpy
pyarrow
scikit-learn
xgboost
lightgbm
shap
matplotlib
jupyter
```
 
**3. Get the data**
Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/amex-default-prediction/data) (requires a free Kaggle account) and place the parquet files under `data/`.
 
**4. Run the notebook**
```bash
jupyter notebook amex-default-prediction.ipynb
```
Run cells top to bottom — the notebook is organized into clearly labeled phases (sampling → EDA → feature engineering → modeling → tuning → interpretation) so you can jump to any section.
 
## Future Improvements
 
- Engineer features from the remaining ~180 statement-level columns beyond the 10 core longitudinal features used here
- Add time-series-specific features (e.g., trend slopes, exponentially weighted moving averages, month-over-month deltas across all statements rather than
  first/last only)
- Train on the full 458K-customer population instead of the 50K development sample, using out-of-core / GPU training
- Try CatBoost and neural network (TabNet / MLP) baselines for comparison
- Deploy the final model behind a lightweight REST API (FastAPI) with a simple scoring/monitoring dashboard
- Add model-drift monitoring and periodic retraining logic for a production setting
- Package feature engineering into reusable, testable pipeline classes (`sklearn.Pipeline` / custom transformers) instead of notebook cells

## 👨‍💻 Author

**Rohit Sen**

M.Sc. Statistics, University of Delhi

LinkedIn:
https://www.linkedin.com/in/rohit-sen-50188b320/

GitHub:
https://github.com/RohitSen0701
