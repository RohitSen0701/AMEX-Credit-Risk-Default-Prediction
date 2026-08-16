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
 
## Methodology
 
The notebook follows a structured, phase-based workflow:
 
**1. Development Sample Creation**
Streamed the 5.5M-row parquet file in batches to profile it without loading it fully into memory, then drew a stratified 50,000-customer sample for development, validating that class balance and statement-history distribution matched the full population.
 
**2. Exploratory Data Analysis (EDA)**
- Data quality audit (dtypes, duplicates, memory footprint)
- Missingness analysis by feature and by customer history
- Univariate distributions of key financial variables
- Feature-vs-target relationships (bivariate default-rate analysis)
- Temporal / customer-history analysis (statement cadence, history length, ~monthly statement gaps)
**3. Feature Engineering**
Converted longitudinal (multi-row-per-customer) data into a single feature vector per customer for 10 key longitudinal indicators (`P_2, B_1, B_2, B_9, D_42, D_48, D_61, D_44, D_55, D_75`):
- **Latest value** — most recent statement value
- **Historical statistics** — mean, median, min, max across full history
- **First-to-latest change** — trend/direction of movement
- **Historical volatility** — standard deviation across history
- **Recent 3-statement average** — short-term behavior
- **Recent-vs-historical delta** — whether recent behavior deviates from the customer's baseline
- **Missingness/history features** — % missing, valid observation count (missingness itself carries signal for risk)
**4. Preprocessing**
- Dropped features with >50% missingness (e.g., `D_42`)
- Customer-level stratified train/validation split (80/20)
- Median imputation (fit on training data only, applied to both sets)
- Removed constant and near-constant (>99.5% single value) features
- Correlation filtering — removed one of each highly correlated (**r > 0.95**) feature pair, reducing ~100 candidate features down to a clean, non-redundant set
- Balanced class weighting to counter the ~3:1 class imbalance
**5. Model Development & Comparison**
Trained and evaluated four models on an identical feature set and validation split:
Logistic Regression → Random Forest → XGBoost → LightGBM
 
**6. Hyperparameter Tuning**
`RandomizedSearchCV` (3-fold CV, 20 candidate combinations, optimized for PR-AUC) over depth, learning rate, subsampling, and regularization parameters for the top-performing model (XGBoost).
 
**7. Threshold Optimization**
Swept the classification threshold from 0.20–0.80 and selected the threshold that maximizes precision subject to a **recall ≥ 93% business constraint** — reflecting the real-world cost asymmetry of missing a defaulter.
 
**8. Model Interpretation**
- XGBoost built-in feature importance (gain-based)
- SHAP (TreeExplainer) global summary plots to explain *direction* and *magnitude* of each feature's impact on individual predictions — the analysis a credit risk stakeholder would actually want to see.
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
 
## Results
 
All four baseline models were evaluated on the same held-out 10,000-customer validation set, using the same selected feature set and a 0.50 classification threshold:
 
| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.9385 | 0.8328 | 0.8922 | 0.6663 | 0.7629 |
| Random Forest | 0.9402 | 0.8361 | 0.8446 | 0.7239 | 0.7796 |
| XGBoost (baseline) | 0.9430 | 0.8469 | 0.8910 | 0.6826 | 0.7730 |
| **LightGBM** (baseline) | 0.9427 | **0.8466** | **0.8985** | 0.6790 | 0.7735 |
 
Gradient-boosted trees (XGBoost, LightGBM) consistently outperformed the linear and bagged-tree baselines, confirming non-linear feature interactions are important for this problem.
 
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
