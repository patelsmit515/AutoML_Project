# ⚡ AutoML Studio — Tabular Machine Learning Platform

AutoML Studio is a Python-based Automated Machine Learning (AutoML) platform for tabular datasets. It provides an intuitive, portfolio-grade Streamlit frontend layered directly on top of a modular, pipeline-driven machine learning backend.

---

## 🌟 Key Features

### 1. Interactive Dataset Ingestion & EDA
### 1. Interactive Dataset Ingestion & EDA
- **CSV Dataset Upload**: Upload your own tabular CSV dataset or select from bundled benchmark datasets (`test.csv`, `iris.csv`, `regression_test.csv`, `missing_values_test.csv`, `missing_regression_test.csv`, `small_test.csv`).
- **Data Preview & Schema Inspection**: Review row counts, column counts, detected numerical and categorical feature distributions, data types, and missing value counts.
- **Automated Validation & Warnings**: Detects empty files, missing target columns, dataset sizes with fewer than 10 rows, duplicate records, and null values before execution.

### 2. Comprehensive AutoML Workflow & Preprocessing
- **Problem Type Support**: Full support for **Binary Classification**, **Multiclass Classification** (e.g. Iris), and **Regression** tasks.
- **Configurable Preprocessing Controls**:
  - *Missing Value Imputation*:
    - **Numerical**: Median (default), Mean, Most Frequent (`SimpleImputer`).
    - **Categorical**: Most Frequent (default), Constant Value with custom replacement text (e.g. `"Unknown"`).
  - *Feature Scaling Strategies*: Standard Scaling (`StandardScaler`), Min-Max Scaling (`MinMaxScaler`), Robust Scaling (`RobustScaler`), No Scaling (`passthrough`).
  - *Categorical Encoding Strategies*: One-Hot Encoding (`OneHotEncoder`), Ordinal Encoding (`OrdinalEncoder`).
  - *Leakage-Free Pipelines*: Missing value imputation, scaling, and encoding are fitted exclusively on training splits inside scikit-learn `Pipeline` and `ColumnTransformer` instances.
- **Multi-Algorithm Model Training**:
  - *Classification*: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gradient Boosting, XGBoost.
  - *Regression*: Linear Regression, Decision Tree, Random Forest, KNN, SVR, Gradient Boosting, XGBoost.
- **Metric-Driven Best Model Selection**: Rank algorithms according to user-selected metrics:
  - *Classification Metrics*: Accuracy, Precision, Recall, F1 Score (with automatic weighted averaging for multiclass).
  - *Regression Metrics*: MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error), R² Score.

### 3. Cross-Validation & Hyperparameter Tuning
- **K-Fold / Stratified K-Fold Validation**: Evaluates model generalizability across 3, 5, or 10 folds with mean score and standard deviation reporting.
- **Hyperparameter Optimization**:
  - **Grid Search (`GridSearchCV`)**: Exhaustive search over curated parameter spaces.
  - **Randomized Search (`RandomizedSearchCV`)**: Configurable random sampling (`n_iter`) over parameter grids.

### 4. Portfolio-Grade Results Dashboard
- **Top Performing Model Hero Card**: Highlights the winning algorithm, test score, and cross-validation performance.
- **Interactive Leaderboard**: Unified model comparison table featuring CV metrics, test metrics, tuning status, and optimal hyperparameters.
- **Visual Analytics**: Matplotlib & Seaborn charts for model comparison and cross-validation variance.
- **Educational Knowledge Base**: In-app guidance explaining when to use each metric and the strengths/trade-offs of each algorithm.

---

## 📁 Repository Structure

```text
AutoML_Project/
├── app.py                      # Streamlit Interactive Web Application
├── requirements.txt            # Python Dependencies
├── architecture.md             # System Architecture Specification
├── prd.md                      # Product Requirements Document
├── README.md                   # Project Documentation & Usage Guide
│
├── src/                        # AutoML Core Backend
│   ├── __init__.py
│   ├── config.py               # Centralized Configuration, Metrics & Model Info
│   ├── data_loader.py          # CSV Data Ingestion
│   ├── validation.py           # Dataset & Configuration Validation
│   ├── preprocessing.py       # Pipelines, Imputation, Scaling, Encoding
│   ├── models.py               # Estimator Factories for Classifiers & Regressors
│   ├── training.py             # Data Splitting, Pipeline Creation & Cross-Validation
│   ├── evaluation.py           # Metric Computation & Best Model Selection
│   ├── tuning.py               # GridSearchCV & RandomizedSearchCV Optimization
│   └── automl.py               # Main Orchestration Engine (`run_automl`)
│
├── data/                       # Benchmark & Smoke Test Datasets
│   ├── test.csv                # Classification benchmark
│   ├── regression_test.csv     # Regression benchmark
│   ├── missing_values_test.csv # Classification with missing values
│   ├── missing_regression_test.csv # Regression with missing values
│   └── small_test.csv          # Edge case small dataset (<10 rows)
│
└── test_*.py                   # Backend & Integration Test Suites
    ├── test_automl.py
    ├── test_automl_tuning.py
    ├── test_classification.py
    ├── test_config_validation.py
    ├── test_dataset_validation.py
    ├── test_invalid_automl.py
    ├── test_missing_regression.py
    ├── test_missing_values.py
    ├── test_preprocessing.py
    ├── test_regression.py
    ├── test_small_dataset.py
    ├── test_small_tuning.py
    ├── test_streamlit_app.py
    └── test_tuning.py
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+ (Python 3.10+ recommended)
- `pip` package manager

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/patelsmit515/AutoML_Project.git
cd AutoML_Project
pip install -r requirements.txt
```

### 3. Launch the Streamlit App
To run the interactive web application, execute:

```bash
streamlit run app.py
```

The application will start locally and open in your default browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

The repository contains a full suite of verification tests for the ML pipeline and frontend helpers:

```bash
# Run backend smoke tests
python test_dataset_validation.py
python test_config_validation.py
python test_preprocessing.py
python test_classification.py
python test_regression.py
python test_missing_values.py
python test_missing_regression.py
python test_small_dataset.py
python test_tuning.py
python test_automl_tuning.py
python test_small_tuning.py
python test_invalid_automl.py
python test_automl.py

# Run frontend integration test suite
python test_streamlit_app.py
```

---

## 📐 Architecture & Principles
1. **Backend as Source of Truth**: The core ML engine (`src/automl.py`, `src/config.py`) handles all computation, validation, modeling, evaluation, and tuning.
2. **Zero Data Leakage**: All imputation, scaling, and encoding transformations are packaged inside scikit-learn `Pipeline` and `ColumnTransformer` objects fitted exclusively on training data splits.
3. **Graceful Degradation**: If individual models encounter issues during training or cross-validation on extreme edge cases, non-fatal warnings are captured while allowing the remaining models to complete execution.
