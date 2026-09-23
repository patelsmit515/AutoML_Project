# Product Requirements Document — AutoML Project

## 1. Product Overview

The AutoML Project is a user-friendly automated machine learning application for **tabular datasets**.

The long-term goal is to allow a user with little or no machine-learning programming experience to:

1. Upload a dataset.
2. Select the target column.
3. Choose classification or regression.
4. Configure or accept sensible preprocessing defaults.
5. Select one or more machine-learning models.
6. Train the models automatically.
7. Evaluate and compare their results.
8. Understand what the metrics and models mean.
9. Optionally optimize model hyperparameters.
10. Identify a strong-performing model according to a metric chosen by the user.
11. View useful visualizations and model information.
12. Eventually export useful results or trained models.

The application is intended to automate repetitive ML workflow steps while keeping the process understandable and transparent.

The project is being developed **backend-first**. The core ML engine is being built and tested before the interactive Streamlit interface is added.

---

# 2. Product Philosophy

The application should not behave like a completely opaque "press a button and trust the answer" AutoML system.

The user should be able to understand:

* What the application did.
* Why preprocessing was applied.
* Which models were trained.
* What each metric means.
* Why a model was selected according to the chosen metric.
* What limitations the results have.
* Whether the dataset itself has potential problems.

The application should automate the technical work while still giving the user meaningful control and explanations.

---

# 3. Problem Statement

Many students, analysts, and non-specialist users have tabular data but do not know how to turn it into a machine-learning model.

A typical workflow requires knowledge of:

* Data validation.
* Feature and target selection.
* Missing-value handling.
* Categorical encoding.
* Feature scaling.
* Train/test splitting.
* Model selection.
* Model evaluation.
* Hyperparameter tuning.
* Result interpretation.

The project aims to combine these steps into one understandable workflow.

The goal is not to hide machine learning from the user, but to make the workflow easier to use and understand.

---

# 4. Target Users

## Primary Users

* Students learning machine learning.
* Beginners who understand their dataset but have limited ML programming experience.
* Analysts who want to create quick baseline models.
* Users who want to compare several traditional ML algorithms without writing the complete training pipeline themselves.

## Secondary Users

* Intermediate data scientists who want a quick baseline comparison before building a custom solution.
* Developers or learners who want to inspect how an automated ML workflow is constructed.

---

# 5. Product Scope

## 5.1 Currently Supported

The current backend is designed for:

* Tabular CSV datasets.
* Classification problems.
* Regression problems.
* Numerical features.
* Categorical features represented by supported pandas data types.
* Missing-value handling.
* Feature scaling.
* Categorical encoding.
* Multiple traditional machine-learning algorithms.
* Train/test evaluation.
* Metric-based model comparison.

## 5.2 Currently Out of Scope

The current product does not aim to support:

* Deep-learning/neural-network AutoML.
* Image datasets.
* Audio datasets.
* General unstructured text datasets.
* Specialized time-series forecasting.
* Production model serving.
* Full MLOps infrastructure.
* Distributed training.
* Extremely large datasets requiring distributed processing.

These may be considered later if the project grows beyond its original scope.

---

# 6. Core User Workflow

The intended user workflow is:

```text
Upload Dataset
      ↓
Validate Dataset
      ↓
Select Target Column
      ↓
Select Problem Type
      ↓
Configure Preprocessing
      ↓
Configure Training
      ↓
Train Models
      ↓
Evaluate Models
      ↓
Compare Results
      ↓
Select Evaluation Metric
      ↓
Identify Best-Performing Model
      ↓
Explain Results
      ↓
(Optional Future)
Optimize Model
      ↓
(Optional Future)
Visualize / Export Results
```

The backend should support this workflow independently of the user interface.

---

# 7. Functional Requirements

## FR-01 — Load Dataset

The system must be able to load a CSV file into a pandas DataFrame.

### Current Status

Implemented.

### Module

`src/data_loader.py`

Current responsibility:

```text
CSV → pandas DataFrame
```

The loader should remain simple and should not contain preprocessing or model-training logic.

---

# 8. Dataset Validation

## FR-02 — Validate Dataset

The system must perform basic validation before model training.

Current validation checks include:

* Empty dataset.
* Target column existence.
* Very small datasets.
* Duplicate rows.
* Missing values.

Validation separates problems into:

### Errors

Problems that prevent the workflow from continuing.

Example:

```text
The target column 'Purchased' is not present in the dataset.
```

### Warnings

Problems that do not necessarily prevent training but should be shown to the user.

Examples:

```text
The dataset has fewer than 10 rows.
```

```text
Dataset contains duplicate rows.
```

```text
Dataset contains missing values.
```

### Current Status

Implemented.

### Module

`src/validation.py`

Advanced validation will be added gradually rather than attempting to implement every possible dataset problem immediately.

---

# 9. Target and Problem Type

## FR-03 — Target Selection

The user must be able to select the column that the model should predict.

The selected target must not be included as an input feature.

This separation is important to prevent target leakage.

### Status

Backend supported.

UI implementation is pending.

---

## FR-04 — Problem Type

The system must support:

* Classification.
* Regression.

The current design allows the problem type to be explicitly selected.

Automatic problem-type detection may be considered later.

### Current Default Configuration

Defined in:

`src/config.py`

```text
Classification
Regression
```

---

# 10. Feature Identification

## FR-05 — Identify Numerical and Categorical Features

The system should automatically separate feature columns into:

* Numerical columns.
* Categorical columns.

The target column must be excluded before identification.

### Current Status

Implemented.

### Module

`src/preprocessing.py`

Current implementation uses pandas data types to identify supported numerical and categorical columns.

---

# 11. Preprocessing

## FR-06 — Missing-Value Handling

The system should provide configurable missing-value strategies.

### Numerical Data

Supported strategies through the preprocessing system include:

* Mean.
* Median.
* Most frequent.
* Other strategies can be added if required.

### Categorical Data

Supported strategies include:

* Most frequent.
* Constant.

The default configuration currently uses:

```text
Numerical → Median
Categorical → Most Frequent
```

### Important Design Requirement

Preprocessing should be performed inside sklearn pipelines wherever possible.

This helps prevent data leakage because preprocessing parameters are learned from the training data rather than from the complete dataset.

### Current Status

Implemented.

---

# 12. Feature Scaling

## FR-07 — Scaling

The user should be able to choose:

* Standard Scaling.
* Min-Max Scaling.
* Robust Scaling.
* No Scaling.

Current default:

```text
Standard Scaling
```

### Current Status

Implemented in:

`src/preprocessing.py`

---

# 13. Categorical Encoding

## FR-08 — Encoding

The user should be able to choose:

* One-Hot Encoding.
* Ordinal Encoding.

Current default:

```text
One-Hot Encoding
```

Unknown categorical values should be handled safely.

The current implementation uses:

```text
OneHotEncoder(handle_unknown="ignore")
```

for one-hot encoding and an unknown-value strategy for ordinal encoding.

### Current Status

Implemented.

---

# 14. Model Selection

## FR-09 — Classification Models

The system currently supports:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. KNN
5. SVM
6. Gradient Boosting
7. XGBoost

### Status

Implemented.

---

## FR-10 — Regression Models

The system currently supports:

1. Linear Regression
2. Decision Tree
3. Random Forest
4. KNN
5. SVR
6. Gradient Boosting
7. XGBoost

### Status

Implemented.

---

# 15. Training

## FR-11 — Train/Test Split

The system must split the dataset into training and testing portions.

The test size should be configurable.

Current supported options:

```text
10%
20%
25%
30%
40%
```

Current default:

```text
20%
```

A random state should be used to make results reproducible.

Current default:

```text
42
```

### Status

Implemented.

### Module

`src/training.py`

---

# 16. ML Pipeline

## FR-12 — Preprocessing + Model Pipeline

Each model should be trained through an sklearn Pipeline.

Conceptually:

```text
Raw Features
     ↓
Preprocessor
     ↓
Imputation
     ↓
Scaling / Encoding
     ↓
Model
```

This means each model receives a consistent preprocessing workflow.

It also keeps preprocessing and model training connected and reduces the risk of data leakage.

### Status

Implemented.

---

# 17. Model Evaluation

## FR-13 — Classification Metrics

The system currently calculates:

* Accuracy.
* Precision.
* Recall.
* F1 Score.

The metric definitions and explanations are stored in:

`src/config.py`

### Current Limitation

The current implementation is primarily designed around binary classification metrics.

Multiclass classification support should be added later with appropriate averaging strategies.

---

## FR-14 — Regression Metrics

The system currently calculates:

* MAE.
* MSE.
* RMSE.
* R² Score.

The metric definitions and explanations are stored in:

`src/config.py`

---

# 18. Model Comparison

## FR-15 — Compare Models

The system should produce a structured result for each trained model.

Conceptually:

```text
Model                  Accuracy    Precision    Recall    F1
----------------------------------------------------------------
Logistic Regression      ...
Decision Tree            ...
Random Forest             ...
KNN                       ...
SVM                       ...
Gradient Boosting        ...
XGBoost                   ...
```

For regression:

```text
Model                  MAE      MSE      RMSE      R²
----------------------------------------------------------------
Linear Regression        ...
Decision Tree             ...
Random Forest             ...
...
```

### Status

Backend evaluation and comparison are implemented through the existing testing workflow.

The final user-facing comparison interface is not yet built.

---

# 19. Best Model Selection

## FR-16 — Metric-Based Model Selection

The user should be able to select the metric used to identify the top-performing model.

For classification:

```text
Accuracy
Precision
Recall
F1 Score
```

For regression:

```text
MAE
MSE
RMSE
R² Score
```

The system must understand whether a metric is:

```text
Higher is better
```

or:

```text
Lower is better
```

For example:

```text
Accuracy → higher is better
R² → higher is better

MAE → lower is better
MSE → lower is better
RMSE → lower is better
```

This direction is stored centrally in `config.py`.

### Status

Implemented.

### Module

`src/evaluation.py`

---

# 20. Model and Metric Explanations

## FR-17 — Explain Results

The application should not only display numbers.

It should explain:

### Metrics

For each metric, the application should provide:

* What the metric measures.
* Whether higher or lower is better.
* When the metric is useful.
* Important limitations or warnings.

### Models

For each model, the application should provide understandable information about:

* What type of model it is.
* How it generally works.
* Important characteristics.
* Relevant strengths/limitations or trade-offs.

The metadata for these explanations is centralized in `config.py`.

### Current Status

The information has been authored in the configuration layer.

The information has **not yet been connected to the user interface**.

---

# 21. Training Configuration

The application should eventually allow the user to control:

### Test Size

```text
10%
20%
25%
30%
40%
```

### Cross-Validation Folds

```text
3
5
10
```

### Model Selection

```text
All applicable models
```

or selected models.

### Preprocessing

```text
Scaling
Encoding
Numerical imputation
Categorical imputation
```

The configuration system is already being prepared for these options.

---

# 22. Cross-Validation

Cross-validation is part of the planned training configuration.

Current configuration supports:

```text
3 folds
5 folds
10 folds
```

### Current Status

The configuration exists, but cross-validation is not yet fully integrated into the main training workflow.

It should be integrated after the basic orchestration layer is complete.

Cross-validation should not be added merely for the sake of adding another feature; it should be incorporated where it provides meaningful model comparison or hyperparameter optimization.

---

# 23. Hyperparameter Optimization

## FR-18 — Hyperparameter Search

The project is planned to support:

* Grid Search.
* Randomized Search.

Current configuration already contains:

```text
Grid Search
Randomized Search
```

and a default iteration count for randomized search.

### Current Status

Configuration exists.

Training integration has not yet been implemented.

### Planned Purpose

The system should eventually allow models to be improved by testing different hyperparameter combinations instead of relying only on default model parameters.

The application should clearly distinguish:

```text
Baseline model
```

from:

```text
Optimized model
```

so the user can see whether optimization actually improved the result.

---

# 24. Result Visualization

## FR-19 — Visualize Results

The future application should provide useful visualizations rather than displaying only tables.

Potential visualizations include:

* Model metric comparison.
* Actual vs. predicted values for regression.
* Confusion matrix for classification.
* Feature importance where supported.
* Other model-specific diagnostic plots where appropriate.

Visualization should be introduced after the core workflow is stable.

### Status

Not implemented.

---

# 25. Export

## FR-20 — Export Results

Future versions may allow users to export:

* Predictions.
* Model comparison results.
* Evaluation metrics.
* Trained preprocessing + model pipelines.

### Status

Not implemented.

This feature should not be prioritized until the core workflow and UI are stable.

---

# 26. User Interface

## FR-21 — Interactive UI

The planned interface will use Streamlit.

The UI should eventually provide a simple flow such as:

```text
1. Upload Dataset
2. Inspect Dataset
3. Select Target
4. Select Problem Type
5. Configure Preprocessing
6. Select Models
7. Configure Training
8. Train
9. Compare Results
10. Inspect Best Model
11. View Explanations
12. Optimize Model (future)
13. Visualize Results
14. Export Results (future)
```

The UI should call the backend rather than duplicate machine-learning logic.

### Important Architecture Rule

Streamlit should primarily handle:

* User input.
* Display.
* Navigation.
* Interaction.

The ML logic should remain in the `src/` backend modules.

### Current Status

Not implemented.

---

# 27. Current Backend Architecture

The current backend contains:

```text
src/
├── config.py
├── data_loader.py
├── validation.py
├── preprocessing.py
├── models.py
├── training.py
├── evaluation.py
└── __init__.py
```

Responsibilities:

```text
config.py
    ↓
Central configuration and metadata

data_loader.py
    ↓
Load dataset

validation.py
    ↓
Check dataset problems

preprocessing.py
    ↓
Identify columns
Build preprocessing pipelines

models.py
    ↓
Provide ML model objects

training.py
    ↓
Split data
Build pipelines
Train models
Generate predictions

evaluation.py
    ↓
Calculate metrics
Select model
```

---

# 28. Planned Orchestration Layer

The next major backend component should be:

```text
src/automl.py
```

Its purpose is to connect the existing modules into one complete workflow.

The orchestration layer should approximately perform:

```text
Load / receive dataset
        ↓
Validate
        ↓
Separate X and y
        ↓
Identify columns
        ↓
Create preprocessor
        ↓
Select models
        ↓
Split data
        ↓
Create model pipelines
        ↓
Train models
        ↓
Predict
        ↓
Evaluate
        ↓
Create structured comparison results
        ↓
Select best model
```

The orchestration layer should **reuse existing functions**.

It should not duplicate preprocessing, model creation, training, or evaluation logic.

This is the next development milestone after the current core modules.

---

# 29. Development Stages

The project will be developed incrementally.

## Stage 1 — Core Configuration

Status: COMPLETE

Created centralized configuration for:

* Models.
* Metrics.
* Defaults.
* Test-size options.
* Cross-validation options.
* Optimization methods.
* Explanations and metadata.

---

## Stage 2 — Data Loading and Validation

Status: COMPLETE

Implemented:

* CSV loading.
* Basic dataset validation.
* Error/warning separation.

---

## Stage 3 — Preprocessing

Status: COMPLETE

Implemented:

* Numerical/categorical identification.
* Missing-value handling.
* Scaling.
* Encoding.
* sklearn `ColumnTransformer`.
* sklearn pipelines.

---

## Stage 4 — Models

Status: COMPLETE

Implemented:

* 7 classification models.
* 7 regression models.

Models are provided as fresh estimator instances.

---

## Stage 5 — Training

Status: COMPLETE

Implemented:

* Train/test splitting.
* Reproducibility through random state.
* Model pipelines.
* Model training.
* Predictions.

---

## Stage 6 — Evaluation

Status: COMPLETE

Implemented:

* Classification metrics.
* Regression metrics.
* Metric direction handling.
* Best-model selection.

---

## Stage 7 — Smoke Testing

Status: COMPLETE

The backend has been tested with small sample datasets.

### Classification Test

Dataset:

```text
data/test.csv
```

Features:

```text
Age
Salary
City
```

Target:

```text
Purchased
```

Random Forest training completed successfully.

The smoke test produced:

```text
Accuracy: 1.0000
Precision: 1.0000
Recall: 1.0000
F1: 1.0000
```

These values should **not** be interpreted as meaningful model performance because the dataset is extremely small and the test set contained only one observation.

The purpose of the test was to confirm that the complete classification pipeline works.

### Regression Test

Dataset:

```text
data/regression_test.csv
```

Target:

```text
Salary
```

All seven regression models were trained and evaluated successfully.

The test selected Linear Regression according to R².

Again, the dataset is intentionally tiny and highly structured, so these results are only workflow/smoke-test results and are not representative of real-world model performance.

---

# 30. Current Development Status

The project is currently at:

```text
CORE ML ENGINE → COMPLETE
ORCHESTRATION → NEXT
STREAMLIT UI → FUTURE
HYPERPARAMETER OPTIMIZATION → FUTURE
VISUALIZATION → FUTURE
EXPORT → FUTURE
```

The most important next implementation is:

```text
src/automl.py
```

The purpose is to turn the currently separate working components into one reusable AutoML workflow.

---

# 31. Development Method

This project must be developed incrementally.

The preferred working method is:

```text
Understand
    ↓
Make one small change
    ↓
Run/test it
    ↓
Inspect the result
    ↓
Fix if necessary
    ↓
Only then continue
```

Do not implement several unrelated features at once.

Do not rewrite working modules simply to make them look different.

Before changing an existing module:

1. Understand its current responsibility.
2. Check how other modules use it.
3. Make the smallest necessary change.
4. Run the relevant test.
5. Confirm the output.
6. Continue to the next component.

The project should remain understandable to the person building it.

---

# 32. Design Principles

## Separation of Responsibilities

Each module should have one clear responsibility.

For example:

```text
models.py
```

should provide models.

It should not perform evaluation.

```text
evaluation.py
```

should evaluate models.

It should not load datasets.

```text
automl.py
```

should coordinate the workflow.

It should not contain duplicated implementations of every underlying operation.

---

## Centralized Configuration

User-facing options and shared defaults should remain centralized in:

```text
config.py
```

This prevents different parts of the application from using inconsistent values.

---

## Reusable Backend

The backend should work independently of Streamlit.

This allows:

* Testing through Python scripts.
* Future command-line use.
* Future alternative interfaces.
* Easier debugging.
* Easier maintenance.

---

## Avoid Premature Complexity

Do not add advanced features before the basic workflow is reliable.

The priority is:

```text
Reliable core
    ↓
Reusable orchestration
    ↓
User interface
    ↓
Advanced ML features
```

---

# 33. Known Limitations

Current limitations include:

### Small Test Datasets

The included datasets are smoke-test datasets only.

They should not be used to demonstrate realistic model performance.

### Binary Classification Metrics

Current classification evaluation assumes binary classification behavior.

Multiclass support is a future improvement.

### Data Types

Current feature identification primarily handles standard numerical and object/categorical columns.

More advanced data-type handling may be added later.

### Hyperparameter Optimization

The configuration exists, but optimization is not yet connected to training.

### Cross-Validation

Configuration exists, but the complete workflow does not yet use it.

### High-Cardinality Categories

One-hot encoding can create a very large number of features when categorical columns contain many unique values.

A safeguard or warning may be added later.

### Large Datasets

The current application is designed around in-memory pandas workflows.

Large-dataset handling is not currently a priority.

---

# 34. Future Roadmap

The planned development order is approximately:

## Phase A — Backend Completion

1. Build `src/automl.py`.
2. Test the complete end-to-end backend workflow.
3. Return structured results suitable for a UI.

## Phase B — Backend Robustness

4. Clean and finalize requirements.
5. Improve validation.
6. Integrate cross-validation where appropriate.
7. Add multiclass classification support.
8. Handle edge cases discovered during testing.

## Phase C — Interactive Application

9. Build the Streamlit interface.
10. Connect UI controls to backend configuration.
11. Display validation errors/warnings.
12. Display model comparison.
13. Display metric explanations.
14. Display model explanations.

## Phase D — Advanced ML

15. Add Grid Search.
16. Add Randomized Search.
17. Compare baseline vs. optimized models.
18. Add relevant model details and parameter information.

## Phase E — Visualization and Export

19. Add result visualizations.
20. Add prediction export.
21. Add model/result export where appropriate.

The exact order may change if testing reveals a better dependency order.

---

# 35. Success Criteria

The project should eventually satisfy the following:

### Basic Success

A user can:

```text
CSV
 ↓
Target
 ↓
Problem Type
 ↓
Training
 ↓
Evaluation
 ↓
Model Comparison
```

without writing ML code.

### Transparency Success

The user can understand:

* What the selected metric means.
* Why higher/lower is better.
* What the models generally do.
* Why a model was identified according to the selected metric.
* Important limitations of the results.

### Reliability Success

The application should:

* Validate obvious dataset problems.
* Avoid unnecessary crashes.
* Keep preprocessing inside model pipelines.
* Produce reproducible results where appropriate.
* Clearly distinguish warnings from blocking errors.

### Performance Goal

For a small-to-medium dataset, the basic workflow should ideally complete within a reasonable interactive time, with the exact acceptable time depending on dataset size and selected models.

---

# 36. Important Rule for Future Development

This document describes both the **current state** and the **future plan**.

Future developers/assistants must not assume that planned features have already been implemented.

The current confirmed stopping point is:

```text
config.py          COMPLETE
data_loader.py     COMPLETE
validation.py      COMPLETE
preprocessing.py   COMPLETE
models.py          COMPLETE
training.py        COMPLETE
evaluation.py      COMPLETE
smoke tests        PASSING

NEXT:
src/automl.py
```

When continuing the project, first inspect:

1. `prd.md`
2. `architecture.md`
3. Current project tree
4. Relevant source files
5. Latest test output
6. Git status

Then continue from the actual code state rather than assuming that the roadmap has already been completed.

The development style should remain:

```text
Small change → test → understand → continue
```

The goal is to build a reliable, understandable AutoML application rather than simply adding as many features as possible.
