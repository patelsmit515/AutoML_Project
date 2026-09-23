# Product Requirements Document — AutoML Project

## 1. Overview
A no-code / low-code AutoML tool that lets a non-expert user upload a tabular dataset (CSV), pick a target column, and automatically train, evaluate, and compare multiple machine learning models — without writing code. The system explains its choices in plain language (metric definitions, model trade-offs) so users can make informed decisions, not just read numbers.

**Current state:** The core ML engine (data loading, validation, preprocessing, training, evaluation, model selection) is built and tested via scripts. The interactive UI (Streamlit, per `requirements.txt`) has not yet been built — this is the next milestone.

## 2. Problem Statement
Business users, students, and analysts often have a dataset and a question ("can I predict X?") but lack the ML expertise to clean data, choose an algorithm, tune it, and interpret metrics correctly. Existing AutoML tools are either too complex (require ML knowledge) or too opaque (black-box "best model" with no explanation).

## 3. Goals
- Let a user go from raw CSV → trained, evaluated models in a few clicks.
- Support both **classification** and **regression** problems automatically.
- Make results interpretable: explain what each metric and model means, not just show numbers.
- Surface data quality issues (missing values, duplicates, too few rows) before training.
- Recommend a "best model" based on a user-selected metric, with reasoning.

### Non-Goals (for current scope)
- Deep learning / neural networks.
- Unstructured data (text, images, time series with seasonality).
- Hyperparameter tuning is configured (Grid/Randomized Search options exist in config) but not yet wired into the training flow.
- Model deployment / serving as an API.

## 4. Target Users
- **Primary:** Non-technical or semi-technical analysts/students who understand their data domain but not ML internals.
- **Secondary:** Data scientists who want a fast baseline model comparison before deeper custom work.

## 5. Core User Flow
1. **Upload data** — user uploads a CSV file.
2. **Validate** — system checks for empty dataset, too few rows, duplicate rows, missing values, missing/invalid target column, and no numerical columns; shows errors (blocking) and warnings (non-blocking).
3. **Select target column & problem type** — user picks the column to predict; system (or user) determines classification vs. regression.
4. **Configure preprocessing** (optional, sensible defaults provided) — scaling method (standard/min-max/robust/none), encoding (one-hot/ordinal), missing-value strategy for numeric (mean/median/most frequent) and categorical (most frequent/constant) columns.
5. **Configure training** — test size (10–40%), cross-validation folds (3/5/10), which models to run (default: all applicable to the problem type).
6. **Train** — system splits data, builds a preprocessing + model pipeline per selected algorithm, and trains each.
7. **Evaluate & compare** — system computes metrics per model:
   - Classification: accuracy, precision, recall, F1.
   - Regression: MAE, MSE, RMSE, R².
8. **Recommend best model** — user selects an optimization metric; system highlights the top-performing model (correctly handling "lower is better" metrics like MAE/MSE/RMSE vs. "higher is better" metrics).
9. **Explain results** — for each metric and model shown, display plain-language description, when to use it, and its known limitations/warnings (all metadata already authored in `config.py`).

## 6. Functional Requirements

| # | Requirement | Status |
|---|---|---|
| 1 | Load CSV into a dataframe | ✅ Implemented (`data_loader.py`) |
| 2 | Validate dataset (empty, row count, target presence, duplicates, missing values, no numeric columns) | ✅ Implemented (`validation.py`) |
| 3 | Auto-identify numerical vs. categorical feature columns | ✅ Implemented (`preprocessing.py`) |
| 4 | Configurable imputation, scaling, encoding | ✅ Implemented (`preprocessing.py`, `config.py`) |
| 5 | Train/test split with configurable size & random state | ✅ Implemented (`training.py`) |
| 6 | Build sklearn Pipeline (preprocessor + model) | ✅ Implemented (`training.py`) |
| 7 | 7 classification models, 7 regression models | ✅ Implemented (`models.py`) |
| 8 | Compute classification & regression metrics | ✅ Implemented (`evaluation.py`) |
| 9 | Select best model by chosen metric | ✅ Implemented (`evaluation.py`) |
| 10 | Human-readable model/metric descriptions | ✅ Authored (`config.py`), not yet surfaced in UI |
| 11 | Interactive UI for the full flow above | ❌ Not built (Streamlit dependency present, no app file yet) |
| 12 | Hyperparameter optimization (Grid/Randomized Search) | ⚠️ Config options defined, not implemented in training logic |
| 13 | Visualizations of results (matplotlib/seaborn in deps) | ❌ Not built |
| 14 | Export trained model / predictions | ❌ Not built |

## 7. Success Metrics
- User can complete upload → best-model result in under 2 minutes for a small-to-medium dataset.
- Zero unhandled crashes on malformed CSVs (validation catches issues first).
- Users report understanding *why* a model/metric was chosen (qualitative feedback / usability testing).

## 8. Open Questions
- Should problem type (classification vs. regression) be auto-detected from the target column's data type/cardinality, or always user-selected?
- Should the app support datasets beyond in-memory CSV size (e.g., large files, chunked loading)?
- Is model/pipeline persistence (saving a trained pipeline for reuse) in scope for v1?
- Priority order for the two unimplemented items: hyperparameter search vs. the UI itself?

## 9. Risks
- Without hyperparameter tuning, "best model" comparisons use default sklearn parameters, which may undersell some algorithms (e.g., XGBoost, SVM) relative to their tuned potential — worth flagging to users in the UI copy.
- One-hot encoding with high-cardinality categorical columns could cause dimensionality blow-up; no safeguard currently exists.
