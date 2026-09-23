# Architecture — AutoML Project

## 1. Summary
A Python backend engine for tabular AutoML, organized as a set of composable modules under `src/`. Each stage of the ML workflow (load → validate → preprocess → train → evaluate) is a separate module with pure functions, wired together by thin driver scripts. No web framework or persistence layer is wired in yet — `streamlit` is a declared dependency for a planned UI that hasn't been implemented.

## 2. Tech Stack
- **Language:** Python
- **Core libraries:** pandas, numpy, scikit-learn, xgboost (imported directly in `models.py`, not in `requirements.txt` — see risks)
- **Planned UI:** Streamlit
- **Planned visualization:** matplotlib, seaborn
- **Testing:** ad-hoc script-based smoke tests (`test_preprocessing.py`, `test_training.py`, `test_regression.py`) — not pytest-style assertions

## 3. Project Structure
```
AutoML_Project/
├── data/
│   ├── test.csv                  # classification sample (Age, Salary, City, Purchased)
│   └── regression_test.csv       # regression sample (Age, Experience, Salary)
├── src/
│   ├── config.py                 # metric/model metadata, defaults, option constants
│   ├── data_loader.py            # CSV → DataFrame
│   ├── validation.py             # dataset quality checks
│   ├── preprocessing.py          # column typing, imputation, scaling, encoding pipeline
│   ├── models.py                 # model factories (classification & regression)
│   ├── training.py               # split, pipeline assembly, fit, predict
│   └── evaluation.py             # metrics, best-model selection
├── test_preprocessing.py         # smoke test: preprocessing on test.csv
├── test_training.py              # smoke test: single-model training on test.csv
├── test_regression.py            # smoke test: all regression models on regression_test.csv
└── requirements.txt
```

## 4. Module Responsibilities

### `config.py`
Central source of truth for:
- `METRIC_INFO` — name, direction (higher/lower is better), description, when-to-use, and warning text for each metric.
- `CLASSIFICATION_MODEL_INFO` / regression equivalent — description, when-to-use, advantages, limitations, scaling recommendation per model.
- Lists of available metrics/models per problem type.
- Default settings: test size (0.2), random state (42), CV folds (5), scaling (standard), encoding (one-hot), imputation strategies, optimization method (grid search), n_iter (20).
- UI-facing option dictionaries (test size options, CV fold options, scaling/encoding/imputation option labels) — designed to back dropdowns/selectors in the not-yet-built UI.

This module has no logic, only data — it's the contract between the ML engine and any future UI.

### `data_loader.py`
Single function `load_data(file_path)` wrapping `pd.read_csv`. No error handling for malformed files, wrong delimiters, or encoding issues yet.

### `validation.py`
`validate_dataset(df, target_column)` returns `(errors, warnings)`:
- **Errors (blocking):** empty dataset, missing target column.
- **Warnings (non-blocking):** <10 rows, duplicate rows, missing values present, no numerical columns.
This split lets a future UI decide what halts the flow vs. what's just surfaced as a caution.

### `preprocessing.py`
- `identify_columns(df, target_column)` — splits feature columns into numerical (`number` dtype) vs. categorical (`object` dtype). Note: this misses boolean, datetime, and category dtypes.
- `handle_missing_values(...)` — standalone imputation utility (currently unused by the main pipeline path, which does imputation inline via `create_preprocessor`).
- `get_scaler` / `get_encoder` — factory functions mapping config strings to sklearn transformers.
- `create_preprocessor(...)` — builds a `ColumnTransformer` with parallel numerical (impute → scale) and categorical (impute → encode) `Pipeline`s. This is the reusable preprocessing artifact fit inside every model pipeline.

### `models.py`
Two factory functions returning `{name: unfitted_estimator}` dictionaries:
- `get_classification_models()` — Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gradient Boosting, XGBoost.
- `get_regression_models()` — Linear Regression, Decision Tree, Random Forest, KNN, SVR, Gradient Boosting, XGBoost.
Each call constructs fresh, unfitted instances with default hyperparameters.

### `training.py`
- `split_data(X, y, test_size, random_state)` — thin wrapper over `train_test_split`.
- `create_model_pipeline(preprocessor, model)` — combines preprocessor + estimator into a single sklearn `Pipeline`, so preprocessing is fit only on training data and applied consistently at predict time.
- `train_model(pipeline, X_train, y_train)` / `make_predictions(pipeline, X_test)` — fit/predict wrappers.

### `evaluation.py`
- `evaluate_classification(y_true, y_pred)` → accuracy, precision, recall, F1 (binary-classification assumptions: `precision_score`/`recall_score`/`f1_score` are called without `average=`, which defaults to binary — will break on multiclass targets).
- `evaluate_regression(y_true, y_pred)` → MAE, MSE, RMSE, R².
- `select_best_model(results_df, metric, problem_type)` — picks `idxmax`/`idxmin` on a results DataFrame depending on whether the metric is "higher is better" or "lower is better" (MAE/MSE/RMSE use `idxmin`; everything else `idxmax`).

## 5. Data Flow (current, script-driven)
```
CSV file
  → load_data()
  → validate_dataset()               [errors/warnings]
  → identify_columns()               [numerical_columns, categorical_columns]
  → create_preprocessor()            [ColumnTransformer]
  → split_data()                     [X_train, X_test, y_train, y_test]
  → get_*_models()                   [{name: estimator}]
  → for each model:
        create_model_pipeline()
        train_model()
        make_predictions()
        evaluate_*()                 [metrics dict]
  → aggregate into results_df
  → select_best_model()
```
This flow is currently exercised by the three top-level `test_*.py` scripts, each hardcoding a dataset path and target column — there is no orchestrating "run pipeline for arbitrary dataset" entry point yet.

## 6. Planned / Missing Layers
- **UI layer (Streamlit):** dependency declared, no `app.py` or equivalent exists. Would consume `config.py`'s option dictionaries and metadata directly.
- **Hyperparameter search:** `OPTIMIZATION_METHODS` (grid/random search) and `DEFAULT_N_ITER` are defined in config but not wired into `training.py` — no `GridSearchCV`/`RandomizedSearchCV` usage in the codebase yet.
- **Visualization:** matplotlib/seaborn are dependencies but unused in `src/`.
- **Orchestration entry point:** no single function/CLI that takes (file path, target column, config overrides) and returns full results — logic currently lives only in test scripts.
- **Automated tests:** current `test_*.py` files are runnable demo scripts (print statements), not `pytest`/`unittest` assertions — no CI-friendly test suite yet.

## 7. Known Issues / Risks
- `xgboost` is imported in `models.py` but not listed in `requirements.txt` — will cause an `ImportError` on a clean install.
- `identify_columns` only recognizes `number` and `object` dtypes; boolean/datetime/categorical dtype columns fall through uncategorized.
- Classification metrics assume binary targets (no `average` parameter set for multiclass).
- `handle_missing_values` in `preprocessing.py` is dead code relative to the pipeline path (imputation is duplicated inside `create_preprocessor`).
- No handling for extremely large datasets or memory limits during `pd.read_csv`.

## 8. Suggested Next Steps
1. Add `xgboost` to `requirements.txt`.
2. Build the Streamlit UI, consuming `config.py` metadata directly for labels/help text.
3. Wire `GridSearchCV`/`RandomizedSearchCV` into `training.py` using the existing `OPTIMIZATION_METHODS` config.
4. Generalize classification metrics to support multiclass targets.
5. Add a single orchestration function (e.g., `run_automl(file_path, target_column, config_overrides)`) to replace the duplicated logic across the three test scripts.
6. Convert smoke-test scripts into real `pytest` tests with assertions.
