import pandas as pd

from src.config import (
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_CLASSIFICATION_METRIC,
    DEFAULT_REGRESSION_METRIC,
    DEFAULT_OPTIMIZATION_METHOD,
    DEFAULT_CV_FOLDS,
    DEFAULT_N_ITER,
    DEFAULT_SCALING,
    DEFAULT_ENCODING,
    DEFAULT_NUMERICAL_IMPUTATION,
    DEFAULT_CATEGORICAL_IMPUTATION,
    DEFAULT_CATEGORICAL_FILL_VALUE
)

from src.validation import (
    validate_dataset,
    validate_automl_config
)

from src.preprocessing import (
    identify_columns,
    create_preprocessor
)

from src.training import (
    split_data,
    create_model_pipeline,
    train_model,
    make_predictions,
    cross_validate_model
)

from src.models import (
    get_classification_models,
    get_regression_models
)

from src.evaluation import (
    evaluate_classification,
    evaluate_regression,
    select_best_model
)

from src.tuning import tune_model


# ============================================================
# AutoML Workflow
# ============================================================

def run_automl(
    df,
    target_column,
    problem_type,
    test_size=DEFAULT_TEST_SIZE,
    random_state=DEFAULT_RANDOM_STATE,
    metric=None,
    numerical_strategy=DEFAULT_NUMERICAL_IMPUTATION,
    categorical_strategy=DEFAULT_CATEGORICAL_IMPUTATION,
    categorical_fill_value=DEFAULT_CATEGORICAL_FILL_VALUE,
    scaling_strategy=DEFAULT_SCALING,
    encoding_strategy=DEFAULT_ENCODING,
    optimize=False,
    optimization_method=DEFAULT_OPTIMIZATION_METHOD,
    cv_folds=DEFAULT_CV_FOLDS,
    n_iter=DEFAULT_N_ITER
):
    """
    Run the complete AutoML workflow.

    Workflow:
    1. Validate dataset
    2. Determine default metric
    3. Validate AutoML configuration
    4. Separate features and target
    5. Identify numerical and categorical columns
    6. Create preprocessing pipeline
    7. Split data
    8. Select models
    9. Create model pipelines
    10. Optionally optimize hyperparameters
    11. Train models
    12. Cross-validate models
    13. Make predictions
    14. Evaluate models
    15. Create results DataFrame
    16. Create comparison DataFrame
    17. Select best model
    18. Create public summary
    19. Return results
    """

    # ========================================================
    # Dataset Validation
    # ========================================================

    errors, warnings = validate_dataset(
        df,
        target_column
    )

    if errors:
        return {
            "summary": {
                "success": False,
                "problem_type": problem_type,
                "target_column": target_column,
                "metric": metric,
                "best_model": None,
                "comparison_df": pd.DataFrame(),
                "warnings": warnings,
                "errors": errors
            },
            "errors": errors,
            "warnings": warnings
        }

    # ========================================================
    # Default Metric
    # ========================================================

    if metric is None:

        if problem_type == "classification":
            metric = DEFAULT_CLASSIFICATION_METRIC

        elif problem_type == "regression":
            metric = DEFAULT_REGRESSION_METRIC

    # ========================================================
    # Configuration Validation
    # ========================================================

    config_errors = validate_automl_config(
        problem_type=problem_type,
        metric=metric,
        test_size=test_size,
        optimize=optimize,
        optimization_method=optimization_method,
        cv_folds=cv_folds,
        n_iter=n_iter,
        numerical_strategy=numerical_strategy,
        categorical_strategy=categorical_strategy,
        scaling_strategy=scaling_strategy,
        encoding_strategy=encoding_strategy,
        categorical_fill_value=categorical_fill_value
    )

    if config_errors:
        return {
            "summary": {
                "success": False,
                "problem_type": problem_type,
                "target_column": target_column,
                "metric": metric,
                "best_model": None,
                "comparison_df": pd.DataFrame(),
                "warnings": warnings,
                "errors": config_errors
            },
            "errors": config_errors,
            "warnings": warnings
        }

    # ========================================================
    # Features and Target
    # ========================================================

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    # ========================================================
    # Column Identification
    # ========================================================

    numerical_columns, categorical_columns = identify_columns(
        df,
        target_column
    )

    # ========================================================
    # Preprocessing
    # ========================================================

    preprocessor = create_preprocessor(
        numerical_columns,
        categorical_columns,
        numerical_strategy=numerical_strategy,
        categorical_strategy=categorical_strategy,
        scaling_strategy=scaling_strategy,
        encoding_strategy=encoding_strategy,
        categorical_fill_value=categorical_fill_value
    )

    # ========================================================
    # Data Splitting
    # ========================================================

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # ========================================================
    # Model Selection
    # ========================================================

    if problem_type == "classification":

        models = get_classification_models()

    elif problem_type == "regression":

        models = get_regression_models()

    else:

        raise ValueError(
            f"Unsupported problem type: {problem_type}"
        )

    # ========================================================
    # Model Pipelines
    # ========================================================

    model_pipelines = {}

    for model_name, model in models.items():

        model_pipelines[model_name] = create_model_pipeline(
            preprocessor,
            model
        )

    # ========================================================
    # Hyperparameter Optimization
    # ========================================================

    tuning_results = {}
    tuning_errors = {}

    if optimize:

        for model_name, pipeline in model_pipelines.items():

            try:

                tuning_results[model_name] = tune_model(
                    pipeline=pipeline,
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    metric=metric,
                    problem_type=problem_type,
                    method=optimization_method,
                    cv=cv_folds,
                    n_iter=n_iter
                )

            except Exception as e:

                tuning_errors[model_name] = str(e)

        # ----------------------------------------------------
        # Replace Original Pipelines With Tuned Pipelines
        # ----------------------------------------------------

        for model_name, tuning_result in tuning_results.items():

            model_pipelines[model_name] = (
                tuning_result["best_estimator"]
            )

    # ========================================================
    # Model Training
    # ========================================================

    trained_models = {}
    training_errors = {}

    for model_name, pipeline in model_pipelines.items():

        try:

            # Tuned pipelines have already been fitted by
            # GridSearchCV / RandomizedSearchCV.
            if optimize and model_name in tuning_results:

                trained_models[model_name] = pipeline

            else:

                trained_pipeline = train_model(
                    pipeline,
                    X_train,
                    y_train
                )

                trained_models[model_name] = trained_pipeline

        except Exception as e:

            training_errors[model_name] = str(e)

    # ========================================================
    # Cross-Validation
    # ========================================================

    cv_results = {}
    cv_errors = {}

    for model_name, pipeline in model_pipelines.items():

        try:

            cv_results[model_name] = cross_validate_model(
                pipeline,
                X_train,
                y_train,
                scoring=metric,
                problem_type=problem_type,
                cv=cv_folds
            )

        except Exception as e:

            cv_errors[model_name] = str(e)

    # ========================================================
    # Predictions
    # ========================================================

    predictions = {}
    prediction_errors = {}

    for model_name, pipeline in trained_models.items():

        try:

            predictions[model_name] = make_predictions(
                pipeline,
                X_test
            )

        except Exception as e:

            prediction_errors[model_name] = str(e)

    # ========================================================
    # Evaluation
    # ========================================================

    evaluation_results = {}
    evaluation_errors = {}

    for model_name, model_predictions in predictions.items():

        try:

            if problem_type == "classification":

                evaluation_results[model_name] = (
                    evaluate_classification(
                        y_test,
                        model_predictions
                    )
                )

            else:

                evaluation_results[model_name] = (
                    evaluate_regression(
                        y_test,
                        model_predictions
                    )
                )

        except Exception as e:

            evaluation_errors[model_name] = str(e)

    # ========================================================
    # Results DataFrame
    # ========================================================

    results_df = pd.DataFrame.from_dict(
        evaluation_results,
        orient="index"
    )

    results_df.index.name = "model"

    # ========================================================
    # Unified Model Comparison Table
    # ========================================================

    comparison_df = results_df.copy()

    # --------------------------------------------------------
    # Cross-Validation Results
    # --------------------------------------------------------

    cv_mean = {
        model_name: cv_result["mean_score"]
        for model_name, cv_result in cv_results.items()
    }

    cv_std = {
        model_name: cv_result["std_score"]
        for model_name, cv_result in cv_results.items()
    }

    cv_folds = {
        model_name: cv_result["cv_folds"]
        for model_name, cv_result in cv_results.items()
    }

    comparison_df.insert(
        0,
        "cv_mean",
        pd.Series(cv_mean)
    )

    comparison_df.insert(
        1,
        "cv_std",
        pd.Series(cv_std)
    )

    comparison_df.insert(
        2,
        "cv_folds",
        pd.Series(cv_folds)
    )

    # --------------------------------------------------------
    # Tuning Information
    # --------------------------------------------------------

    tuned = {
        model_name: model_name in tuning_results
        for model_name in comparison_df.index
    }

    best_params = {
        model_name: tuning_result["best_params"]
        for model_name, tuning_result
        in tuning_results.items()
    }

    comparison_df["tuned"] = pd.Series(
        tuned
    )

    comparison_df["best_params"] = pd.Series(
        best_params
    )

    # ========================================================
    # Best Model Selection
    # ========================================================

    if results_df.empty:

        best_model = None

    else:

        best_model = select_best_model(
            results_df,
            metric,
            problem_type
        )

    # ========================================================
    # Public AutoML Summary
    # ========================================================

    success = (
        not results_df.empty
        and len(evaluation_results) > 0
    )

    summary = {
        "success": success,
        "problem_type": problem_type,
        "target_column": target_column,
        "metric": metric,
        "best_model": best_model,
        "comparison_df": comparison_df,
        "warnings": warnings,
        "errors": errors
    }

    # ========================================================
    # Return Results
    # ========================================================

    return {

        # ----------------------------------------------------
        # Public Results
        # ----------------------------------------------------

        "summary": summary,

        "errors": errors,
        "warnings": warnings,

        "problem_type": problem_type,
        "target_column": target_column,
        "metric": metric,
        "best_model": best_model,

        "comparison_df": comparison_df,
        "results_df": results_df,

        # ----------------------------------------------------
        # Dataset Information
        # ----------------------------------------------------

        "X": X,
        "y": y,

        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,

        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        "preprocessor": preprocessor,

        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        "models": models,
        "model_pipelines": model_pipelines,
        "trained_models": trained_models,

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        "training_errors": training_errors,

        # ----------------------------------------------------
        # Cross-Validation
        # ----------------------------------------------------

        "cv_results": cv_results,
        "cv_errors": cv_errors,

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        "predictions": predictions,
        "prediction_errors": prediction_errors,

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        "evaluation_results": evaluation_results,
        "evaluation_errors": evaluation_errors,

        # ----------------------------------------------------
        # Hyperparameter Tuning & Preprocessing
        # ----------------------------------------------------

        "numerical_strategy": numerical_strategy,
        "categorical_strategy": categorical_strategy,
        "categorical_fill_value": categorical_fill_value,
        "scaling_strategy": scaling_strategy,
        "encoding_strategy": encoding_strategy,

        "tuning_results": tuning_results,
        "tuning_errors": tuning_errors,

        "optimize": optimize,
        "optimization_method": optimization_method,
        "cv_folds": cv_folds,
        "n_iter": n_iter
    }