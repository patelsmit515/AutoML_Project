import pandas as pd

from src.config import (
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_CLASSIFICATION_METRIC,
    DEFAULT_REGRESSION_METRIC
)

from src.validation import validate_dataset

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


# ============================================================
# AutoML Workflow
# ============================================================

def run_automl(
    df,
    target_column,
    problem_type,
    test_size=DEFAULT_TEST_SIZE,
    random_state=DEFAULT_RANDOM_STATE,
    metric=None
):
    """
    Run the complete AutoML workflow.

    Workflow:
    1. Validate dataset
    2. Separate features and target
    3. Identify numerical and categorical columns
    4. Create preprocessing pipeline
    5. Split data
    6. Select models
    7. Create model pipelines
    8. Train models
    9. Cross-validate models
    10. Make predictions
    11. Evaluate models
    12. Create results DataFrame
    13. Select best model
    """

    # ========================================================
    # Validation
    # ========================================================

    errors, warnings = validate_dataset(
        df,
        target_column
    )

    if errors:
        return {
            "errors": errors,
            "warnings": warnings
        }

    # ========================================================
    # Features and Target
    # ========================================================

    X = df.drop(columns=[target_column])
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
        categorical_columns
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

        if metric is None:
            metric = DEFAULT_CLASSIFICATION_METRIC

    elif problem_type == "regression":

        models = get_regression_models()

        if metric is None:
            metric = DEFAULT_REGRESSION_METRIC

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
    # Model Training
    # ========================================================

    trained_models = {}
    training_errors = {}

    for model_name, pipeline in model_pipelines.items():

        try:

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
                problem_type=problem_type
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
    # Return Results
    # ========================================================

    return {
        "errors": errors,
        "warnings": warnings,

        "problem_type": problem_type,
        "target_column": target_column,

        "X": X,
        "y": y,

        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,

        "preprocessor": preprocessor,

        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,

        "models": models,
        "model_pipelines": model_pipelines,
        "trained_models": trained_models,

        "training_errors": training_errors,

        "cv_results": cv_results,
        "cv_errors": cv_errors,

        "predictions": predictions,

        "prediction_errors": prediction_errors,

        "evaluation_results": evaluation_results,
        "evaluation_errors": evaluation_errors,

        "results_df": results_df,

        "metric": metric,

        "best_model": best_model
    }