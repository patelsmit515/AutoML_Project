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
    make_predictions
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
    9. Make predictions
    10. Evaluate models
    11. Create results DataFrame
    12. Select best model
    """

    # --------------------------------------------------
    # Step 1: Validate dataset
    # --------------------------------------------------

    errors, warnings = validate_dataset(
        df,
        target_column
    )

    if errors:
        return {
            "errors": errors,
            "warnings": warnings
        }

    # --------------------------------------------------
    # Step 2: Separate features and target
    # --------------------------------------------------

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --------------------------------------------------
    # Step 3: Identify numerical and categorical columns
    # --------------------------------------------------

    numerical_columns, categorical_columns = identify_columns(
        df,
        target_column
    )

    # --------------------------------------------------
    # Step 4: Create preprocessing pipeline
    # --------------------------------------------------

    preprocessor = create_preprocessor(
        numerical_columns,
        categorical_columns
    )

    # --------------------------------------------------
    # Step 5: Split data
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # --------------------------------------------------
    # Step 6: Select models
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Step 7: Create a pipeline for every model
    # --------------------------------------------------

    model_pipelines = {}

    for model_name, model in models.items():

        model_pipelines[model_name] = create_model_pipeline(
            preprocessor,
            model
        )

    # --------------------------------------------------
    # Step 8: Train every model
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Step 9: Make predictions
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Step 10: Evaluate every model
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Step 11: Create results DataFrame
    # --------------------------------------------------

    results_df = pd.DataFrame.from_dict(
        evaluation_results,
        orient="index"
    )

    results_df.index.name = "model"

    # --------------------------------------------------
    # Step 12: Select best model
    # --------------------------------------------------

    if results_df.empty:

        best_model = None

    else:

        best_model = select_best_model(
            results_df,
            metric,
            problem_type
        )

    # --------------------------------------------------
    # Step 13: Return complete AutoML results
    # --------------------------------------------------

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

        "predictions": predictions,

        "prediction_errors": prediction_errors,

        "evaluation_results": evaluation_results,

        "evaluation_errors": evaluation_errors,

        "results_df": results_df,

        "metric": metric,

        "best_model": best_model
    }