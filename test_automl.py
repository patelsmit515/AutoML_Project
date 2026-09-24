from src.data_loader import load_data
from src.automl import run_automl


# ============================================================
# Load Dataset
# ============================================================

df = load_data("data/test.csv")


# ============================================================
# Run AutoML
# ============================================================

result = run_automl(
    df=df,
    target_column="Purchased",
    problem_type="classification"
)


# ============================================================
# Validation Results
# ============================================================

if result["errors"]:

    print("Errors:")

    for error in result["errors"]:
        print(error)

else:

    print("AutoML completed successfully!")


    # ========================================================
    # Dataset Information
    # ========================================================

    print("\nWarnings:")

    for warning in result["warnings"]:
        print(warning)

    print("\nProblem type:")
    print(result["problem_type"])

    print("\nTarget column:")
    print(result["target_column"])

    print("\nNumerical columns:")
    print(result["numerical_columns"])

    print("\nCategorical columns:")
    print(result["categorical_columns"])


    # ========================================================
    # Models
    # ========================================================

    print("\nModels:")

    for model_name, model in result["models"].items():
        print(f"{model_name}: {model}")


    # ========================================================
    # Trained Models
    # ========================================================

    print("\nTrained models:")

    for model_name in result["trained_models"]:
        print(model_name)


    # ========================================================
    # Training Errors
    # ========================================================

    print("\nTraining errors:")

    if result["training_errors"]:

        for model_name, error in result["training_errors"].items():
            print(f"{model_name}: {error}")

    else:

        print("None")


    # ========================================================
    # Cross-Validation Results
    # ========================================================

    print("\nCross-validation results:")

    if result["cv_results"]:

        for model_name, cv_result in result["cv_results"].items():

            print(f"\n{model_name}")
            print(f"Scores: {cv_result['scores']}")
            print(f"Mean score: {cv_result['mean_score']}")
            print(f"Standard deviation: {cv_result['std_score']}")
            print(f"CV folds: {cv_result['cv_folds']}")

    else:

        print("None")


    # ========================================================
    # Cross-Validation Errors
    # ========================================================

    print("\nCross-validation errors:")

    if result["cv_errors"]:

        for model_name, error in result["cv_errors"].items():
            print(f"{model_name}: {error}")

    else:

        print("None")


    # ========================================================
    # Prediction Errors
    # ========================================================

    print("\nPrediction errors:")

    if result["prediction_errors"]:

        for model_name, error in result["prediction_errors"].items():
            print(f"{model_name}: {error}")

    else:

        print("None")


    # ========================================================
    # Evaluation Errors
    # ========================================================

    print("\nEvaluation errors:")

    if result["evaluation_errors"]:

        for model_name, error in result["evaluation_errors"].items():
            print(f"{model_name}: {error}")

    else:

        print("None")


    # ========================================================
    # Evaluation Results
    # ========================================================

    print("\nEvaluation results:")

    print(result["results_df"])
    
    # ========================================================
    # Unified Model Comparison
    # ========================================================

    print("\nModel comparison:")

    print(result["comparison_df"])


    # ========================================================
    # Selected Metric
    # ========================================================

    print("\nSelected metric:")

    print(result["metric"])


    # ========================================================
    # Best Model
    # ========================================================

    print("\nBest model:")

    print(result["best_model"])
    
    # ============================================================
    # Public Summary
    # ============================================================

    print("\nPublic summary:")

    print(
        result["summary"]
    )