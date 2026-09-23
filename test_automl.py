from src.data_loader import load_data
from src.automl import run_automl


# Load classification dataset
df = load_data("data/test.csv")


# Run AutoML
result = run_automl(
    df=df,
    target_column="Purchased",
    problem_type="classification"
)


# Check for validation errors
if result["errors"]:

    print("Errors:")

    for error in result["errors"]:
        print(error)

else:

    print("AutoML completed successfully!")

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

    print("\nModels:")
    for model_name, model in result["models"].items():
        print(f"{model_name}: {model}")

    print("\nTrained models:")
    for model_name in result["trained_models"]:
        print(model_name)

    print("\nTraining errors:")
    if result["training_errors"]:
        for model_name, error in result["training_errors"].items():
            print(f"{model_name}: {error}")
    else:
        print("None")

    print("\nPrediction errors:")
    if result["prediction_errors"]:
        for model_name, error in result["prediction_errors"].items():
            print(f"{model_name}: {error}")
    else:
        print("None")

    print("\nEvaluation errors:")
    if result["evaluation_errors"]:
        for model_name, error in result["evaluation_errors"].items():
            print(f"{model_name}: {error}")
    else:
        print("None")

    print("\nEvaluation results:")
    print(result["results_df"])

    print("\nSelected metric:")
    print(result["metric"])

    print("\nBest model:")
    print(result["best_model"])