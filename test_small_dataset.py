from src.data_loader import load_data
from src.automl import run_automl


# Load the intentionally small classification dataset
df = load_data("data/small_test.csv")


# Run AutoML
result = run_automl(
    df=df,
    target_column="Purchased",
    problem_type="classification"
)


# Check for validation errors
if result["errors"]:

    print("Validation errors:")

    for error in result["errors"]:
        print(error)

else:

    print("AutoML completed.")

    print("\nTraining errors:")

    if result["training_errors"]:
        for model_name, error in result["training_errors"].items():
            print(f"{model_name}: {error}")
    else:
        print("None")

    print("\nTrained models:")

    for model_name in result["trained_models"]:
        print(model_name)

    print("\nEvaluation results:")

    print(result["results_df"])

    print("\nBest model:")

    print(result["best_model"])