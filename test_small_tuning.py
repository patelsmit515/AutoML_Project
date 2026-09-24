import pandas as pd

from src.automl import run_automl


# ============================================================
# Load Small Classification Dataset
# ============================================================

df = pd.read_csv("data/small_test.csv")


# ============================================================
# Run AutoML With Hyperparameter Tuning
# ============================================================

result = run_automl(
    df=df,
    target_column="Purchased",
    problem_type="classification",
    optimize=True,
    optimization_method="grid_search",
    cv_folds=5
)


# ============================================================
# Display Tuning Errors
# ============================================================

print("\nTuning errors:")

for model_name, error in result["tuning_errors"].items():

    print(f"\n{model_name}:")
    print(error)


# ============================================================
# Display Successful Tuning Results
# ============================================================

print("\nSuccessful tuning results:")

for model_name, tuning_result in result["tuning_results"].items():

    print(f"\n{model_name}")

    print("Best parameters:")
    print(tuning_result["best_params"])

    print("Best CV score:")
    print(tuning_result["best_score"])

    print("Actual CV folds:")
    print(tuning_result["cv_folds"])


# ============================================================
# Final Evaluation
# ============================================================

print("\nFinal evaluation:")
print(result["results_df"])


# ============================================================
# Best Model
# ============================================================

print("\nBest model:")
print(result["best_model"])