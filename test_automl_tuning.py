import pandas as pd

from src.automl import run_automl


# ============================================================
# Load Data
# ============================================================

df = pd.read_csv("data/regression_test.csv")


# ============================================================
# Run AutoML With Optimization
# ============================================================

result = run_automl(
    df=df,
    target_column="Salary",
    problem_type="regression",
    optimize=True,
    optimization_method="grid_search",
    cv_folds=3
)


# ============================================================
# Display Tuning Results
# ============================================================

print("\nTuning errors:")
print(result["tuning_errors"])

print("\nTuning results:")

for model_name, tuning_result in result["tuning_results"].items():

    print(f"\n{model_name}")
    print("Best parameters:")
    print(tuning_result["best_params"])
    print("Best CV score:")
    print(tuning_result["best_score"])


# ============================================================
# Display Final Evaluation
# ============================================================

print("\nFinal evaluation:")
print(result["results_df"])

print("\nBest model:")
print(result["best_model"])