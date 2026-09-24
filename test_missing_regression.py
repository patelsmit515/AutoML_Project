import pandas as pd

from src.automl import run_automl


# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv(
    "data/missing_regression_test.csv"
)


# ============================================================
# Run AutoML
# ============================================================

result = run_automl(
    df=df,
    target_column="Salary",
    problem_type="regression"
)


# ============================================================
# Display Results
# ============================================================

print("\nErrors:")
print(result["errors"])


print("\nWarnings:")
print(result["warnings"])


print("\nTraining errors:")
print(result["training_errors"])


print("\nCross-validation errors:")
print(result["cv_errors"])


print("\nEvaluation errors:")
print(result["evaluation_errors"])


print("\nTrained models:")
print(
    list(result["trained_models"].keys())
)


print("\nEvaluation results:")
print(result["results_df"])