import pandas as pd

from src.automl import run_automl


# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv(
    "data/missing_values_test.csv"
)


# ============================================================
# Check Original Missing Values
# ============================================================

print("\nOriginal missing values:")
print(df.isnull().sum())


# ============================================================
# Run AutoML
# ============================================================

result = run_automl(
    df=df,
    target_column="Purchased",
    problem_type="classification"
)


# ============================================================
# Display Errors
# ============================================================

print("\nErrors:")
print(result["errors"])


# ============================================================
# Display Warnings
# ============================================================

print("\nWarnings:")
print(result["warnings"])


# ============================================================
# Display Training Errors
# ============================================================

print("\nTraining errors:")
print(result["training_errors"])


# ============================================================
# Display Evaluation Errors
# ============================================================

print("\nEvaluation errors:")
print(result["evaluation_errors"])


# ============================================================
# Display Successful Models
# ============================================================

print("\nTrained models:")
print(
    list(result["trained_models"].keys())
)


# ============================================================
# Display Evaluation Results
# ============================================================

print("\nEvaluation results:")
print(result["results_df"])