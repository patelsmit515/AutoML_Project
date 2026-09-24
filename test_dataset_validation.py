import pandas as pd

from src.automl import run_automl


# ============================================================
# Empty Dataset
# ============================================================

df = pd.DataFrame()


# ============================================================
# Run AutoML
# ============================================================

result = run_automl(
    df=df,
    target_column="Purchased",
    problem_type="classification"
)


# ============================================================
# Display Results
# ============================================================

print("\nErrors:")
print(result["errors"])


print("\nWarnings:")
print(result["warnings"])


print("\nTrained models:")
print(result.get("trained_models"))