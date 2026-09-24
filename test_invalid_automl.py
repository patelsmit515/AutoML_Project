import pandas as pd

from src.automl import run_automl


# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("data/test.csv")


# ============================================================
# Helper Function
# ============================================================

def test_configuration(title, **kwargs):

    result = run_automl(
        df=df,
        target_column="Purchased",
        problem_type="classification",
        **kwargs
    )

    print(f"\n{title}")
    print("-" * 50)

    print("Errors:")
    print(result["errors"])

    print("Trained models:")
    print(result.get("trained_models"))


# ============================================================
# Invalid Metric
# ============================================================

test_configuration(
    "Invalid Metric",
    metric="r2_score"
)


# ============================================================
# Invalid Test Size
# ============================================================

test_configuration(
    "Invalid Test Size",
    test_size=0.35
)


# ============================================================
# Invalid CV Folds
# ============================================================

test_configuration(
    "Invalid CV Folds",
    cv_folds=7
)


# ============================================================
# Invalid Optimization Method
# ============================================================

test_configuration(
    "Invalid Optimization Method",
    optimize=True,
    optimization_method="invalid_method"
)


# ============================================================
# Invalid Number of Iterations
# ============================================================

test_configuration(
    "Invalid n_iter",
    optimize=True,
    n_iter=0
)