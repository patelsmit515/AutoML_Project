import pandas as pd

from src.preprocessing import create_preprocessor, identify_columns
from src.models import get_regression_models
from src.training import split_data, create_model_pipeline
from src.tuning import tune_model


# ============================================================
# Load and Prepare Data
# ============================================================

df = pd.read_csv("data/regression_test.csv")

target_column = "Salary"
X = df.drop(columns=[target_column])
y = df[target_column]

numerical_columns, categorical_columns = identify_columns(
    df,
    target_column
)

preprocessor = create_preprocessor(
    numerical_columns,
    categorical_columns
)

X_train, X_test, y_train, y_test = split_data(X, y)


# ============================================================
# Tune One Model
# ============================================================

models = get_regression_models()
model = models["Linear Regression"]

pipeline = create_model_pipeline(preprocessor, model)

tuning_result = tune_model(
    pipeline=pipeline,
    model_name="Linear Regression",
    X_train=X_train,
    y_train=y_train,
    metric="r2_score",
    problem_type="regression",
    method="grid_search",
    cv=3
)


# ============================================================
# Display Tuning Results
# ============================================================

print("\nBest parameters:")
print(tuning_result["best_params"])

print("\nBest cross-validation score:")
print(tuning_result["best_score"])

print("\nBest estimator:")
print(tuning_result["best_estimator"])