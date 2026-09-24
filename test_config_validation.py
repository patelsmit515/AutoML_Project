from src.validation import validate_automl_config


# ============================================================
# Valid Configuration
# ============================================================

errors = validate_automl_config(
    problem_type="classification",
    metric="accuracy",
    test_size=0.2,
    optimize=False,
    optimization_method="grid_search",
    cv_folds=5,
    n_iter=20
)

print("\nValid configuration errors:")
print(errors)


# ============================================================
# Invalid Metric
# ============================================================

errors = validate_automl_config(
    problem_type="classification",
    metric="r2_score",
    test_size=0.2,
    optimize=False,
    optimization_method="grid_search",
    cv_folds=5,
    n_iter=20
)

print("\nInvalid metric errors:")
print(errors)


# ============================================================
# Invalid Test Size
# ============================================================

errors = validate_automl_config(
    problem_type="classification",
    metric="accuracy",
    test_size=0.35,
    optimize=False,
    optimization_method="grid_search",
    cv_folds=5,
    n_iter=20
)

print("\nInvalid test size errors:")
print(errors)


# ============================================================
# Invalid CV Folds
# ============================================================

errors = validate_automl_config(
    problem_type="classification",
    metric="accuracy",
    test_size=0.2,
    optimize=False,
    optimization_method="grid_search",
    cv_folds=7,
    n_iter=20
)

print("\nInvalid CV folds errors:")
print(errors)


# ============================================================
# Invalid Optimization Method
# ============================================================

errors = validate_automl_config(
    problem_type="classification",
    metric="accuracy",
    test_size=0.2,
    optimize=True,
    optimization_method="invalid_method",
    cv_folds=5,
    n_iter=20
)

print("\nInvalid optimization method errors:")
print(errors)


# ============================================================
# Invalid Number of Iterations
# ============================================================

errors = validate_automl_config(
    problem_type="classification",
    metric="accuracy",
    test_size=0.2,
    optimize=True,
    optimization_method="grid_search",
    cv_folds=5,
    n_iter=0
)

print("\nInvalid n_iter errors:")
print(errors)