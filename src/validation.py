from src.config import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    OPTIMIZATION_METHODS,
    TEST_SIZE_OPTIONS,
    CV_FOLD_OPTIONS
)


# ============================================================
# Dataset Validation
# ============================================================

def validate_dataset(df, target_column):

    errors = []
    warnings = []

    if df.empty:
        errors.append("The dataset is empty.")
        return errors, warnings

    if target_column not in df.columns:
        errors.append(
            f"The target column '{target_column}' is not present "
            "in the dataset."
        )
        return errors, warnings

    if len(df) < 10:
        warnings.append("The dataset has fewer than 10 rows.")

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        warnings.append(
            f"Dataset contains {duplicate_count} duplicate rows."
        )

    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values > 0]

    if not missing_columns.empty:
        warnings.append("Dataset contains missing values.")

    return errors, warnings


# ============================================================
# AutoML Configuration Validation
# ============================================================

def validate_automl_config(
    problem_type,
    metric,
    test_size,
    optimize,
    optimization_method,
    cv_folds,
    n_iter
):

    errors = []

    # --------------------------------------------------------
    # Problem Type
    # --------------------------------------------------------

    if problem_type not in ["classification", "regression"]:

        errors.append(
            f"Unsupported problem type: {problem_type}"
        )

        return errors

    # --------------------------------------------------------
    # Metric
    # --------------------------------------------------------

    if problem_type == "classification":

        if metric not in CLASSIFICATION_METRICS:

            errors.append(
                f"'{metric}' is not a valid classification metric."
            )

    elif problem_type == "regression":

        if metric not in REGRESSION_METRICS:

            errors.append(
                f"'{metric}' is not a valid regression metric."
            )

    # --------------------------------------------------------
    # Test Size
    # --------------------------------------------------------

    if test_size not in TEST_SIZE_OPTIONS:

        errors.append(
            f"Invalid test size: {test_size}. "
            f"Choose from {TEST_SIZE_OPTIONS}."
        )

    # --------------------------------------------------------
    # Cross-Validation Folds
    # --------------------------------------------------------

    if cv_folds not in CV_FOLD_OPTIONS:

        errors.append(
            f"Invalid CV folds: {cv_folds}. "
            f"Choose from {CV_FOLD_OPTIONS}."
        )

    # --------------------------------------------------------
    # Optimization Method
    # --------------------------------------------------------

    if optimization_method not in OPTIMIZATION_METHODS:

        errors.append(
            f"Unsupported optimization method: "
            f"{optimization_method}"
        )

    # --------------------------------------------------------
    # Number of Iterations
    # --------------------------------------------------------

    if not isinstance(n_iter, int) or n_iter < 1:

        errors.append(
            "n_iter must be an integer greater than or equal to 1."
        )

    # --------------------------------------------------------
    # Optimization Flag
    # --------------------------------------------------------

    if not isinstance(optimize, bool):

        errors.append(
            "optimize must be either True or False."
        )

    return errors