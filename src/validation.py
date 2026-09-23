def validate_dataset(df, target_column):

    errors = []
    warnings = []

    # ============================================================
    # Empty Dataset Check
    # ============================================================

    if df.empty:
        errors.append("The dataset is empty.")
        return errors, warnings

    # ============================================================
    # Target Column Check
    # ============================================================

    if target_column not in df.columns:
        errors.append(
            f"The target column '{target_column}' is not present "
            "in the dataset."
        )

        return errors, warnings

    # ============================================================
    # Insufficient Rows Check
    # ============================================================

    if len(df) < 10:
        warnings.append(
            "The dataset has fewer than 10 rows."
        )

    # ============================================================
    # Duplicate Rows Check
    # ============================================================

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        warnings.append(
            f"Dataset contains {duplicate_count} duplicate rows."
        )

    # ============================================================
    # Missing Values Check
    # ============================================================

    missing_values = df.isnull().sum()

    missing_columns = missing_values[
        missing_values > 0
    ]

    if not missing_columns.empty:
        warnings.append(
            "Dataset contains missing values."
        )

    return errors, warnings