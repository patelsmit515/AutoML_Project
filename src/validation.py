def validate_dataset(df, target_column):
    errors= []
    warnings = []
    
    #empty dataset check
    if df.empty:
        errors.append("The dataset is empty.")
    
    # Insufficient rows check
    if len(df) < 10:
        warnings.append("The dataset has fewer than 10 rows.")
    
    # Missing target column check
    if target_column not in df.columns:
        errors.append(f"The target column '{target_column}' is not present in the dataset.")
    
    # Duplicate values check
    duplicate_count = df.duplicated().sum()
    
    if duplicate_count > 0:
        warnings.append(f"Dataset contains {duplicate_count} duplicate rows.")
    
    # Missing value check
    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values > 0]

    if not missing_columns.empty:
        warnings.append('Dataset contains missing values.')
    
    #Check data types
    if df.select_dtypes(include="number").shape[1] == 0:
        warnings.append("Dataset contains no numerical columns.")

    return errors, warnings  