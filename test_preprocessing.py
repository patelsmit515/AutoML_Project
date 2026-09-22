from src.data_loader import load_data
from src.preprocessing import (
    identify_columns,
    create_preprocessor
)


# Load dataset
df = load_data("data/test.csv")

# Target column
target_column = "Purchased"


# Identify numerical and categorical columns
numerical_columns, categorical_columns = identify_columns(
    df,
    target_column
)

print("Numerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# Create preprocessor
preprocessor = create_preprocessor(
    numerical_columns,
    categorical_columns
)

print("\nPreprocessor created successfully!")
print(preprocessor)