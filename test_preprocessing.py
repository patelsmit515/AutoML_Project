from src.data_loader import load_data
from src.preprocessing import identify_columns, create_preprocessor


# Load dataset
df = load_data("data/test.csv")

# Target column
target_column = "Purchased"

# Identify column types
numerical_columns, categorical_columns = identify_columns(
    df,
    target_column
)

print("Numerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# Create preprocessing pipeline
preprocessor = create_preprocessor(
    numerical_columns,
    categorical_columns
)

# Separate features from target
X = df.drop(columns=[target_column])
y = df[target_column]

# Fit and transform the features
X_processed = preprocessor.fit_transform(X)

print("\nOriginal shape:")
print(X.shape)

print("\nProcessed shape:")
print(X_processed.shape)