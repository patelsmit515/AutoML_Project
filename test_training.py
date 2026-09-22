from src.data_loader import load_data
from src.preprocessing import identify_columns, create_preprocessor
from src.models import get_classification_models
from src.training import split_data, create_model_pipeline, train_model
from src.training import make_predictions
from src.evaluation import evaluate_classification


# Load dataset
df = load_data("data/test.csv")

# Target column
target_column = "Purchased"

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]


# Identify column types
numerical_columns, categorical_columns = identify_columns(
    df,
    target_column
)


# Create preprocessor
preprocessor = create_preprocessor(
    numerical_columns,
    categorical_columns
)


# Split data
X_train, X_test, y_train, y_test = split_data(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Get classification models
models = get_classification_models()

# Select one model
model = models["Random Forest"]


# Create pipeline
pipeline = create_model_pipeline(
    preprocessor,
    model
)


# Train model
trained_pipeline = train_model(
    pipeline,
    X_train,
    y_train
)

print("Model trained successfully!")

# Make predictions
predictions = make_predictions(
    trained_pipeline,
    X_test
)

print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(predictions)

print("Reached evaluation section")

# Evaluate model
results = evaluate_classification(
    y_test,
    predictions
)

print("\nEvaluation results:")

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")


