from src.config import DEFAULT_RANDOM_STATE

from src.data_loader import load_data
from src.preprocessing import (
    identify_columns,
    create_preprocessor
)
from src.models import get_classification_models
from src.training import (
    split_data,
    create_model_pipeline,
    train_model,
    make_predictions
)
from src.evaluation import evaluate_classification


# ============================================================
# Load Dataset
# ============================================================

df = load_data("data/test.csv")


# ============================================================
# Target Column
# ============================================================

target_column = "Purchased"


# ============================================================
# Separate Features and Target
# ============================================================

X = df.drop(columns=[target_column])
y = df[target_column]


# ============================================================
# Identify Column Types
# ============================================================

numerical_columns, categorical_columns = identify_columns(
    df,
    target_column
)


# ============================================================
# Create Preprocessor
# ============================================================

preprocessor = create_preprocessor(
    numerical_columns,
    categorical_columns
)


# ============================================================
# Split Data
# ============================================================

X_train, X_test, y_train, y_test = split_data(
    X,
    y,
    test_size=0.2,
    random_state=DEFAULT_RANDOM_STATE
)


# ============================================================
# Get Classification Models
# ============================================================

models = get_classification_models()


# Select one model
model = models["Random Forest"]


# ============================================================
# Create Pipeline
# ============================================================

pipeline = create_model_pipeline(
    preprocessor,
    model
)


# ============================================================
# Train Model
# ============================================================

trained_pipeline = train_model(
    pipeline,
    X_train,
    y_train
)

print("Model trained successfully!")


# ============================================================
# Make Predictions
# ============================================================

predictions = make_predictions(
    trained_pipeline,
    X_test
)

print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(predictions)


# ============================================================
# Evaluate Model
# ============================================================

results = evaluate_classification(
    y_test,
    predictions
)

print("\nEvaluation results:")

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")