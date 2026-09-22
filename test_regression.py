from src.data_loader import load_data
from src.preprocessing import identify_columns, create_preprocessor
from src.models import get_regression_models
from src.training import (
    split_data,
    create_model_pipeline,
    train_model,
    make_predictions
)
from src.evaluation import evaluate_regression

# Load dataset
df = load_data("data/regression_test.csv")

# Target column
target_column = "Salary"

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
    test_size=0.25,
    random_state=42
)


# Get regression models
models = get_regression_models()

# Get all regression models
models = get_regression_models()

all_results = {}

# Train and evaluate each model
for model_name, model in models.items():

    print(f"\n{'=' * 40}")
    print(f"Model: {model_name}")
    print(f"{'=' * 40}")

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

    # Make predictions
    predictions = make_predictions(
        trained_pipeline,
        X_test
    )

    # Evaluate model
    results = evaluate_regression(
        y_test,
        predictions
    )
    
    all_results[model_name] = results
    

    # Display results
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
        
import pandas as pd

results_df = pd.DataFrame.from_dict(
    all_results,
    orient="index"
)

print("\n\nModel comparison:")
print(results_df)