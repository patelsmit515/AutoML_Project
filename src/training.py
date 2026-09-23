from src.config import (
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# Data Splitting
# ============================================================

def split_data(
    X,
    y,
    test_size=DEFAULT_TEST_SIZE,
    random_state=DEFAULT_RANDOM_STATE
):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# Model Pipeline Creation
# ============================================================

def create_model_pipeline(preprocessor, model):

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


# ============================================================
# Model Training
# ============================================================

def train_model(pipeline, X_train, y_train):

    pipeline.fit(
        X_train,
        y_train
    )

    return pipeline


# ============================================================
# Predictions
# ============================================================

def make_predictions(pipeline, X_test):

    predictions = pipeline.predict(X_test)

    return predictions