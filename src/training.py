from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Splitting data
def split_data(
    X,
    y,
    test_size=0.2,
    random_state=42
):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

# 
def create_model_pipeline(preprocessor, model):
    
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline

#model training
def train_model(pipeline, X_train, y_train):

    pipeline.fit(X_train, y_train)

    return pipeline

# Predictions 
def make_predictions(pipeline, X_test):

    predictions = pipeline.predict(X_test)

    return predictions