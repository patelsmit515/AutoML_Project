from src.config import (
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_CV_FOLDS
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    KFold
)

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
# Model Pipeline
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
    pipeline.fit(X_train, y_train)

    return pipeline


# ============================================================
# Predictions
# ============================================================

def make_predictions(pipeline, X_test):
    predictions = pipeline.predict(X_test)

    return predictions


# ============================================================
# Scoring Conversion
# ============================================================

def get_cv_scoring(metric, is_multiclass=False):
    if is_multiclass:
        scoring_map = {
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1_score": "f1_weighted",

            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "rmse": "neg_root_mean_squared_error",
            "r2_score": "r2"
        }
    else:
        scoring_map = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1_score": "f1",

            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "rmse": "neg_root_mean_squared_error",
            "r2_score": "r2"
        }

    if metric not in scoring_map:
        raise ValueError(
            f"Unsupported cross-validation metric: {metric}"
        )

    return scoring_map[metric]


# ============================================================
# Cross-Validation
# ============================================================

def cross_validate_model(
    pipeline,
    X,
    y,
    scoring,
    problem_type,
    cv=DEFAULT_CV_FOLDS
):
    is_multiclass = False

    if problem_type == "classification":

        class_counts = y.value_counts()

        if len(class_counts) > 2:
            is_multiclass = True

        minimum_class_count = class_counts.min()

        actual_cv = min(
            cv,
            minimum_class_count
        )

        if actual_cv < 2:
            raise ValueError(
                "Cross-validation requires at least "
                "2 samples in each class."
            )

        cv_strategy = StratifiedKFold(
            n_splits=actual_cv,
            shuffle=True,
            random_state=DEFAULT_RANDOM_STATE
        )

    elif problem_type == "regression":

        actual_cv = min(
            cv,
            len(y)
        )

        if actual_cv < 2:
            raise ValueError(
                "Cross-validation requires at least "
                "2 samples."
            )

        cv_strategy = KFold(
            n_splits=actual_cv,
            shuffle=True,
            random_state=DEFAULT_RANDOM_STATE
        )

    else:

        raise ValueError(
            f"Unsupported problem type: {problem_type}"
        )

    scoring_name = get_cv_scoring(scoring, is_multiclass=is_multiclass)

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv_strategy,
        scoring=scoring_name,
        error_score="raise"
    )

    # Convert negative sklearn regression scores
    # back to their normal positive metric values.
    if scoring_name.startswith("neg_"):
        scores = -scores

    return {
        "scores": scores,
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "cv_folds": actual_cv
    }