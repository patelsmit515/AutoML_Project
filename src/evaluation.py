from src.config import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    METRIC_INFO
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# Classification Evaluation
# ============================================================

def evaluate_classification(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    # Determine whether target is binary or multiclass
    unique_classes = set(y_true)
    is_multiclass = len(unique_classes) > 2
    avg_strategy = "weighted" if is_multiclass else "binary"

    precision = precision_score(
        y_true,
        y_pred,
        average=avg_strategy,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average=avg_strategy,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average=avg_strategy,
        zero_division=0
    )

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return results


# ============================================================
# Regression Evaluation
# ============================================================

def evaluate_regression(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = mean_squared_error(
        y_true,
        y_pred
    ) ** 0.5

    r2 = r2_score(
        y_true,
        y_pred
    )

    results = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2
    }

    return results


# ============================================================
# Best Model Selection
# ============================================================

def select_best_model(results_df, metric, problem_type):

    if problem_type == "classification":

        if metric not in CLASSIFICATION_METRICS:
            raise ValueError(
                f"'{metric}' is not a valid classification metric."
            )

    elif problem_type == "regression":

        if metric not in REGRESSION_METRICS:
            raise ValueError(
                f"'{metric}' is not a valid regression metric."
            )

    else:

        raise ValueError(
            f"Unsupported problem type: {problem_type}"
        )

    direction = METRIC_INFO[metric]["direction"]

    if direction == "higher":
        best_model = results_df[metric].idxmax()

    elif direction == "lower":
        best_model = results_df[metric].idxmin()

    else:
        raise ValueError(
            f"Unsupported metric direction: {direction}"
        )

    return best_model