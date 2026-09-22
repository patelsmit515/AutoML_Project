from src.config import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS
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

# Classification evaluation function
def evaluate_classification(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return results


# Regression evaluation function
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

def select_best_model(results_df, metric, problem_type):

    
    if problem_type == "classification":

        if metric not in CLASSIFICATION_METRICS:
            raise ValueError(
                f"'{metric}' is not a valid classification metric."
            )

        best_model = results_df[metric].idxmax()

    elif problem_type == "regression":

        if metric not in REGRESSION_METRICS:
            raise ValueError(
                f"'{metric}' is not a valid regression metric."
            )

        if metric in ["mae", "mse", "rmse"]:
            best_model = results_df[metric].idxmin()

        else:
            best_model = results_df[metric].idxmax()

    else:
        raise ValueError(
            f"Unsupported problem type: {problem_type}"
        )

    return best_model