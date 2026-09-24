from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.config import (
    DEFAULT_OPTIMIZATION_METHOD,
    DEFAULT_N_ITER,
    DEFAULT_CV_FOLDS
)

from src.training import get_cv_scoring


# ============================================================
# Hyperparameter Search Spaces
# ============================================================

def get_parameter_grid(model_name):
    parameter_grids = {
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["lbfgs"]
        },

        "Decision Tree": {
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_split": [2, 5, 10]
        },

        "Random Forest": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5]
        },

        "KNN": {
            "model__n_neighbors": [3, 5, 7],
            "model__weights": ["uniform", "distance"]
        },

        "SVM": {
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["linear", "rbf"]
        },

        "Gradient Boosting": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [2, 3, 5]
        },

        "XGBoost": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5, 7]
        },

        "Linear Regression": {
            "model__fit_intercept": [True, False]
        },

        "SVR": {
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["linear", "rbf"],
            "model__epsilon": [0.1, 0.2, 0.5]
        }
    }

    if model_name not in parameter_grids:
        raise ValueError(
            f"No parameter grid is defined for '{model_name}'."
        )

    return parameter_grids[model_name]


# ============================================================
# Hyperparameter Optimization
# ============================================================

def tune_model(
    pipeline,
    model_name,
    X_train,
    y_train,
    metric,
    problem_type,
    method=DEFAULT_OPTIMIZATION_METHOD,
    cv=DEFAULT_CV_FOLDS,
    n_iter=DEFAULT_N_ITER
):
    parameter_grid = get_parameter_grid(model_name)
    scoring = get_cv_scoring(metric)

    if method == "grid_search":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            error_score="raise"
        )

    elif method == "random_search":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameter_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=42,
            error_score="raise"
        )

    else:
        raise ValueError(
            f"Unsupported optimization method: {method}"
        )

    search.fit(X_train, y_train)

    best_score = search.best_score_

    # Convert sklearn's negative-loss scores back to positive values.
    if scoring.startswith("neg_"):
        best_score = -best_score

    return {
        "model_name": model_name,
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": best_score,
        "metric": metric,
        "method": method,
        "cv_folds": cv
    }