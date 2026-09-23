# ============================================================
# Metric Information
# ============================================================

METRIC_INFO = {

    # Classification metrics
    "accuracy": {
        "name": "Accuracy",
        "direction": "higher",
        "description": (
            "The percentage of predictions that the model got correct."
        ),
        "when_to_use": (
            "Use when the classes are reasonably balanced "
            "and different types of mistakes have similar importance."
        ),
        "warning": (
            "Can be misleading when one class is much more common "
            "than the others."
        )
    },

    "precision": {
        "name": "Precision",
        "direction": "higher",
        "description": (
            "Of the cases the model predicted as positive, "
            "how many were actually positive."
        ),
        "when_to_use": (
            "Use when false positive predictions are particularly costly."
        ),
        "warning": (
            "A high precision can still occur while the model misses "
            "many actual positive cases."
        )
    },

    "recall": {
        "name": "Recall",
        "direction": "higher",
        "description": (
            "Of all the actual positive cases, "
            "how many the model successfully identified."
        ),
        "when_to_use": (
            "Use when missing a positive case is particularly costly."
        ),
        "warning": (
            "A high recall can come with more false positive predictions."
        )
    },

    "f1_score": {
        "name": "F1 Score",
        "direction": "higher",
        "description": (
            "A balance between precision and recall."
        ),
        "when_to_use": (
            "Use when you want to balance false positives "
            "and false negatives, especially with imbalanced classes."
        ),
        "warning": (
            "F1 does not directly show how many predictions "
            "were correct overall."
        )
    },

    # Regression metrics
    "mae": {
        "name": "MAE — Mean Absolute Error",
        "direction": "lower",
        "description": (
            "The average size of the prediction errors."
        ),
        "when_to_use": (
            "Use when you want an easy-to-understand measure "
            "of the typical prediction error."
        ),
        "warning": (
            "Large errors are not given as much extra weight "
            "as they are with MSE or RMSE."
        )
    },

    "mse": {
        "name": "MSE — Mean Squared Error",
        "direction": "lower",
        "description": (
            "The average of the squared prediction errors."
        ),
        "when_to_use": (
            "Use when large prediction errors should be "
            "penalized more heavily."
        ),
        "warning": (
            "Because the errors are squared, the value is not "
            "in the original target units."
        )
    },

    "rmse": {
        "name": "RMSE — Root Mean Squared Error",
        "direction": "lower",
        "description": (
            "Measures prediction error while penalizing "
            "large errors more heavily."
        ),
        "when_to_use": (
            "Use when large prediction errors are particularly "
            "undesirable and you want the error in the same "
            "units as the target."
        ),
        "warning": (
            "Large errors have a stronger effect on RMSE."
        )
    },

    "r2_score": {
        "name": "R² — R-squared",
        "direction": "higher",
        "description": (
            "Measures how well the model explains variation "
            "in the target compared with a simple baseline."
        ),
        "when_to_use": (
            "Use when you want a general measure of how well "
            "the model fits the data."
        ),
        "warning": (
            "R² does not directly tell you how large "
            "the prediction errors are."
        )
    }
}


# ============================================================
# Available Metrics
# ============================================================

CLASSIFICATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score"
]


REGRESSION_METRICS = [
    "mae",
    "mse",
    "rmse",
    "r2_score"
]


# ============================================================
# Classification Model Information
# ============================================================

CLASSIFICATION_MODEL_INFO = {

    "Logistic Regression": {
        "description": (
            "A linear model that estimates the probability "
            "of belonging to a class."
        ),
        "when_to_use": (
            "Useful when the relationship between the features "
            "and the target is relatively simple."
        ),
        "advantages": (
            "Simple, fast, and relatively easy to interpret."
        ),
        "limitations": (
            "May struggle with complex nonlinear relationships."
        ),
        "scaling": "Recommended",
    },

    "Decision Tree": {
        "description": (
            "A tree-based model that makes decisions using "
            "a sequence of rules."
        ),
        "when_to_use": (
            "Useful when relationships between features "
            "and the target may be nonlinear."
        ),
        "advantages": (
            "Easy to understand and does not require feature scaling."
        ),
        "limitations": (
            "Can overfit if the tree becomes too complex."
        ),
        "scaling": "Not required",
    },

    "Random Forest": {
        "description": (
            "Combines many decision trees to make a prediction."
        ),
        "when_to_use": (
            "Useful for many types of structured/tabular datasets."
        ),
        "advantages": (
            "Can capture complex relationships and is generally robust."
        ),
        "limitations": (
            "Can require more computation than a single decision tree."
        ),
        "scaling": "Not required",
    },

    "KNN": {
        "description": (
            "Makes predictions based on the closest observations "
            "in the dataset."
        ),
        "when_to_use": (
            "Useful when similar observations are expected "
            "to have similar outcomes."
        ),
        "advantages": (
            "Simple and can model nonlinear relationships."
        ),
        "limitations": (
            "Can be affected by feature scale and large datasets."
        ),
        "scaling": "Recommended",
    },

    "SVM": {
        "description": (
            "Finds a boundary that separates different classes."
        ),
        "when_to_use": (
            "Useful for datasets where clear boundaries between "
            "classes may exist."
        ),
        "advantages": (
            "Can model complex boundaries using different kernels."
        ),
        "limitations": (
            "Can become computationally expensive on large datasets."
        ),
        "scaling": "Recommended",
    },

    "Gradient Boosting": {
        "description": (
            "Builds models sequentially, with each new model "
            "trying to improve previous errors."
        ),
        "when_to_use": (
            "Useful for structured data and complex relationships."
        ),
        "advantages": (
            "Can achieve strong predictive performance on tabular data."
        ),
        "limitations": (
            "Can take longer to train and may overfit if poorly configured."
        ),
        "scaling": "Not required",
    },

    "XGBoost": {
        "description": (
            "An optimized gradient boosting algorithm "
            "designed for efficient and powerful predictions."
        ),
        "when_to_use": (
            "Often useful for structured/tabular datasets "
            "with complex relationships."
        ),
        "advantages": (
            "Powerful, efficient, and supports many useful parameters."
        ),
        "limitations": (
            "More complex and has many parameters to configure."
        ),
        "scaling": "Not required",
    }
}


# ============================================================
# Regression Model Information
# ============================================================

REGRESSION_MODEL_INFO = {

    "Linear Regression": {
        "description": (
            "A model that predicts a numerical value by "
            "fitting a linear relationship between the features "
            "and the target."
        ),
        "when_to_use": (
            "Useful when the relationship between the features "
            "and target is approximately linear."
        ),
        "advantages": (
            "Simple, fast, and relatively easy to interpret."
        ),
        "limitations": (
            "May perform poorly when relationships are strongly nonlinear."
        ),
        "scaling": "Not required",
    },

    "Decision Tree": {
        "description": (
            "A tree-based model that predicts a numerical value "
            "using a sequence of decision rules."
        ),
        "when_to_use": (
            "Useful when the relationship between features "
            "and the target may be nonlinear."
        ),
        "advantages": (
            "Easy to understand and does not require feature scaling."
        ),
        "limitations": (
            "Can overfit when the tree becomes too complex."
        ),
        "scaling": "Not required",
    },

    "Random Forest": {
        "description": (
            "Combines many decision trees to produce "
            "a numerical prediction."
        ),
        "when_to_use": (
            "Useful for many types of structured/tabular datasets "
            "with potentially complex relationships."
        ),
        "advantages": (
            "Can capture nonlinear relationships and is generally robust."
        ),
        "limitations": (
            "Can require more computation than a single decision tree."
        ),
        "scaling": "Not required",
    },

    "KNN": {
        "description": (
            "Predicts a numerical value based on nearby observations "
            "in the feature space."
        ),
        "when_to_use": (
            "Useful when similar observations are expected "
            "to have similar target values."
        ),
        "advantages": (
            "Simple and can model nonlinear relationships."
        ),
        "limitations": (
            "Can be affected by feature scale and large datasets."
        ),
        "scaling": "Recommended",
    },

    "SVR": {
        "description": (
            "Support Vector Regression predicts numerical values "
            "by finding a function that fits the data within "
            "a specified error tolerance."
        ),
        "when_to_use": (
            "Useful for smaller or medium-sized datasets "
            "where complex relationships need to be modeled."
        ),
        "advantages": (
            "Can model nonlinear relationships using different kernels."
        ),
        "limitations": (
            "Can become computationally expensive on larger datasets "
            "and requires careful parameter selection."
        ),
        "scaling": "Recommended",
    },

    "Gradient Boosting": {
        "description": (
            "Builds regression models sequentially, with each new "
            "model trying to correct errors made by previous models."
        ),
        "when_to_use": (
            "Useful for structured data and complex relationships."
        ),
        "advantages": (
            "Can provide strong predictive performance on tabular data."
        ),
        "limitations": (
            "Can take longer to train and may overfit "
            "if poorly configured."
        ),
        "scaling": "Not required",
    },

    "XGBoost": {
        "description": (
            "An optimized gradient boosting algorithm designed "
            "for efficient and powerful numerical predictions."
        ),
        "when_to_use": (
            "Often useful for structured/tabular datasets "
            "with complex relationships."
        ),
        "advantages": (
            "Powerful, efficient, and supports many useful parameters."
        ),
        "limitations": (
            "More complex and has many parameters to configure."
        ),
        "scaling": "Not required",
    }
}


# ============================================================
# Available Models
# ============================================================

CLASSIFICATION_MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "KNN",
    "SVM",
    "Gradient Boosting",
    "XGBoost"
]


REGRESSION_MODEL_NAMES = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "KNN",
    "SVR",
    "Gradient Boosting",
    "XGBoost"
]


# ============================================================
# Supported Problem Types
# ============================================================

PROBLEM_TYPES = {
    "classification": "Classification",
    "regression": "Regression"
}


# ============================================================
# Default Training Settings
# ============================================================

DEFAULT_TEST_SIZE = 0.2

DEFAULT_RANDOM_STATE = 42

DEFAULT_CV_FOLDS = 5


# ============================================================
# Preprocessing Options
# ============================================================

SCALING_OPTIONS = {
    "standard": "Standard Scaling",
    "minmax": "Min-Max Scaling",
    "robust": "Robust Scaling",
    "none": "No Scaling"
}


ENCODING_OPTIONS = {
    "onehot": "One-Hot Encoding",
    "ordinal": "Ordinal Encoding"
}


NUMERICAL_IMPUTATION_OPTIONS = {
    "mean": "Mean",
    "median": "Median",
    "most_frequent": "Most Frequent"
}


CATEGORICAL_IMPUTATION_OPTIONS = {
    "most_frequent": "Most Frequent",
    "constant": "Constant Value"
}


# ============================================================
# Default Preprocessing Settings
# ============================================================

DEFAULT_SCALING = "standard"

DEFAULT_ENCODING = "onehot"

DEFAULT_NUMERICAL_IMPUTATION = "median"

DEFAULT_CATEGORICAL_IMPUTATION = "most_frequent"


# ============================================================
# Model Selection Defaults
# ============================================================

DEFAULT_CLASSIFICATION_METRIC = "accuracy"

DEFAULT_REGRESSION_METRIC = "r2_score"

DEFAULT_MODEL_SELECTION = "all"


# ============================================================
# Display Defaults
# ============================================================

DEFAULT_SHOW_MODEL_DETAILS = True

DEFAULT_SHOW_METRIC_DETAILS = True

DEFAULT_SHOW_PLOTS = True


# ============================================================
# Test Size Options
# ============================================================

TEST_SIZE_OPTIONS = [
    0.1,
    0.2,
    0.25,
    0.3,
    0.4
]


# ============================================================
# Cross-Validation Options
# ============================================================

CV_FOLD_OPTIONS = [
    3,
    5,
    10
]


# ============================================================
# Optimization Settings
# ============================================================

OPTIMIZATION_METHODS = {
    "grid_search": "Grid Search",
    "random_search": "Randomized Search"
}


DEFAULT_OPTIMIZATION_METHOD = "grid_search"

DEFAULT_N_ITER = 20