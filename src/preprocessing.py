from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder,
    MinMaxScaler,
    RobustScaler
)

from src.config import (
    DEFAULT_SCALING,
    DEFAULT_ENCODING,
    DEFAULT_NUMERICAL_IMPUTATION,
    DEFAULT_CATEGORICAL_IMPUTATION,
    DEFAULT_CATEGORICAL_FILL_VALUE
)


# ============================================================
# Identify Numerical and Categorical Columns
# ============================================================

def identify_columns(df, target_column):

    features = df.drop(
        columns=[target_column]
    )

    numerical_columns = (
        features
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    categorical_columns = (
        features
        .select_dtypes(include="object")
        .columns
        .tolist()
    )

    return numerical_columns, categorical_columns


# ============================================================
# Handle Missing Values
# ============================================================

def handle_missing_values(
    df,
    numerical_columns,
    categorical_columns,
    numerical_strategy=DEFAULT_NUMERICAL_IMPUTATION,
    categorical_strategy=DEFAULT_CATEGORICAL_IMPUTATION,
    categorical_fill_value=DEFAULT_CATEGORICAL_FILL_VALUE
):

    df = df.copy()

    numerical_imputer = SimpleImputer(
        strategy=numerical_strategy
    )

    if categorical_strategy == "constant":
        categorical_imputer = SimpleImputer(
            strategy="constant",
            fill_value=categorical_fill_value
        )
    else:
        categorical_imputer = SimpleImputer(
            strategy=categorical_strategy
        )

    if numerical_columns:

        df[numerical_columns] = numerical_imputer.fit_transform(
            df[numerical_columns]
        )

    if categorical_columns:

        df[categorical_columns] = categorical_imputer.fit_transform(
            df[categorical_columns]
        )

    return df


# ============================================================
# Get Scaler
# ============================================================

def get_scaler(scaling_strategy):

    if scaling_strategy == "standard":
        return StandardScaler()

    elif scaling_strategy == "minmax":
        return MinMaxScaler()

    elif scaling_strategy == "robust":
        return RobustScaler()

    elif scaling_strategy == "none":
        return "passthrough"

    else:
        raise ValueError(
            f"Invalid scaling strategy: {scaling_strategy}"
        )


# ============================================================
# Get Encoder
# ============================================================

def get_encoder(encoding_strategy):

    if encoding_strategy == "onehot":

        return OneHotEncoder(
            handle_unknown="ignore"
        )

    elif encoding_strategy == "ordinal":

        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

    else:

        raise ValueError(
            f"Invalid encoding strategy: {encoding_strategy}"
        )


# ============================================================
# Create Preprocessor
# ============================================================

def create_preprocessor(
    numerical_columns,
    categorical_columns,
    numerical_strategy=DEFAULT_NUMERICAL_IMPUTATION,
    categorical_strategy=DEFAULT_CATEGORICAL_IMPUTATION,
    scaling_strategy=DEFAULT_SCALING,
    encoding_strategy=DEFAULT_ENCODING,
    categorical_fill_value=DEFAULT_CATEGORICAL_FILL_VALUE
):

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=numerical_strategy
                )
            ),
            (
                "scaler",
                get_scaler(scaling_strategy)
            )
        ]
    )

    if categorical_strategy == "constant":
        categorical_imputer = SimpleImputer(
            strategy="constant",
            fill_value=categorical_fill_value
        )
    else:
        categorical_imputer = SimpleImputer(
            strategy=categorical_strategy
        )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                categorical_imputer
            ),
            (
                "encoder",
                get_encoder(encoding_strategy)
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    return preprocessor