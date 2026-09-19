import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, StandardScaler,
                    OrdinalEncoder,MinMaxScaler,RobustScaler)




#Num and Cat Columns identifier
def identify_columns(df, target_column):
    features = df.drop(columns=[target_column])

    numerical_columns = features.select_dtypes(include="number").columns.tolist()
    
    categorical_columns = features.select_dtypes(include="object").columns.tolist()
    
    return numerical_columns, categorical_columns


#Null values imputer
def handle_missing_values(
    df,
    numerical_columns,
    categorical_columns,
    numerical_strategy="median",
    categorical_strategy="most_frequent"
):
    df = df.copy()
    
    numerical_imputer = SimpleImputer(
        strategy = numerical_strategy
    )

    categorical_imputer = SimpleImputer(
        strategy = categorical_strategy
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

#scaling Num data
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
        raise ValueError("Invalid scaling strategy.")

# Encoding Categorical data

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
        raise ValueError("Invalid encoding strategy.")

# Converting data 
def create_preprocessor(
    numerical_columns,
    categorical_columns,
    numerical_strategy="median",
    categorical_strategy="most_frequent",
    scaling_strategy="standard",
    encoding_strategy="onehot"
):

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                strategy = numerical_strategy
            )),
            ("scaler", get_scaler(scaling_strategy))
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                strategy=categorical_strategy
            )),
            ("encoder", get_encoder(encoding_strategy))
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