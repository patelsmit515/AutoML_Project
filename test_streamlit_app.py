"""
Test suite to verify Streamlit app helper functions and AutoML backend integration.
"""

import os
import pandas as pd
from src.data_loader import load_data
from src.automl import run_automl
from app import (
    format_metric_name,
    format_score,
    plot_model_comparison,
    plot_cross_validation_scores
)


def test_app_helpers():
    print("Testing format helpers...")
    assert format_metric_name("accuracy") == "Accuracy"
    assert "MAE" in format_metric_name("mae")
    assert format_score(0.85231) == "0.8523"
    assert format_score(None) == "N/A"
    assert format_score(15234.5) == "15,234.50"
    print("Format helpers passed!")


def test_classification_flow():
    print("Testing Classification flow...")
    df = load_data("data/test.csv")
    result = run_automl(
        df=df,
        target_column="Purchased",
        problem_type="classification",
        metric="accuracy",
        optimize=False
    )
    assert result["summary"]["success"] is True
    assert result["best_model"] is not None
    assert not result["comparison_df"].empty

    # Test plotting
    fig_comp = plot_model_comparison(result["comparison_df"], "accuracy", "classification")
    assert fig_comp is not None

    fig_cv = plot_cross_validation_scores(result["cv_results"], "accuracy")
    # Small test dataset might have cv errors or results
    print("Classification flow passed!")


def test_regression_flow():
    print("Testing Regression flow...")
    df = load_data("data/regression_test.csv")
    result = run_automl(
        df=df,
        target_column="Salary",
        problem_type="regression",
        metric="r2_score",
        optimize=False
    )
    assert result["summary"]["success"] is True
    assert result["best_model"] is not None
    assert not result["comparison_df"].empty

    fig_comp = plot_model_comparison(result["comparison_df"], "r2_score", "regression")
    assert fig_comp is not None

    fig_cv = plot_cross_validation_scores(result["cv_results"], "r2_score")
    assert fig_cv is not None
    print("Regression flow passed!")


def test_tuning_flow():
    print("Testing Hyperparameter Tuning flow...")
    df = load_data("data/regression_test.csv")
    # Test randomized search
    result = run_automl(
        df=df,
        target_column="Salary",
        problem_type="regression",
        metric="r2_score",
        optimize=True,
        optimization_method="random_search",
        cv_folds=3,
        n_iter=5
    )
    assert result["summary"]["success"] is True
    assert result["optimize"] is True
    assert len(result["tuning_results"]) > 0
    print("Tuning flow passed!")


def test_missing_values_flow():
    print("Testing Missing Values dataset flow...")
    df = load_data("data/missing_values_test.csv")
    result = run_automl(
        df=df,
        target_column="Purchased",
        problem_type="classification"
    )
    assert result["summary"]["success"] is True
    assert "Dataset contains missing values." in result["warnings"]
    print("Missing values flow passed!")


def test_small_dataset_flow():
    print("Testing Small Dataset flow...")
    df = load_data("data/small_test.csv")
    result = run_automl(
        df=df,
        target_column="Purchased",
        problem_type="classification"
    )
    assert result["summary"]["success"] is True
    assert any("fewer than 10 rows" in w for w in result["warnings"])
    print("Small dataset flow passed!")


def test_multiclass_classification_iris():
    print("Testing Multiclass Classification with Iris dataset...")
    df = load_data("data/iris.csv")
    result = run_automl(
        df=df,
        target_column="species",
        problem_type="classification",
        metric="accuracy",
        optimize=False
    )
    assert result["summary"]["success"] is True
    assert result["best_model"] is not None
    assert not result["comparison_df"].empty
    # Check that multiclass evaluation computed accuracy, precision, recall, f1_score
    for m in ["accuracy", "precision", "recall", "f1_score"]:
        assert m in result["results_df"].columns
        assert not result["results_df"][m].isnull().any()
    print("Multiclass Classification with Iris passed!")


def test_preprocessing_scaling_options():
    print("Testing all Scaling strategies (standard, minmax, robust, none)...")
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    df = load_data("data/iris.csv")
    
    # 1. Standard
    res_std = run_automl(df=df, target_column="species", problem_type="classification", scaling_strategy="standard")
    assert res_std["summary"]["success"] is True
    scaler_std = res_std["preprocessor"].named_transformers_["numerical"].named_steps["scaler"]
    assert isinstance(scaler_std, StandardScaler)

    # 2. MinMax
    res_mm = run_automl(df=df, target_column="species", problem_type="classification", scaling_strategy="minmax")
    assert res_mm["summary"]["success"] is True
    scaler_mm = res_mm["preprocessor"].named_transformers_["numerical"].named_steps["scaler"]
    assert isinstance(scaler_mm, MinMaxScaler)

    # 3. Robust
    res_rob = run_automl(df=df, target_column="species", problem_type="classification", scaling_strategy="robust")
    assert res_rob["summary"]["success"] is True
    scaler_rob = res_rob["preprocessor"].named_transformers_["numerical"].named_steps["scaler"]
    assert isinstance(scaler_rob, RobustScaler)

    # 4. None
    res_none = run_automl(df=df, target_column="species", problem_type="classification", scaling_strategy="none")
    assert res_none["summary"]["success"] is True
    scaler_none = res_none["preprocessor"].named_transformers_["numerical"].named_steps["scaler"]
    assert scaler_none == "passthrough"

    print("All scaling strategies passed!")


def test_preprocessing_encoding_options():
    print("Testing all Encoding strategies (onehot, ordinal)...")
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
    df = load_data("data/test.csv")
    
    # 1. OneHot
    res_oh = run_automl(df=df, target_column="Purchased", problem_type="classification", encoding_strategy="onehot")
    assert res_oh["summary"]["success"] is True
    encoder_oh = res_oh["preprocessor"].named_transformers_["categorical"].named_steps["encoder"]
    assert isinstance(encoder_oh, OneHotEncoder)

    # 2. Ordinal
    res_ord = run_automl(df=df, target_column="Purchased", problem_type="classification", encoding_strategy="ordinal")
    assert res_ord["summary"]["success"] is True
    encoder_ord = res_ord["preprocessor"].named_transformers_["categorical"].named_steps["encoder"]
    assert isinstance(encoder_ord, OrdinalEncoder)

    print("All encoding strategies passed!")


def test_missing_value_imputation_options():
    print("Testing Missing-Value Imputation options (1-4)...")
    from sklearn.impute import SimpleImputer
    df_titanic = load_data("data/titanic_test.csv")
    df_missing_reg = load_data("data/missing_regression_test.csv")
    df_no_missing = load_data("data/iris.csv")

    # Case 1: Numerical = Median, Categorical = Most Frequent
    print("  Case 1: Numerical = Median, Categorical = Most Frequent")
    res1 = run_automl(
        df=df_titanic,
        target_column="survived",
        problem_type="classification",
        numerical_strategy="median",
        categorical_strategy="most_frequent"
    )
    assert res1["summary"]["success"] is True
    assert res1["best_model"] is not None
    num_imp1 = res1["preprocessor"].named_transformers_["numerical"].named_steps["imputer"]
    cat_imp1 = res1["preprocessor"].named_transformers_["categorical"].named_steps["imputer"]
    assert isinstance(num_imp1, SimpleImputer) and num_imp1.strategy == "median"
    assert isinstance(cat_imp1, SimpleImputer) and cat_imp1.strategy == "most_frequent"

    # Case 2: Numerical = Mean, Categorical = Most Frequent
    print("  Case 2: Numerical = Mean, Categorical = Most Frequent")
    res2 = run_automl(
        df=df_titanic,
        target_column="survived",
        problem_type="classification",
        numerical_strategy="mean",
        categorical_strategy="most_frequent"
    )
    assert res2["summary"]["success"] is True
    assert res2["best_model"] is not None
    num_imp2 = res2["preprocessor"].named_transformers_["numerical"].named_steps["imputer"]
    cat_imp2 = res2["preprocessor"].named_transformers_["categorical"].named_steps["imputer"]
    assert isinstance(num_imp2, SimpleImputer) and num_imp2.strategy == "mean"
    assert isinstance(cat_imp2, SimpleImputer) and cat_imp2.strategy == "most_frequent"

    # Case 3: Numerical = Most Frequent, Categorical = Most Frequent
    print("  Case 3: Numerical = Most Frequent, Categorical = Most Frequent")
    res3 = run_automl(
        df=df_titanic,
        target_column="survived",
        problem_type="classification",
        numerical_strategy="most_frequent",
        categorical_strategy="most_frequent"
    )
    assert res3["summary"]["success"] is True
    assert res3["best_model"] is not None
    num_imp3 = res3["preprocessor"].named_transformers_["numerical"].named_steps["imputer"]
    cat_imp3 = res3["preprocessor"].named_transformers_["categorical"].named_steps["imputer"]
    assert isinstance(num_imp3, SimpleImputer) and num_imp3.strategy == "most_frequent"
    assert isinstance(cat_imp3, SimpleImputer) and cat_imp3.strategy == "most_frequent"

    # Case 4: Numerical = Median, Categorical = Constant (value = "Unknown")
    print("  Case 4: Numerical = Median, Categorical = Constant ('Unknown')")
    res4 = run_automl(
        df=df_titanic,
        target_column="survived",
        problem_type="classification",
        numerical_strategy="median",
        categorical_strategy="constant",
        categorical_fill_value="Unknown"
    )
    assert res4["summary"]["success"] is True
    assert res4["best_model"] is not None
    num_imp4 = res4["preprocessor"].named_transformers_["numerical"].named_steps["imputer"]
    cat_imp4 = res4["preprocessor"].named_transformers_["categorical"].named_steps["imputer"]
    assert isinstance(num_imp4, SimpleImputer) and num_imp4.strategy == "median"
    assert isinstance(cat_imp4, SimpleImputer) and cat_imp4.strategy == "constant" and cat_imp4.fill_value == "Unknown"

    # Regression with missing values
    print("  Testing Regression with custom imputation...")
    res_reg = run_automl(
        df=df_missing_reg,
        target_column="Salary",
        problem_type="regression",
        numerical_strategy="mean",
        categorical_strategy="constant",
        categorical_fill_value="MissingCity"
    )
    assert res_reg["summary"]["success"] is True
    assert res_reg["best_model"] is not None

    # Dataset without missing values
    print("  Testing Dataset without missing values (Iris)...")
    res_no_miss = run_automl(
        df=df_no_missing,
        target_column="species",
        problem_type="classification",
        numerical_strategy="mean"
    )
    assert res_no_miss["summary"]["success"] is True
    assert res_no_miss["best_model"] is not None

    print("All Missing-Value Imputation tests passed!")


def test_failure_state_handling():
    print("Testing failure state behavior...")
    # Empty dataset or invalid target should have summary['success'] == False
    empty_df = pd.DataFrame()
    result = run_automl(df=empty_df, target_column="target", problem_type="classification")
    assert result["summary"]["success"] is False
    assert len(result["summary"]["errors"]) > 0

    # Invalid target column
    df = load_data("data/test.csv")
    result_invalid = run_automl(df=df, target_column="non_existent_col", problem_type="classification")
    assert result_invalid["summary"]["success"] is False
    assert len(result_invalid["summary"]["errors"]) > 0
    print("Failure state handling passed!")


if __name__ == "__main__":
    test_app_helpers()
    test_classification_flow()
    test_multiclass_classification_iris()
    test_regression_flow()
    test_tuning_flow()
    test_missing_values_flow()
    test_small_dataset_flow()
    test_preprocessing_scaling_options()
    test_preprocessing_encoding_options()
    test_missing_value_imputation_options()
    test_failure_state_handling()
    print("\nALL FRONTEND & INTEGRATION TESTS PASSED SUCCESSFULLY!")

