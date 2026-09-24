"""
AutoML Studio — Streamlit Frontend for Tabular AutoML Backend.

This application provides a professional, interactive interface for the
existing AutoML machine learning backend, allowing users to upload datasets,
configure preprocessing/training/tuning parameters, run AutoML pipelines,
and inspect model comparisons, evaluation metrics, cross-validation, and
hyperparameter search results.
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Backend Imports (Source of Truth)
# ============================================================
from src.config import (
    METRIC_INFO,
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    CLASSIFICATION_MODEL_INFO,
    REGRESSION_MODEL_INFO,
    CLASSIFICATION_MODEL_NAMES,
    REGRESSION_MODEL_NAMES,
    PROBLEM_TYPES,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_CV_FOLDS,
    DEFAULT_CLASSIFICATION_METRIC,
    DEFAULT_REGRESSION_METRIC,
    TEST_SIZE_OPTIONS,
    CV_FOLD_OPTIONS,
    OPTIMIZATION_METHODS,
    DEFAULT_OPTIMIZATION_METHOD,
    DEFAULT_N_ITER,
    SCALING_OPTIONS,
    ENCODING_OPTIONS,
    NUMERICAL_IMPUTATION_OPTIONS,
    CATEGORICAL_IMPUTATION_OPTIONS,
    DEFAULT_SCALING,
    DEFAULT_ENCODING,
    DEFAULT_NUMERICAL_IMPUTATION,
    DEFAULT_CATEGORICAL_IMPUTATION,
    DEFAULT_CATEGORICAL_FILL_VALUE
)
from src.data_loader import load_data
from src.preprocessing import identify_columns
from src.validation import validate_dataset, validate_automl_config
from src.automl import run_automl


# ============================================================
# Page Configuration & UI Theme
# ============================================================
st.set_page_config(
    page_title="AutoML Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Custom CSS Styling
# ============================================================
def apply_custom_styles():
    st.markdown(
        """
        <style>
        /* Base typography and clean layout */
        .main {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Metric Card styling */
        .metric-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            transition: all 0.2s ease-in-out;
        }
        .metric-card:hover {
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);
            border-color: #cbd5e1;
        }
        .metric-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #0f172a;
        }
        .metric-subtext {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 4px;
        }

        /* Best Model Highlight Hero Card */
        .hero-best-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #ffffff;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.15);
        }
        .hero-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #38bdf8;
            font-weight: 700;
        }
        .hero-model-name {
            font-size: 2rem;
            font-weight: 800;
            margin: 4px 0 12px 0;
            color: #ffffff;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: #38bdf8;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        .hero-badge-accent {
            display: inline-block;
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.4);
            color: #4ade80;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 8px;
            margin-bottom: 8px;
        }

        /* Status badges */
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-success { background: #dcfce7; color: #15803d; }
        .badge-warning { background: #fef9c3; color: #a16207; }
        .badge-info { background: #e0f2fe; color: #0369a1; }
        .badge-danger { background: #fee2e2; color: #b91c1c; }

        /* Section headers */
        .section-header {
            font-size: 1.25rem;
            font-weight: 700;
            color: #1e293b;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #f1f5f9;
        }

        /* Information cards */
        .info-box {
            background-color: #f8fafc;
            border-left: 4px solid #3b82f6;
            padding: 14px 18px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# Helper Functions for Formatting and Visualizations
# ============================================================
def format_metric_name(metric_key):
    """Return user-friendly display name for metric key."""
    if metric_key in METRIC_INFO:
        return METRIC_INFO[metric_key]["name"]
    return metric_key.replace("_", " ").title()


def format_score(score, metric_key=None):
    """Format numeric score neatly with precision."""
    if score is None or pd.isna(score):
        return "N/A"
    try:
        val = float(score)
        if abs(val) >= 10000:
            return f"{val:,.2f}"
        elif abs(val) >= 100:
            return f"{val:.2f}"
        elif abs(val) >= 1:
            return f"{val:.4f}"
        elif abs(val) == 0:
            return "0.0000"
        else:
            return f"{val:.4f}"
    except (ValueError, TypeError):
        return str(score)


def plot_model_comparison(comparison_df, metric_key, problem_type):
    """Generate a clean matplotlib comparison chart."""
    if comparison_df.empty or metric_key not in comparison_df.columns:
        return None

    direction = METRIC_INFO.get(metric_key, {}).get("direction", "higher")
    is_higher_better = direction == "higher"
    
    # Sort DataFrame by metric for plotting
    plot_df = comparison_df.copy()
    plot_df = plot_df.sort_values(by=metric_key, ascending=is_higher_better)

    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.6)))
    
    # Palette styling
    colors = sns.color_palette("Blues_r", len(plot_df))
    if not is_higher_better:
        colors = sns.color_palette("Reds_r", len(plot_df))

    bars = ax.barh(plot_df.index, plot_df[metric_key], color=colors, edgecolor="#334155", height=0.6, alpha=0.9)

    # Format title and labels
    metric_title = format_metric_name(metric_key)
    ax.set_title(f"Model Comparison — {metric_title} ({'Higher is better' if is_higher_better else 'Lower is better'})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(metric_title, fontsize=10, fontweight="bold")
    ax.set_ylabel("Model", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Add data value labels on bars
    for bar in bars:
        width = bar.get_width()
        text_val = format_score(width, metric_key)
        x_pos = width + (width * 0.01 if width >= 0 else -abs(width * 0.05))
        align = "left" if width >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f" {text_val}", va="center", ha=align, fontsize=9, fontweight="bold", color="#1e293b")

    plt.tight_layout()
    return fig


def plot_cross_validation_scores(cv_results, metric_key):
    """Generate a bar plot with error bars for Cross-Validation mean and std."""
    if not cv_results:
        return None

    models = []
    means = []
    stds = []

    for model_name, cv_data in cv_results.items():
        models.append(model_name)
        means.append(cv_data["mean_score"])
        stds.append(cv_data["std_score"])

    df_cv = pd.DataFrame({"Model": models, "CV_Mean": means, "CV_Std": stds})
    
    direction = METRIC_INFO.get(metric_key, {}).get("direction", "higher")
    is_higher_better = direction == "higher"
    df_cv = df_cv.sort_values(by="CV_Mean", ascending=is_higher_better)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df_cv) * 0.6)))
    y_pos = np.arange(len(df_cv))

    ax.barh(y_pos, df_cv["CV_Mean"], xerr=df_cv["CV_Std"], align="center", alpha=0.85, color="#6366f1", edgecolor="#312e81", capsize=5, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_cv["Model"], fontsize=10, fontweight="bold")
    
    metric_title = format_metric_name(metric_key)
    ax.set_title(f"Cross-Validation Performance (Mean ± Std Dev) — {metric_title}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(f"CV Mean ({metric_title})", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


# ============================================================
# Main Application Component Rendering
# ============================================================
def render_header():
    """Render top header banner."""
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown("<h1 style='margin-bottom:0;'>⚡ AutoML Studio</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b; font-size:1.05rem; margin-top:0;'>Automated Machine Learning & Model Comparison Platform for Tabular Data</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            "<div style='text-align:right; padding-top:10px;'>"
            "<span class='badge badge-info'>v1.0 Production</span>"
            "</div>",
            unsafe_allow_html=True
        )
    st.divider()


def render_dataset_overview(df, filename, target_column=None):
    """Render dataset statistics, column types, and missing value information."""
    st.markdown("<div class='section-header'>📊 Dataset Overview</div>", unsafe_allow_html=True)

    num_rows, num_cols = df.shape
    num_cols_detected = []
    cat_cols_detected = []

    if target_column and target_column in df.columns:
        num_cols_detected, cat_cols_detected = identify_columns(df, target_column)
    else:
        num_cols_detected = df.select_dtypes(include="number").columns.tolist()
        cat_cols_detected = df.select_dtypes(include="object").columns.tolist()

    total_missing = int(df.isnull().sum().sum())
    cols_with_missing = df.columns[df.isnull().any()].tolist()

    # Stat metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Filename</div>
                <div class='metric-value' style='font-size:1.1rem; word-break:break-all;'>{filename}</div>
                <div class='metric-subtext'>Source Dataset</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Total Rows</div>
                <div class='metric-value'>{num_rows:,}</div>
                <div class='metric-subtext'>Observations</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Total Columns</div>
                <div class='metric-value'>{num_cols:,}</div>
                <div class='metric-subtext'>Features + Target</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Feature Types</div>
                <div class='metric-value'>{len(num_cols_detected)}N / {len(cat_cols_detected)}C</div>
                <div class='metric-subtext'>Num / Categorical</div>
            </div>
        """, unsafe_allow_html=True)
    with c5:
        missing_color = "#b91c1c" if total_missing > 0 else "#15803d"
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Missing Values</div>
                <div class='metric-value' style='color:{missing_color};'>{total_missing:,}</div>
                <div class='metric-subtext'>{len(cols_with_missing)} columns affected</div>
            </div>
        """, unsafe_allow_html=True)

    # Missing value warning if applicable
    if total_missing > 0:
        missing_details = [f"**{col}** ({df[col].isnull().sum()} missing)" for col in cols_with_missing]
        st.warning(f"⚠️ **Missing values detected** in: {', '.join(missing_details)}. Automatic median/most-frequent imputation will be applied.")

    if num_rows < 10:
        st.warning(f"⚠️ **Small Dataset Notice**: The dataset contains {num_rows} rows (fewer than 10). Model training will run for demonstration, but evaluation metrics should not be used for production assessment.")

    # Dataset Preview Accordion
    with st.expander("🔍 Dataset Preview & Column Schema", expanded=True):
        col_prev, col_info = st.columns([0.7, 0.3])
        with col_prev:
            st.markdown("**First Rows:**")
            st.dataframe(df.head(10), use_container_width=True)
        with col_info:
            st.markdown("**Column Metadata:**")
            schema_df = pd.DataFrame({
                "Column": df.columns,
                "Type": [str(t) for t in df.dtypes],
                "Nulls": df.isnull().sum().values,
                "Unique": df.nunique().values
            })
            st.dataframe(schema_df, use_container_width=True, hide_index=True)


def render_best_model_hero(summary, results_dict):
    """Render prominent best model showcase card."""
    best_model_name = summary.get("best_model")
    metric = summary.get("metric")
    problem_type = summary.get("problem_type")
    comparison_df = results_dict.get("comparison_df", pd.DataFrame())
    cv_results = results_dict.get("cv_results", {})
    tuning_results = results_dict.get("tuning_results", {})
    optimize = results_dict.get("optimize", False)

    if not best_model_name or comparison_df.empty:
        return

    # Extract score
    score = comparison_df.loc[best_model_name, metric] if (best_model_name in comparison_df.index and metric in comparison_df.columns) else None
    formatted_score = format_score(score, metric)
    metric_title = format_metric_name(metric)

    # Extract CV info
    cv_mean = comparison_df.loc[best_model_name, "cv_mean"] if (best_model_name in comparison_df.index and "cv_mean" in comparison_df.columns) else None
    cv_std = comparison_df.loc[best_model_name, "cv_std"] if (best_model_name in comparison_df.index and "cv_std" in comparison_df.columns) else None
    cv_text = f"{format_score(cv_mean)} (± {format_score(cv_std)})" if cv_mean is not None and not pd.isna(cv_mean) else "N/A"

    is_tuned = best_model_name in tuning_results
    opt_badge = f"<span class='hero-badge-accent'>⚡ Tuned ({results_dict.get('optimization_method', '').replace('_', ' ').title()})</span>" if is_tuned else "<span class='hero-badge'>Standard Parameters</span>"

    scaling_name = SCALING_OPTIONS.get(results_dict.get("scaling_strategy", "standard"), "Standard")
    encoding_name = ENCODING_OPTIONS.get(results_dict.get("encoding_strategy", "onehot"), "One-Hot")
    num_imp_name = NUMERICAL_IMPUTATION_OPTIONS.get(results_dict.get("numerical_strategy", "median"), "Median")
    cat_strat = results_dict.get("categorical_strategy", "most_frequent")
    cat_fill = results_dict.get("categorical_fill_value", "Unknown")
    cat_imp_name = f"Constant ('{cat_fill}')" if cat_strat == "constant" else CATEGORICAL_IMPUTATION_OPTIONS.get(cat_strat, "Most Frequent")

    badges_html = f"<span class='hero-badge'>Problem: {problem_type.title()}</span>"
    badges_html += f"<span class='hero-badge'>Metric: {metric_title}</span>"
    if results_dict.get("numerical_columns"):
        badges_html += f"<span class='hero-badge'>Num Imputation: {num_imp_name}</span>"
        badges_html += f"<span class='hero-badge'>Scaling: {scaling_name}</span>"
    if results_dict.get("categorical_columns"):
        badges_html += f"<span class='hero-badge'>Cat Imputation: {cat_imp_name}</span>"
        badges_html += f"<span class='hero-badge'>Encoding: {encoding_name}</span>"
    badges_html += opt_badge

    st.markdown(f"""
        <div class='hero-best-card'>
            <div class='hero-title'>🏆 Top Performing Model</div>
            <div class='hero-model-name'>{best_model_name}</div>
            <div style='margin-bottom:12px;'>
                {badges_html}
            </div>
            <div style='display:flex; flex-wrap:wrap; gap:24px; margin-top:16px; border-top:1px solid rgba(255,255,255,0.1); padding-top:16px;'>
                <div>
                    <div style='font-size:0.8rem; color:#94a3b8; text-transform:uppercase;'>Test Set {metric_title}</div>
                    <div style='font-size:1.8rem; font-weight:800; color:#38bdf8;'>{formatted_score}</div>
                </div>
                <div>
                    <div style='font-size:0.8rem; color:#94a3b8; text-transform:uppercase;'>Cross-Validation Score (Mean ± Std)</div>
                    <div style='font-size:1.8rem; font-weight:800; color:#f8fafc;'>{cv_text}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_comparison_leaderboard(comparison_df, metric, problem_type, best_model_name):
    """Render clean interactive model leaderboard table and comparison chart."""
    st.markdown("<div class='section-header'>📋 Model Leaderboard & Comparison</div>", unsafe_allow_html=True)

    if comparison_df.empty:
        st.info("No comparison data available.")
        return

    # Clean display dataframe
    display_df = comparison_df.copy()
    
    # Format numeric columns neatly
    for col in display_df.columns:
        if col in ["cv_mean", "cv_std"] or col in CLASSIFICATION_METRICS or col in REGRESSION_METRICS:
            display_df[col] = display_df[col].apply(lambda x: format_score(x))

    # Format best_params column for clean display
    if "best_params" in display_df.columns:
        display_df["best_params"] = display_df["best_params"].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else ("None" if pd.isna(x) else str(x))
        )

    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "cv_mean": st.column_config.TextColumn("CV Mean", help="Cross-Validation Mean Score"),
            "cv_std": st.column_config.TextColumn("CV Std Dev", help="Cross-Validation Standard Deviation"),
            "cv_folds": st.column_config.NumberColumn("CV Folds", help="Number of folds used"),
            "tuned": st.column_config.CheckboxColumn("Tuned", help="Was hyperparameter tuning applied?"),
            "best_params": st.column_config.TextColumn("Best Parameters", help="Optimal hyperparameters from search")
        }
    )

    # Plot comparison chart
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_comp = plot_model_comparison(comparison_df, metric, problem_type)
        if fig_comp:
            st.pyplot(fig_comp)
    with col_chart2:
        cv_results = st.session_state.get("automl_result", {}).get("cv_results", {})
        fig_cv = plot_cross_validation_scores(cv_results, metric)
        if fig_cv:
            st.pyplot(fig_cv)


def render_best_model_tab(results_dict):
    """Render details, parameters, and explanations for the best model."""
    best_model_name = results_dict.get("best_model")
    problem_type = results_dict.get("problem_type")
    comparison_df = results_dict.get("comparison_df", pd.DataFrame())
    tuning_results = results_dict.get("tuning_results", {})

    if not best_model_name:
        st.info("No best model selected.")
        return

    st.markdown(f"### 🏆 Model Profile: {best_model_name}")

    # Model metadata from config
    info_dict = CLASSIFICATION_MODEL_INFO if problem_type == "classification" else REGRESSION_MODEL_INFO
    model_meta = info_dict.get(best_model_name, {})

    col_meta1, col_meta2 = st.columns([0.6, 0.4])

    with col_meta1:
        if model_meta:
            st.markdown(f"**Description:** {model_meta.get('description', 'N/A')}")
            st.markdown(f"**When to use:** {model_meta.get('when_to_use', 'N/A')}")
            st.markdown(f"**Key Advantages:** {model_meta.get('advantages', 'N/A')}")
            st.markdown(f"**Limitations:** {model_meta.get('limitations', 'N/A')}")
            st.markdown(f"**Feature Scaling:** `{model_meta.get('scaling', 'Standard')}`")

    with col_meta2:
        st.markdown("**Test Set Evaluation Metrics:**")
        if best_model_name in comparison_df.index:
            metrics_list = CLASSIFICATION_METRICS if problem_type == "classification" else REGRESSION_METRICS
            metric_records = []
            for m in metrics_list:
                if m in comparison_df.columns:
                    val = comparison_df.loc[best_model_name, m]
                    metric_records.append({
                        "Metric": format_metric_name(m),
                        "Score": format_score(val, m)
                    })
            st.dataframe(pd.DataFrame(metric_records), use_container_width=True, hide_index=True)

    # Tuning details if tuned
    if best_model_name in tuning_results:
        tune_data = tuning_results[best_model_name]
        st.markdown("#### ⚡ Optimal Hyperparameters Discovered:")
        st.json(tune_data.get("best_params", {}))
        st.info(f"Cross-Validation Best Search Score: **{format_score(tune_data.get('best_score'))}** ({tune_data.get('cv_folds')} folds)")


def render_cross_validation_tab(results_dict):
    """Render dedicated Cross-Validation breakdown and fold performance."""
    st.markdown("### 🔄 Cross-Validation Analysis")
    st.caption("Cross-validation tests model consistency across multiple training/validation splits to guard against overfitting.")

    cv_results = results_dict.get("cv_results", {})
    cv_errors = results_dict.get("cv_errors", {})
    comparison_df = results_dict.get("comparison_df", pd.DataFrame())
    metric = results_dict.get("metric", "metric")

    if cv_errors:
        st.warning("⚠️ Cross-Validation Notices / Errors encountered for some models:")
        for model_name, err in cv_errors.items():
            st.markdown(f"- **{model_name}**: `{err}`")

    if not cv_results:
        st.info("No cross-validation results available.")
        return

    # Create CV Summary Table
    cv_summary_data = []
    for model_name, data in cv_results.items():
        scores_str = ", ".join([f"{s:.4f}" for s in data["scores"]])
        cv_summary_data.append({
            "Model": model_name,
            "CV Folds": data["cv_folds"],
            "CV Mean": format_score(data["mean_score"]),
            "CV Std Dev": format_score(data["std_score"]),
            "Fold Scores": scores_str
        })

    cv_summary_df = pd.DataFrame(cv_summary_data)
    st.dataframe(cv_summary_df, use_container_width=True, hide_index=True)

    # Test Set vs CV Mean Comparison Table
    if not comparison_df.empty and metric in comparison_df.columns:
        st.markdown(f"#### ⚖️ Test Set vs. CV Mean Comparison ({format_metric_name(metric)})")
        comp_subset = comparison_df[[metric, "cv_mean", "cv_std"]].copy()
        comp_subset.columns = ["Test Set Score", "CV Mean Score", "CV Std Dev"]
        for c in comp_subset.columns:
            comp_subset[c] = comp_subset[c].apply(format_score)
        st.dataframe(comp_subset, use_container_width=True)


def render_tuning_tab(results_dict):
    """Render detailed hyperparameter tuning results and search parameters."""
    st.markdown("### ⚙️ Hyperparameter Optimization Details")

    optimize = results_dict.get("optimize", False)
    if not optimize:
        st.info("ℹ️ Hyperparameter optimization was disabled for this run. All models were trained with baseline configurations.")
        return

    opt_method = results_dict.get("optimization_method", "grid_search").replace("_", " ").title()
    cv_folds = results_dict.get("cv_folds", 5)
    n_iter = results_dict.get("n_iter", 20)
    tuning_results = results_dict.get("tuning_results", {})
    tuning_errors = results_dict.get("tuning_errors", {})

    st.markdown(f"""
        - **Optimization Strategy:** `{opt_method}`
        - **Validation Strategy:** `{cv_folds}-Fold Cross-Validation`
        {f"- **Randomized Iterations:** `{n_iter}` combinations evaluated" if results_dict.get('optimization_method') == 'random_search' else ""}
    """)

    if tuning_errors:
        st.warning("⚠️ Some models encountered tuning errors:")
        for model_name, err in tuning_errors.items():
            st.markdown(f"- **{model_name}**: `{err}`")

    if tuning_results:
        st.markdown("#### 🎯 Tuned Models & Best Discovered Parameters")
        for model_name, res in tuning_results.items():
            with st.expander(f"🔹 {model_name} (CV Score: {format_score(res.get('best_score'))})", expanded=True):
                st.markdown(f"**Best Parameters:**")
                st.json(res.get("best_params", {}))
                st.markdown(f"**Metric Evaluated:** `{res.get('metric')}` | **Actual CV Folds:** `{res.get('cv_folds')}`")


def render_documentation_tab(problem_type):
    """Render educational guide for metrics and models."""
    st.markdown("### 📚 Machine Learning Knowledge Base")

    st.markdown("#### 📐 Metric Definitions & Guidance")
    metrics_list = CLASSIFICATION_METRICS if problem_type == "classification" else REGRESSION_METRICS

    for m_key in metrics_list:
        m_info = METRIC_INFO.get(m_key, {})
        with st.expander(f"🔹 {m_info.get('name', m_key)} ({'Higher is better' if m_info.get('direction') == 'higher' else 'Lower is better'})"):
            st.markdown(f"**Description:** {m_info.get('description', '')}")
            st.markdown(f"**When to use:** {m_info.get('when_to_use', '')}")
            st.markdown(f"**Warning/Trade-off:** {m_info.get('warning', '')}")

    st.markdown("#### 🤖 Supported Algorithms")
    models_info = CLASSIFICATION_MODEL_INFO if problem_type == "classification" else REGRESSION_MODEL_INFO

    for name, info in models_info.items():
        with st.expander(f"🔹 {name}"):
            st.markdown(f"**Description:** {info.get('description', '')}")
            st.markdown(f"**When to use:** {info.get('when_to_use', '')}")
            st.markdown(f"**Advantages:** {info.get('advantages', '')}")
            st.markdown(f"**Limitations:** {info.get('limitations', '')}")
            st.markdown(f"**Feature Scaling Requirement:** `{info.get('scaling', 'Standard')}`")


# ============================================================
# Main Application Flow
# ============================================================
def main():
    apply_custom_styles()
    render_header()

    # Session State Initialization
    if "automl_result" not in st.session_state:
        st.session_state["automl_result"] = None
    if "loaded_df" not in st.session_state:
        st.session_state["loaded_df"] = None
    if "loaded_filename" not in st.session_state:
        st.session_state["loaded_filename"] = None

    # ========================================================
    # Sidebar: Data Source & Configuration
    # ========================================================
    with st.sidebar:
        st.markdown("### 📁 1. Dataset Selection")

        sample_options = {
            "None (Upload Custom CSV)": None,
            "Classification Demo (test.csv)": "data/test.csv",
            "Multiclass Classification Demo (iris.csv)": "data/iris.csv",
            "Titanic Mixed Demo (titanic_test.csv)": "data/titanic_test.csv",
            "Regression Demo (regression_test.csv)": "data/regression_test.csv",
            "Missing Values Classification (missing_values_test.csv)": "data/missing_values_test.csv",
            "Missing Values Regression (missing_regression_test.csv)": "data/missing_regression_test.csv",
            "Small Dataset Demo (small_test.csv)": "data/small_test.csv"
        }

        selected_sample = st.selectbox(
            "Load Sample or Upload File:",
            options=list(sample_options.keys()),
            index=0,
            help="Choose a pre-packaged benchmark dataset or upload your own CSV."
        )

        uploaded_file = None
        if selected_sample == "None (Upload Custom CSV)":
            uploaded_file = st.file_uploader("Upload CSV Dataset:", type=["csv"])
            if uploaded_file is not None:
                try:
                    df_uploaded = pd.read_csv(uploaded_file)
                    st.session_state["loaded_df"] = df_uploaded
                    st.session_state["loaded_filename"] = uploaded_file.name
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
            else:
                if st.session_state.get("loaded_filename") not in [None, "Custom Upload"]:
                    # Cleared upload
                    st.session_state["loaded_df"] = None
                    st.session_state["loaded_filename"] = None
        else:
            sample_path = sample_options[selected_sample]
            if os.path.exists(sample_path):
                st.session_state["loaded_df"] = load_data(sample_path)
                st.session_state["loaded_filename"] = os.path.basename(sample_path)
            else:
                st.error(f"Sample file not found at {sample_path}")

        current_df = st.session_state.get("loaded_df")
        current_filename = st.session_state.get("loaded_filename", "No file loaded")

        st.divider()

        # ====================================================
        # Sidebar: Problem & Target Configuration
        # ====================================================
        st.markdown("### ⚙️ 2. Problem Setup")

        if current_df is not None and not current_df.empty:
            columns_list = list(current_df.columns)
            default_target_idx = len(columns_list) - 1  # Often the last column
            target_col = st.selectbox(
                "Target Column:",
                options=columns_list,
                index=default_target_idx,
                help="Select the column you want the machine learning models to predict."
            )

            # Determine default problem type based on target column values or user selection
            is_probably_reg = False
            if target_col in current_df.columns:
                target_series = current_df[target_col].dropna()
                if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 10:
                    is_probably_reg = True

            problem_type_keys = list(PROBLEM_TYPES.keys())  # ["classification", "regression"]
            default_prob_idx = 1 if is_probably_reg else 0

            problem_type = st.radio(
                "Problem Type:",
                options=problem_type_keys,
                format_func=lambda x: PROBLEM_TYPES[x],
                index=default_prob_idx,
                help="Classification predicts discrete classes/categories. Regression predicts continuous numerical quantities."
            )

            st.divider()

            # ================================================
            # Sidebar: Preprocessing Settings
            # ================================================
            st.markdown("### 🎛️ 3. Preprocessing Settings")

            features_preview = current_df.drop(columns=[target_col]) if target_col in current_df.columns else current_df
            num_cols_preview = features_preview.select_dtypes(include="number").columns.tolist()
            cat_cols_preview = features_preview.select_dtypes(include="object").columns.tolist()

            # Numerical Preprocessing controls
            if num_cols_preview:
                st.markdown("**Numerical Preprocessing:**")
                numerical_strategy = st.selectbox(
                    "Numerical Imputation:",
                    options=list(NUMERICAL_IMPUTATION_OPTIONS.keys()),
                    index=list(NUMERICAL_IMPUTATION_OPTIONS.keys()).index(DEFAULT_NUMERICAL_IMPUTATION),
                    format_func=lambda x: NUMERICAL_IMPUTATION_OPTIONS[x],
                    help="Strategy for imputing missing numerical values (fitted on training data only)."
                )
                scaling_strategy = st.selectbox(
                    "Feature Scaling Strategy:",
                    options=list(SCALING_OPTIONS.keys()),
                    index=list(SCALING_OPTIONS.keys()).index(DEFAULT_SCALING),
                    format_func=lambda x: SCALING_OPTIONS[x],
                    help="Strategy used to scale numerical features (fitted on training data only)."
                )
            else:
                numerical_strategy = DEFAULT_NUMERICAL_IMPUTATION
                scaling_strategy = DEFAULT_SCALING
                st.caption("ℹ️ *No numerical feature columns detected.*")

            # Categorical Preprocessing controls
            if cat_cols_preview:
                st.markdown("**Categorical Preprocessing:**")
                categorical_strategy = st.selectbox(
                    "Categorical Imputation:",
                    options=list(CATEGORICAL_IMPUTATION_OPTIONS.keys()),
                    index=list(CATEGORICAL_IMPUTATION_OPTIONS.keys()).index(DEFAULT_CATEGORICAL_IMPUTATION),
                    format_func=lambda x: CATEGORICAL_IMPUTATION_OPTIONS[x],
                    help="Strategy for imputing missing categorical values (fitted on training data only)."
                )
                categorical_fill_value = DEFAULT_CATEGORICAL_FILL_VALUE
                if categorical_strategy == "constant":
                    categorical_fill_value = st.text_input(
                        "Categorical Constant Replacement Value:",
                        value=DEFAULT_CATEGORICAL_FILL_VALUE,
                        help="Custom string value used to replace missing categorical entries."
                    )
                encoding_strategy = st.selectbox(
                    "Categorical Encoding Strategy:",
                    options=list(ENCODING_OPTIONS.keys()),
                    index=list(ENCODING_OPTIONS.keys()).index(DEFAULT_ENCODING),
                    format_func=lambda x: ENCODING_OPTIONS[x],
                    help="Strategy used to encode categorical features into numerical representations."
                )
            else:
                categorical_strategy = DEFAULT_CATEGORICAL_IMPUTATION
                categorical_fill_value = DEFAULT_CATEGORICAL_FILL_VALUE
                encoding_strategy = DEFAULT_ENCODING
                st.caption("ℹ️ *No categorical feature columns detected.*")

            st.divider()

            # ================================================
            # Sidebar: Training Settings
            # ================================================
            st.markdown("### 🛠️ 4. Training Settings")

            # Test Size
            test_size = st.selectbox(
                "Test Split Ratio:",
                options=TEST_SIZE_OPTIONS,
                index=TEST_SIZE_OPTIONS.index(DEFAULT_TEST_SIZE),
                format_func=lambda x: f"{int(x * 100)}% Test / {int((1 - x) * 100)}% Train",
                help="Fraction of dataset reserved for evaluating final model performance."
            )

            # Metric Selection
            avail_metrics = CLASSIFICATION_METRICS if problem_type == "classification" else REGRESSION_METRICS
            default_metric = DEFAULT_CLASSIFICATION_METRIC if problem_type == "classification" else DEFAULT_REGRESSION_METRIC
            
            metric = st.selectbox(
                "Optimization & Evaluation Metric:",
                options=avail_metrics,
                index=avail_metrics.index(default_metric) if default_metric in avail_metrics else 0,
                format_func=format_metric_name,
                help="The primary metric used to rank models and determine the top performer."
            )

            # CV Folds
            cv_folds = st.selectbox(
                "Cross-Validation Folds:",
                options=CV_FOLD_OPTIONS,
                index=CV_FOLD_OPTIONS.index(DEFAULT_CV_FOLDS),
                help="Number of folds used for internal cross-validation."
            )

            # Hyperparameter Tuning
            st.divider()
            st.markdown("### ⚡ 5. Hyperparameter Tuning")

            optimize = st.toggle(
                "Enable Hyperparameter Optimization",
                value=False,
                help="When enabled, systematically searches for optimal model hyperparameters using cross-validation."
            )

            opt_method = DEFAULT_OPTIMIZATION_METHOD
            n_iter_val = DEFAULT_N_ITER

            if optimize:
                opt_method_keys = list(OPTIMIZATION_METHODS.keys())
                opt_method = st.selectbox(
                    "Search Method:",
                    options=opt_method_keys,
                    format_func=lambda x: OPTIMIZATION_METHODS[x],
                    index=opt_method_keys.index(DEFAULT_OPTIMIZATION_METHOD),
                    help="Grid Search tests all parameter combinations. Randomized Search samples random parameter configurations."
                )

                if opt_method == "random_search":
                    n_iter_val = st.number_input(
                        "Random Search Iterations (n_iter):",
                        min_value=1,
                        max_value=100,
                        value=DEFAULT_N_ITER,
                        step=1,
                        help="Number of parameter settings sampled during RandomizedSearchCV."
                    )

            st.divider()

            # Run Button
            run_clicked = st.button("🚀 Run AutoML Pipeline", type="primary", use_container_width=True)

        else:
            st.info("👈 Please upload a CSV dataset or select a sample dataset above to configure AutoML.")
            target_col = None
            problem_type = "classification"
            numerical_strategy = DEFAULT_NUMERICAL_IMPUTATION
            categorical_strategy = DEFAULT_CATEGORICAL_IMPUTATION
            categorical_fill_value = DEFAULT_CATEGORICAL_FILL_VALUE
            scaling_strategy = DEFAULT_SCALING
            encoding_strategy = DEFAULT_ENCODING
            test_size = DEFAULT_TEST_SIZE
            metric = DEFAULT_CLASSIFICATION_METRIC
            cv_folds = DEFAULT_CV_FOLDS
            optimize = False
            opt_method = DEFAULT_OPTIMIZATION_METHOD
            n_iter_val = DEFAULT_N_ITER
            run_clicked = False

    # ========================================================
    # Main Body: Empty State vs Active Execution
    # ========================================================
    if current_df is None or current_df.empty:
        # Welcome Landing Page
        st.markdown("""
            <div style='text-align:center; padding: 40px 20px;'>
                <h2 style='color:#1e293b; font-weight:800;'>Welcome to AutoML Studio</h2>
                <p style='color:#64748b; font-size:1.1rem; max-width:650px; margin:0 auto 30px auto;'>
                    An end-to-end Automated Machine Learning platform. Upload any tabular dataset or load a demo dataset from the sidebar to automatically preprocess features, train multiple algorithms, run cross-validation, and find the best model.
                </p>
            </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("""
                <div class='metric-card'>
                    <div class='metric-label'>Step 1</div>
                    <div style='font-size:1.1rem; font-weight:700; margin:6px 0;'>Upload Dataset</div>
                    <div class='metric-subtext'>Upload a CSV file with tabular features and a target column.</div>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
                <div class='metric-card'>
                    <div class='metric-label'>Step 2</div>
                    <div style='font-size:1.1rem; font-weight:700; margin:6px 0;'>Configure Problem</div>
                    <div class='metric-subtext'>Select classification or regression, target column, and primary metric.</div>
                </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
                <div class='metric-card'>
                    <div class='metric-label'>Step 3</div>
                    <div style='font-size:1.1rem; font-weight:700; margin:6px 0;'>Automated Training</div>
                    <div class='metric-subtext'>Pipelines handle missing data, encoding, scaling, and hyperparameter tuning.</div>
                </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown("""
                <div class='metric-card'>
                    <div class='metric-label'>Step 4</div>
                    <div style='font-size:1.1rem; font-weight:700; margin:6px 0;'>Compare & Analyze</div>
                    <div class='metric-subtext'>Inspect leaderboards, cross-validation metrics, and top model parameters.</div>
                </div>
            """, unsafe_allow_html=True)
        return

    # Dataset is loaded: Render Dataset Overview
    render_dataset_overview(current_df, current_filename, target_col)

    # ========================================================
    # AutoML Execution Trigger
    # ========================================================
    if run_clicked:
        # Pre-execution validation check
        val_errors, val_warnings = validate_dataset(current_df, target_col)
        cfg_errors = validate_automl_config(
            problem_type=problem_type,
            metric=metric,
            test_size=test_size,
            optimize=optimize,
            optimization_method=opt_method,
            cv_folds=cv_folds,
            n_iter=n_iter_val,
            numerical_strategy=numerical_strategy,
            categorical_strategy=categorical_strategy,
            scaling_strategy=scaling_strategy,
            encoding_strategy=encoding_strategy,
            categorical_fill_value=categorical_fill_value
        )

        all_blocking_errors = val_errors + cfg_errors

        if all_blocking_errors:
            st.error("❌ **Cannot run AutoML due to validation errors:**")
            for err in all_blocking_errors:
                st.markdown(f"- {err}")
        else:
            with st.spinner("🤖 Running AutoML: preprocessing data, training models, running cross-validation, and evaluating metrics..."):
                try:
                    result = run_automl(
                        df=current_df,
                        target_column=target_col,
                        problem_type=problem_type,
                        test_size=test_size,
                        random_state=DEFAULT_RANDOM_STATE,
                        metric=metric,
                        numerical_strategy=numerical_strategy,
                        categorical_strategy=categorical_strategy,
                        categorical_fill_value=categorical_fill_value,
                        scaling_strategy=scaling_strategy,
                        encoding_strategy=encoding_strategy,
                        optimize=optimize,
                        optimization_method=opt_method,
                        cv_folds=cv_folds,
                        n_iter=int(n_iter_val)
                    )
                    st.session_state["automl_result"] = result
                    
                    # Fix Issue 2: Only show success message when execution actually succeeded
                    if result.get("summary", {}).get("success", False):
                        st.success("✅ AutoML execution completed successfully!")
                    else:
                        st.error("❌ AutoML execution did not complete successfully.")
                except Exception as ex:
                    st.error(f"❌ An unexpected error occurred during AutoML execution: {ex}")
                    st.session_state["automl_result"] = None

    # ========================================================
    # Results Dashboard
    # ========================================================
    automl_result = st.session_state.get("automl_result")

    if automl_result is not None:
        summary = automl_result.get("summary", {})
        success = summary.get("success", False)

        if not success:
            st.error("❌ **AutoML Run Failed**")
            errs = summary.get("errors", []) or automl_result.get("errors", [])
            if errs:
                for e in errs:
                    st.markdown(f"- {e}")
            return

        st.markdown("<div class='section-header'>🏁 AutoML Results & Diagnostics</div>", unsafe_allow_html=True)

        # 1. Best Model Showcase Card
        render_best_model_hero(summary, automl_result)

        # 2. Warnings and Non-Fatal Model Errors if any
        run_warnings = automl_result.get("warnings", [])
        training_errors = automl_result.get("training_errors", {})
        cv_errors = automl_result.get("cv_errors", {})
        tuning_errors = automl_result.get("tuning_errors", {})

        if run_warnings or training_errors or cv_errors or tuning_errors:
            with st.expander("⚠️ Execution Warnings & Model Diagnostics", expanded=False):
                if run_warnings:
                    st.markdown("**Dataset Warnings:**")
                    for w in run_warnings:
                        st.markdown(f"- {w}")
                if training_errors:
                    st.markdown("**Model Training Errors:**")
                    for m, err in training_errors.items():
                        st.markdown(f"- **{m}**: `{err}`")
                if cv_errors:
                    st.markdown("**Cross-Validation Errors:**")
                    for m, err in cv_errors.items():
                        st.markdown(f"- **{m}**: `{err}`")
                if tuning_errors:
                    st.markdown("**Tuning Errors:**")
                    for m, err in tuning_errors.items():
                        st.markdown(f"- **{m}**: `{err}`")

        # 3. Leaderboard & Model Comparison
        comparison_df = automl_result.get("comparison_df", pd.DataFrame())
        best_model_name = automl_result.get("best_model")
        render_comparison_leaderboard(comparison_df, metric, problem_type, best_model_name)

        # 4. In-depth Tabs
        tab_best, tab_cv, tab_tuning, tab_docs = st.tabs([
            "⭐ Top Model Profile",
            "🔄 Cross-Validation Breakdown",
            "⚙️ Hyperparameter Tuning",
            "📚 Metric & Algorithm Guide"
        ])

        with tab_best:
            render_best_model_tab(automl_result)

        with tab_cv:
            render_cross_validation_tab(automl_result)

        with tab_tuning:
            render_tuning_tab(automl_result)

        with tab_docs:
            render_documentation_tab(problem_type)


if __name__ == "__main__":
    main()
