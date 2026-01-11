"""
Prophet Parameter Playground - Cloud Version
A Streamlit app using precomputed Prophet results (no Prophet dependency needed).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Prophet Parameter Playground",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SCENARIO DESCRIPTIONS
# =============================================================================
SCENARIO_DESCRIPTIONS = {
    "Baseline": """
    **Baseline Scenario**: Linear trend + weekly seasonality + low noise.
    
    🎯 **What to explore**: This is a well-behaved dataset. Default parameters should work well.
    Try adjusting `changepoint_prior_scale` to see how it affects trend flexibility.
    """,
    
    "Overfitting Trap": """
    **Overfitting Trap**: Linear trend + very high random noise.
    
    🎯 **What to explore**: Increase `changepoint_prior_scale` and watch the Training MAE improve 
    while Test MAE gets worse—classic overfitting! Use lower values (0.001-0.01) for regularization.
    """,
    
    "The Shock": """
    **The Shock**: Flat trend that jumps instantly by 50% at day 250.
    
    🎯 **What to explore**: Increase `changepoint_prior_scale` (0.1-0.5) and `n_changepoints` 
    to help Prophet detect the sudden shift. Watch how the model struggles or adapts!
    """,
    
    "Saturating Growth": """
    **Saturating Growth**: Logistic curve that flattens out (S-curve).
    
    🎯 **What to explore**: Switch to **Logistic growth** and set an appropriate `cap`. 
    Linear growth will overpredict the future. The cap should be around 100-120.
    """,
    
    "Multiplicative Seasonality": """
    **Multiplicative Seasonality**: Seasonality amplitude grows as trend increases.
    
    🎯 **What to explore**: Switch `seasonality_mode` from Additive to **Multiplicative**. 
    Watch how the forecast uncertainty and fit improve dramatically!
    """
}


# =============================================================================
# LOAD PRECOMPUTED DATA
# =============================================================================
@st.cache_data
def load_param_options():
    """Load parameter options produced by precompute.py."""
    options_path = "precomputed/param_options.pkl"
    if not os.path.exists(options_path):
        st.error("⚠️ Precomputed options not found. Commit `precomputed/param_options.pkl`.")
        st.stop()

    with open(options_path, "rb") as f:
        options = pickle.load(f)
    return options


def _scenario_slug(s: str) -> str:
    return (
        s.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


@st.cache_data
def load_scenario_data(scenario: str):
    """Load one scenario's precomputed data (much lower memory than loading everything)."""
    per_scenario_path = f"precomputed/scenario__{_scenario_slug(scenario)}.pkl"
    legacy_path = "precomputed/prophet_results.pkl"

    if os.path.exists(per_scenario_path):
        with open(per_scenario_path, "rb") as f:
            return pickle.load(f)

    # Backward compatible fallback (can OOM on Streamlit Cloud)
    if os.path.exists(legacy_path):
        with open(legacy_path, "rb") as f:
            all_results = pickle.load(f)
        if scenario not in all_results:
            st.error(f"⚠️ Scenario '{scenario}' not found in legacy precomputed file.")
            st.stop()
        return all_results[scenario]

    st.error(
        "⚠️ Precomputed scenario data not found. "
        "Commit `precomputed/scenario__*.pkl` (preferred) or `precomputed/prophet_results.pkl` (legacy)."
    )
    st.stop()


def generate_param_key(scenario, params):
    """Generate a unique key for a scenario/parameter combination."""
    key_parts = [
        scenario,
        f"cps_{params['changepoint_prior_scale']}",
        f"ncp_{params['n_changepoints']}",
        f"sm_{params['seasonality_mode']}",
        f"sps_{params['seasonality_prior_scale']}",
        f"g_{params['growth']}",
    ]
    if params["growth"] == "logistic":
        key_parts.append(f"cap_{params['cap']}")
    return "__".join(key_parts)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_main_chart(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast: pd.DataFrame,
    changepoints: list,
    show_changepoints: bool,
    split_date
) -> go.Figure:
    """Create the main forecast visualization with Plotly."""
    fig = go.Figure()
    
    # Uncertainty interval (shaded area)
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(0, 100, 255, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Uncertainty Interval"
    ))
    
    # Model forecast line
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        line=dict(color="blue", width=2),
        name="Model Forecast"
    ))
    
    # Training data points
    fig.add_trace(go.Scatter(
        x=train_df["ds"],
        y=train_df["y"],
        mode="markers",
        marker=dict(color="black", size=5, opacity=0.7),
        name="Training Data"
    ))
    
    # Test data points
    fig.add_trace(go.Scatter(
        x=test_df["ds"],
        y=test_df["y"],
        mode="markers",
        marker=dict(color="red", size=6, opacity=0.8),
        name="Test Data (Unseen)"
    ))
    
    # Train/Test split line
    fig.add_shape(
        type="line",
        x0=split_date,
        x1=split_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", dash="dash", width=2)
    )
    fig.add_annotation(
        x=split_date,
        y=1.05,
        yref="paper",
        text="Train/Test Split",
        showarrow=False,
        font=dict(size=10, color="gray")
    )
    
    # Changepoints
    if show_changepoints and changepoints:
        for cp in changepoints:
            cp_ts = pd.Timestamp(cp)
            fig.add_shape(
                type="line",
                x0=cp_ts,
                x1=cp_ts,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="green", dash="dot", width=1),
                opacity=0.7
            )
        # Add a dummy trace for legend
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="green", dash="dot"),
            name="Changepoints"
        ))
    
    fig.update_layout(
        title="Prophet Forecast: Training Fit vs Test Prediction",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        height=500
    )
    
    return fig


def create_components_chart(components: dict) -> go.Figure:
    """Create a components plot showing trend and seasonality."""
    component_names = list(components.keys())
    n_components = len(component_names)
    
    fig = make_subplots(
        rows=n_components,
        cols=1,
        subplot_titles=[c.title() for c in component_names],
        vertical_spacing=0.15
    )
    
    row = 1
    
    # Trend
    if "trend" in components:
        trend_df = components["trend"]
        fig.add_trace(
            go.Scatter(
                x=trend_df["ds"],
                y=trend_df["trend"],
                mode="lines",
                line=dict(color="blue"),
                name="Trend"
            ),
            row=row, col=1
        )
        row += 1
    
    # Weekly seasonality
    if "weekly" in components:
        weekly_df = components["weekly"]
        fig.add_trace(
            go.Scatter(
                x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                y=weekly_df["weekly"].values,
                mode="lines+markers",
                line=dict(color="green"),
                name="Weekly"
            ),
            row=row, col=1
        )
        row += 1
    
    fig.update_layout(
        height=200 * n_components,
        showlegend=False,
        title_text="Forecast Components"
    )
    
    return fig


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.title("📈 Prophet Parameter Playground")
    st.markdown("*Learn how to master Facebook Prophet's hyperparameters through interactive experimentation*")
    
    # Load param options (small)
    options = load_param_options()
    
    # =========================================================================
    # SIDEBAR CONTROLS
    # =========================================================================
    with st.sidebar:
        st.header("🎛️ Controls")
        
        st.divider()
        
        # Scenario Selection
        st.subheader("📊 Data Scenario")
        scenario = st.selectbox(
            "Choose a learning scenario:",
            options=options["scenarios"],
            index=0
        )
        
        st.divider()
        
        # Growth Parameters
        st.subheader("📈 Growth")
        growth = st.radio(
            "Growth Type:",
            options=options["growth"],
            index=0,
            horizontal=True
        )
        
        cap = 100.0
        if growth == "logistic":
            cap = st.select_slider(
                "Cap (Maximum Value):",
                options=options["cap"],
                value=100.0,
                help="The maximum achievable value for logistic growth"
            )
        
        st.divider()
        
        # Trend Parameters
        st.subheader("📉 Trend Flexibility")
        changepoint_prior_scale = st.select_slider(
            "Changepoint Prior Scale:",
            options=options["changepoint_prior_scale"],
            value=0.05,
            help="Higher = more flexible trend (risk of overfitting)"
        )
        
        n_changepoints = st.select_slider(
            "Number of Changepoints:",
            options=options["n_changepoints"],
            value=25,
            help="Potential places where trend can change"
        )
        
        st.divider()
        
        # Seasonality Parameters
        st.subheader("🌊 Seasonality")
        seasonality_mode = st.radio(
            "Seasonality Mode:",
            options=options["seasonality_mode"],
            index=0,
            horizontal=True,
            help="Additive: constant amplitude. Multiplicative: amplitude scales with trend."
        )
        
        seasonality_prior_scale = st.select_slider(
            "Seasonality Prior Scale:",
            options=options["seasonality_prior_scale"],
            value=10.0,
            help="Higher = more flexible seasonality"
        )
        
        st.divider()
        
        # Display Options
        st.subheader("👁️ Display Options")
        show_changepoints = st.checkbox("Show Detected Changepoints", value=False)
    
    # =========================================================================
    # MAIN CONTENT
    # =========================================================================
    
    # Scenario Description
    with st.expander("📖 Scenario Description", expanded=True):
        st.markdown(SCENARIO_DESCRIPTIONS[scenario])
    
    # Get scenario data (lazy-loaded)
    scenario_data = load_scenario_data(scenario)
    train_df = scenario_data["train_df"]
    test_df = scenario_data["test_df"]
    split_date = scenario_data["split_date"]
    
    # Build parameter key
    params = {
        "changepoint_prior_scale": changepoint_prior_scale,
        "n_changepoints": n_changepoints,
        "seasonality_mode": seasonality_mode,
        "seasonality_prior_scale": seasonality_prior_scale,
        "growth": growth,
        "cap": cap,
    }
    param_key = generate_param_key(scenario, params)
    
    # Get forecast data
    if param_key not in scenario_data["forecasts"]:
        st.error(f"⚠️ This parameter combination was not precomputed. Try different settings.")
        st.stop()
    
    forecast_data = scenario_data["forecasts"][param_key]
    forecast = forecast_data["forecast"]
    changepoints = forecast_data["changepoints"]
    components = forecast_data["components"]
    train_mae = forecast_data["train_mae"]
    test_mae = forecast_data["test_mae"]
    
    # =========================================================================
    # METRICS DISPLAY
    # =========================================================================
    st.subheader("📊 Model Performance")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric(
            label="Training MAE (In-Sample)",
            value=f"{train_mae:.2f}",
            help="How well the model fits the training data"
        )
    
    with col2:
        overfitting_ratio = test_mae / train_mae if train_mae > 0 else 1
        delta_color = "normal" if overfitting_ratio < 1.5 else "inverse"
        
        st.metric(
            label="Test MAE (Out-of-Sample)",
            value=f"{test_mae:.2f}",
            delta=f"{((test_mae - train_mae) / train_mae * 100):.1f}% vs Train",
            delta_color=delta_color,
            help="How well the model predicts unseen data"
        )
    
    with col3:
        if overfitting_ratio < 1.2:
            st.success("✅ Good generalization!")
        elif overfitting_ratio < 1.5:
            st.warning("⚠️ Slight overfitting")
        else:
            st.error("🚨 Overfitting detected!")
    
    st.divider()
    
    # =========================================================================
    # MAIN CHART
    # =========================================================================
    st.subheader("📈 Forecast Visualization")
    
    main_chart = create_main_chart(
        train_df=train_df,
        test_df=test_df,
        forecast=forecast,
        changepoints=changepoints,
        show_changepoints=show_changepoints,
        split_date=split_date
    )
    st.plotly_chart(main_chart, width="stretch")
    
    # =========================================================================
    # COMPONENTS CHART
    # =========================================================================
    st.subheader("🔍 Forecast Components")
    
    components_chart = create_components_chart(components)
    st.plotly_chart(components_chart, width="stretch")
    
    # =========================================================================
    # FOOTER
    # =========================================================================
    st.divider()
    st.markdown("""
    ---
    **💡 Tips:**
    - **Red dots** are test data (unseen during training) - compare them to the blue forecast line
    - **Training MAE** shows fit quality; **Test MAE** shows prediction quality
    - If Test MAE >> Training MAE, you're overfitting!
    - Use the sidebar to experiment with different parameters
    """)


if __name__ == "__main__":
    main()
