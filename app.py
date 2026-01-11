"""
Prophet Parameter Playground
A Streamlit app to teach users how to master Facebook Prophet's hyperparameters.
"""

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

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
# DATA GENERATION FUNCTIONS
# =============================================================================
@st.cache_data
def generate_data(scenario_type: str, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic time series data for different learning scenarios.
    
    Args:
        scenario_type: One of the predefined scenario types
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with 'ds' (dates) and 'y' (values) columns
    """
    np.random.seed(seed)
    
    # Create 365 days of daily data
    dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
    t = np.arange(365)
    
    if scenario_type == "Baseline":
        # Linear trend + weekly seasonality + low noise
        trend = 50 + 0.1 * t
        weekly = 5 * np.sin(2 * np.pi * t / 7)
        noise = np.random.normal(0, 2, 365)
        y = trend + weekly + noise
        
    elif scenario_type == "Overfitting Trap":
        # Linear trend + very high random noise
        trend = 50 + 0.05 * t
        weekly = 3 * np.sin(2 * np.pi * t / 7)
        noise = np.random.normal(0, 15, 365)  # High noise!
        y = trend + weekly + noise
        
    elif scenario_type == "The Shock":
        # Flat trend that jumps by 50% at day 250
        trend = np.where(t < 250, 50, 75)  # 50% jump
        weekly = 3 * np.sin(2 * np.pi * t / 7)
        noise = np.random.normal(0, 2, 365)
        y = trend + weekly + noise
        
    elif scenario_type == "Saturating Growth":
        # Logistic/S-curve growth that flattens
        cap = 100
        k = 0.03  # Growth rate
        m = 150   # Midpoint
        trend = cap / (1 + np.exp(-k * (t - m)))
        weekly = 2 * np.sin(2 * np.pi * t / 7)
        noise = np.random.normal(0, 2, 365)
        y = trend + weekly + noise
        
    elif scenario_type == "Multiplicative Seasonality":
        # Seasonality amplitude grows with trend
        trend = 20 + 0.2 * t
        # Multiplicative: seasonality is proportional to trend
        weekly = trend * 0.15 * np.sin(2 * np.pi * t / 7)
        noise = np.random.normal(0, 2, 365)
        y = trend + weekly + noise
        
    else:
        raise ValueError(f"Unknown scenario: {scenario_type}")
    
    df = pd.DataFrame({"ds": dates, "y": y})
    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Split data into training and test sets.
    
    Args:
        df: Full dataset
        train_ratio: Proportion of data for training
        
    Returns:
        Tuple of (train_df, test_df, split_date)
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    split_date = df.iloc[split_idx]["ds"]
    return train_df, test_df, split_date


# =============================================================================
# MODEL FITTING
# =============================================================================
@st.cache_resource
def fit_prophet_model(
    _train_df: pd.DataFrame,
    growth: str,
    cap: float,
    changepoint_prior_scale: float,
    n_changepoints: int,
    seasonality_mode: str,
    seasonality_prior_scale: float,
    weekly_seasonality: bool,
    yearly_seasonality: bool,
    holidays_prior_scale: float,
    scenario_hash: str  # For cache invalidation
):
    """
    Fit a Prophet model with the specified parameters.
    
    Args:
        _train_df: Training data (underscore prefix to avoid hashing)
        growth: 'linear' or 'logistic'
        cap: Capacity for logistic growth
        changepoint_prior_scale: Flexibility of trend changes
        n_changepoints: Number of potential changepoints
        seasonality_mode: 'additive' or 'multiplicative'
        seasonality_prior_scale: Flexibility of seasonality
        weekly_seasonality: Include weekly seasonality
        yearly_seasonality: Include yearly seasonality
        holidays_prior_scale: Flexibility of holiday effects
        scenario_hash: Hash for cache invalidation
        
    Returns:
        Fitted Prophet model
    """
    train_data = _train_df.copy()
    
    # Add cap/floor for logistic growth
    if growth == "logistic":
        train_data["cap"] = cap
        train_data["floor"] = 0
    
    model = Prophet(
        growth=growth,
        changepoint_prior_scale=changepoint_prior_scale,
        n_changepoints=n_changepoints,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=False,
        holidays_prior_scale=holidays_prior_scale,
    )
    
    model.fit(train_data)
    return model


def create_forecast(model, df: pd.DataFrame, growth: str, cap: float):
    """
    Create forecast covering both training and test periods.
    
    Args:
        model: Fitted Prophet model
        df: Full dataset (for date range)
        growth: Growth type
        cap: Capacity for logistic growth
        
    Returns:
        Forecast DataFrame
    """
    future = model.make_future_dataframe(periods=len(df) - len(model.history), freq="D")
    
    if growth == "logistic":
        future["cap"] = cap
        future["floor"] = 0
    
    forecast = model.predict(future)
    return forecast


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_main_chart(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast: pd.DataFrame,
    model,
    show_changepoints: bool,
    split_date
) -> go.Figure:
    """
    Create the main forecast visualization with Plotly.
    
    Args:
        train_df: Training data
        test_df: Test data
        forecast: Prophet forecast
        model: Fitted Prophet model
        show_changepoints: Whether to show changepoint lines
        split_date: Date where train/test split occurs
        
    Returns:
        Plotly figure
    """
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
    
    # Train/Test split line - using shape instead of vline for compatibility
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
    if show_changepoints and hasattr(model, "changepoints") and len(model.changepoints) > 0:
        for cp in model.changepoints:
            fig.add_shape(
                type="line",
                x0=cp,
                x1=cp,
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


def create_components_chart(model, forecast: pd.DataFrame, growth: str = "linear", cap: float = 100.0) -> go.Figure:
    """
    Create a components plot showing trend and seasonality.
    
    Args:
        model: Fitted Prophet model
        forecast: Prophet forecast
        
    Returns:
        Plotly figure with subplots
    """
    components = ["trend"]
    if model.weekly_seasonality:
        components.append("weekly")
    if model.yearly_seasonality:
        components.append("yearly")
    
    n_components = len(components)
    fig = make_subplots(
        rows=n_components,
        cols=1,
        subplot_titles=[c.title() for c in components],
        vertical_spacing=0.1
    )
    
    row = 1
    
    # Trend
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["trend"],
            mode="lines",
            line=dict(color="blue"),
            name="Trend"
        ),
        row=row, col=1
    )
    row += 1
    
    # Weekly seasonality
    if model.weekly_seasonality and "weekly" in forecast.columns:
        # Create a week's worth of data for visualization
        days = pd.date_range(start="2024-01-01", periods=7, freq="D")
        weekly_pred_df = pd.DataFrame({"ds": days})
        if growth == "logistic":
            weekly_pred_df["cap"] = cap
            weekly_pred_df["floor"] = 0
        weekly_df = model.predict(weekly_pred_df)
        
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
    
    # Yearly seasonality
    if model.yearly_seasonality and "yearly" in forecast.columns:
        # Sample yearly pattern
        year_days = pd.date_range(start="2024-01-01", periods=365, freq="D")
        yearly_pred_df = pd.DataFrame({"ds": year_days})
        if growth == "logistic":
            yearly_pred_df["cap"] = cap
            yearly_pred_df["floor"] = 0
        yearly_df = model.predict(yearly_pred_df)
        
        fig.add_trace(
            go.Scatter(
                x=yearly_df["ds"],
                y=yearly_df["yearly"],
                mode="lines",
                line=dict(color="orange"),
                name="Yearly"
            ),
            row=row, col=1
        )
    
    fig.update_layout(
        height=200 * n_components,
        showlegend=False,
        title_text="Forecast Components"
    )
    
    return fig


# =============================================================================
# METRICS CALCULATION
# =============================================================================
def calculate_metrics(train_df, test_df, forecast):
    """
    Calculate training and test MAE.
    
    Args:
        train_df: Training data
        test_df: Test data
        forecast: Prophet forecast
        
    Returns:
        Tuple of (train_mae, test_mae)
    """
    # Merge forecast with actual data
    train_forecast = forecast[forecast["ds"].isin(train_df["ds"])]
    test_forecast = forecast[forecast["ds"].isin(test_df["ds"])]
    
    train_merged = train_df.merge(train_forecast[["ds", "yhat"]], on="ds")
    test_merged = test_df.merge(test_forecast[["ds", "yhat"]], on="ds")
    
    train_mae = mean_absolute_error(train_merged["y"], train_merged["yhat"])
    test_mae = mean_absolute_error(test_merged["y"], test_merged["yhat"])
    
    return train_mae, test_mae


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.title("📈 Prophet Parameter Playground")
    st.markdown("*Learn how to master Facebook Prophet's hyperparameters through interactive experimentation*")
    
    # =========================================================================
    # SIDEBAR CONTROLS
    # =========================================================================
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Reset button
        if st.button("🔄 Reset All Parameters", width="stretch"):
            st.cache_resource.clear()
            st.rerun()
        
        st.divider()
        
        # Scenario Selection
        st.subheader("📊 Data Scenario")
        scenario = st.selectbox(
            "Choose a learning scenario:",
            options=list(SCENARIO_DESCRIPTIONS.keys()),
            index=0
        )
        
        st.divider()
        
        # Growth Parameters
        st.subheader("📈 Growth")
        growth = st.radio(
            "Growth Type:",
            options=["linear", "logistic"],
            index=0,
            horizontal=True
        )
        
        cap = 100.0
        if growth == "logistic":
            cap = st.slider(
                "Cap (Maximum Value):",
                min_value=50.0,
                max_value=200.0,
                value=100.0,
                step=5.0,
                help="The maximum achievable value for logistic growth"
            )
        
        st.divider()
        
        # Trend Parameters
        st.subheader("📉 Trend Flexibility")
        changepoint_prior_scale = st.select_slider(
            "Changepoint Prior Scale:",
            options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
            value=0.05,
            help="Higher = more flexible trend (risk of overfitting)"
        )
        
        n_changepoints = st.slider(
            "Number of Changepoints:",
            min_value=5,
            max_value=50,
            value=25,
            step=5,
            help="Potential places where trend can change"
        )
        
        st.divider()
        
        # Seasonality Parameters
        st.subheader("🌊 Seasonality")
        seasonality_mode = st.radio(
            "Seasonality Mode:",
            options=["additive", "multiplicative"],
            index=0,
            horizontal=True,
            help="Additive: constant amplitude. Multiplicative: amplitude scales with trend."
        )
        
        seasonality_prior_scale = st.select_slider(
            "Seasonality Prior Scale:",
            options=[0.01, 0.1, 1.0, 5.0, 10.0],
            value=10.0,
            help="Higher = more flexible seasonality"
        )
        
        st.divider()
        
        # Seasonality Components
        st.subheader("📅 Seasonality Components")
        weekly_seasonality = st.checkbox("Weekly Seasonality", value=True)
        yearly_seasonality = st.checkbox("Yearly Seasonality", value=False)
        
        st.divider()
        
        # Holidays
        st.subheader("🎉 Holidays")
        holidays_prior_scale = st.slider(
            "Holidays Prior Scale:",
            min_value=0.01,
            max_value=20.0,
            value=10.0,
            step=0.5,
            help="Flexibility of holiday effects"
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
    
    # Generate and split data
    df = generate_data(scenario)
    train_df, test_df, split_date = split_data(df)
    
    # Create a hash for cache invalidation
    scenario_hash = f"{scenario}_{growth}_{cap}_{changepoint_prior_scale}_{n_changepoints}"
    scenario_hash += f"_{seasonality_mode}_{seasonality_prior_scale}_{weekly_seasonality}"
    scenario_hash += f"_{yearly_seasonality}_{holidays_prior_scale}"
    
    # Fit model
    with st.spinner("Fitting Prophet model..."):
        model = fit_prophet_model(
            _train_df=train_df,
            growth=growth,
            cap=cap,
            changepoint_prior_scale=changepoint_prior_scale,
            n_changepoints=n_changepoints,
            seasonality_mode=seasonality_mode,
            seasonality_prior_scale=seasonality_prior_scale,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            holidays_prior_scale=holidays_prior_scale,
            scenario_hash=scenario_hash
        )
    
    # Create forecast
    forecast = create_forecast(model, df, growth, cap)
    
    # Calculate metrics
    train_mae, test_mae = calculate_metrics(train_df, test_df, forecast)
    
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
        # Color code based on overfitting
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
        # Overfitting indicator
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
        model=model,
        show_changepoints=show_changepoints,
        split_date=split_date
    )
    st.plotly_chart(main_chart, width="stretch")
    
    # =========================================================================
    # COMPONENTS CHART
    # =========================================================================
    st.subheader("🔍 Forecast Components")
    
    components_chart = create_components_chart(model, forecast, growth, cap)
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
