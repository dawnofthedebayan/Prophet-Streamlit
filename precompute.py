"""
Precompute Prophet forecasts for all scenario/parameter combinations.
Run this locally before deploying to Streamlit Cloud.

Usage: python precompute.py
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import pickle
import os
from itertools import product
import warnings

warnings.filterwarnings("ignore")


def _scenario_slug(s: str) -> str:
    return (
        s.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

# =============================================================================
# DATA GENERATION (same as app.py)
# =============================================================================
def generate_data(scenario_type: str, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic time series data for different learning scenarios."""
    np.random.seed(seed)
    
    dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
    t = np.arange(365)
    
    if scenario_type == "Baseline":
        trend = 50 + 0.1 * t
        weekly = 5 * np.sin(2 * np.pi * t / 7)
        noise = np.random.normal(0, 2, 365)
        y = trend + weekly + noise
        
    elif scenario_type == "Overfitting Trap":
        trend = 50 + 0.05 * t
        weekly = 3 * np.sin(2 * np.pi * t / 7)
        noise = np.random.normal(0, 15, 365)
        y = trend + weekly + noise
        
    elif scenario_type == "The Shock":
        trend = np.where(t < 250, 50, 75)
        weekly = 3 * np.sin(2 * np.pi * t / 7)
        noise = np.random.normal(0, 2, 365)
        y = trend + weekly + noise
        
    elif scenario_type == "Saturating Growth":
        cap = 100
        k = 0.03
        m = 150
        trend = cap / (1 + np.exp(-k * (t - m)))
        weekly = 2 * np.sin(2 * np.pi * t / 7)
        noise = np.random.normal(0, 2, 365)
        y = trend + weekly + noise
        
    elif scenario_type == "Multiplicative Seasonality":
        trend = 20 + 0.2 * t
        weekly = trend * 0.15 * np.sin(2 * np.pi * t / 7)
        noise = np.random.normal(0, 2, 365)
        y = trend + weekly + noise
    else:
        raise ValueError(f"Unknown scenario: {scenario_type}")
    
    return pd.DataFrame({"ds": dates, "y": y})


def split_data(df: pd.DataFrame, train_ratio: float = 0.8):
    """Split data into training and test sets."""
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    split_date = df.iloc[split_idx]["ds"]
    return train_df, test_df, split_date


# =============================================================================
# PARAMETER GRID
# =============================================================================
SCENARIOS = [
    "Baseline",
    "Overfitting Trap", 
    "The Shock",
    "Saturating Growth",
    "Multiplicative Seasonality"
]

# Discrete parameter options for the grid
PARAM_GRID = {
    "changepoint_prior_scale": [0.001, 0.01, 0.05, 0.1, 0.3, 0.5],
    "n_changepoints": [10, 25, 40],
    "seasonality_mode": ["additive", "multiplicative"],
    "seasonality_prior_scale": [0.1, 1.0, 10.0],
    "growth": ["linear", "logistic"],
    "cap": [80.0, 100.0, 120.0],  # Only used for logistic
    "weekly_seasonality": [True],
    "yearly_seasonality": [False],
}


def fit_and_forecast(train_df, df, params):
    """Fit Prophet model and create forecast."""
    train_data = train_df.copy()
    
    growth = params["growth"]
    cap = params["cap"]
    
    if growth == "logistic":
        train_data["cap"] = cap
        train_data["floor"] = 0
    
    model = Prophet(
        growth=growth,
        changepoint_prior_scale=params["changepoint_prior_scale"],
        n_changepoints=params["n_changepoints"],
        seasonality_mode=params["seasonality_mode"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        weekly_seasonality=params["weekly_seasonality"],
        yearly_seasonality=params["yearly_seasonality"],
        daily_seasonality=False,
    )
    
    model.fit(train_data)
    
    # Create forecast
    future = model.make_future_dataframe(periods=len(df) - len(train_df), freq="D")
    if growth == "logistic":
        future["cap"] = cap
        future["floor"] = 0
    
    forecast = model.predict(future)
    
    # Get changepoints
    changepoints = list(model.changepoints) if hasattr(model, "changepoints") else []
    
    # Extract components for visualization
    components = {
        "trend": forecast[["ds", "trend"]].copy(),
    }
    
    if params["weekly_seasonality"] and "weekly" in forecast.columns:
        # Get weekly pattern
        days = pd.date_range(start="2024-01-01", periods=7, freq="D")
        weekly_pred_df = pd.DataFrame({"ds": days})
        if growth == "logistic":
            weekly_pred_df["cap"] = cap
            weekly_pred_df["floor"] = 0
        weekly_forecast = model.predict(weekly_pred_df)
        components["weekly"] = weekly_forecast[["ds", "weekly"]].copy()
    
    return forecast, changepoints, components


def calculate_metrics(train_df, test_df, forecast):
    """Calculate training and test MAE."""
    train_forecast = forecast[forecast["ds"].isin(train_df["ds"])]
    test_forecast = forecast[forecast["ds"].isin(test_df["ds"])]
    
    train_merged = train_df.merge(train_forecast[["ds", "yhat"]], on="ds")
    test_merged = test_df.merge(test_forecast[["ds", "yhat"]], on="ds")
    
    train_mae = mean_absolute_error(train_merged["y"], train_merged["yhat"])
    test_mae = mean_absolute_error(test_merged["y"], test_merged["yhat"])
    
    return train_mae, test_mae


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


def main():
    """Precompute all forecasts and save to disk."""
    os.makedirs("precomputed", exist_ok=True)
    
    # Generate all parameter combinations
    param_keys = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combinations = list(product(*param_values))
    
    # Filter out invalid combinations (cap only matters for logistic)
    valid_combinations = []
    for combo in all_combinations:
        params = dict(zip(param_keys, combo))
        # Skip cap variations for linear growth
        if params["growth"] == "linear" and params["cap"] != 100.0:
            continue
        valid_combinations.append(params)
    
    total = len(SCENARIOS) * len(valid_combinations)
    print(f"Precomputing {total} scenario/parameter combinations...")
    
    count = 0
    for scenario in SCENARIOS:
        print(f"\n📊 Processing scenario: {scenario}")
        
        # Generate data for this scenario
        df = generate_data(scenario)
        train_df, test_df, split_date = split_data(df)
        
        # Store data
        scenario_data = {
            "df": df,
            "train_df": train_df,
            "test_df": test_df,
            "split_date": split_date,
            "forecasts": {}
        }
        
        for params in valid_combinations:
            count += 1
            param_key = generate_param_key(scenario, params)
            
            try:
                forecast, changepoints, components = fit_and_forecast(train_df, df, params)
                train_mae, test_mae = calculate_metrics(train_df, test_df, forecast)
                
                # Store results
                scenario_data["forecasts"][param_key] = {
                    "params": params,
                    "forecast": forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].copy(),
                    "changepoints": [cp.isoformat() for cp in changepoints],
                    "components": components,
                    "train_mae": train_mae,
                    "test_mae": test_mae,
                }
                
                if count % 20 == 0:
                    print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")
                    
            except Exception as e:
                print(f"  ⚠️ Error for {param_key}: {e}")
                continue

        scenario_output_path = f"precomputed/scenario__{_scenario_slug(scenario)}.pkl"
        with open(scenario_output_path, "wb") as f:
            pickle.dump(scenario_data, f)

        print(f"✅ Saved {scenario} to {scenario_output_path}")
        print(f"   File size: {os.path.getsize(scenario_output_path) / 1024 / 1024:.2f} MB")
    
    # Also save parameter options for the app
    param_options = {
        "scenarios": SCENARIOS,
        "changepoint_prior_scale": PARAM_GRID["changepoint_prior_scale"],
        "n_changepoints": PARAM_GRID["n_changepoints"],
        "seasonality_mode": PARAM_GRID["seasonality_mode"],
        "seasonality_prior_scale": PARAM_GRID["seasonality_prior_scale"],
        "growth": PARAM_GRID["growth"],
        "cap": PARAM_GRID["cap"],
    }
    
    with open("precomputed/param_options.pkl", "wb") as f:
        pickle.dump(param_options, f)
    
    print("✅ Saved parameter options to precomputed/param_options.pkl")


if __name__ == "__main__":
    main()
