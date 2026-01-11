---
title: Prophet Parameter Playground
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# Prophet Parameter Playground

An interactive Streamlit app to teach users how to master Facebook Prophet's hyperparameters through hands-on experimentation.

## Features

- **5 Learning Scenarios**: Baseline, Overfitting Trap, The Shock, Saturating Growth, Multiplicative Seasonality
- **Interactive Parameter Controls**: Adjust changepoint prior scale, seasonality mode, growth type, and more
- **Train/Test Split Visualization**: See how well your model generalizes to unseen data
- **Real-time Metrics**: Compare Training MAE vs Test MAE to detect overfitting
- **Component Plots**: Visualize trend and seasonality components

## How It Works

This app uses **precomputed Prophet forecasts** to provide instant feedback without requiring Prophet to be installed on the server. All parameter combinations have been pre-generated locally.

## Usage

1. Select a **Data Scenario** from the sidebar
2. Adjust the **hyperparameters** using the sliders and radio buttons
3. Observe how the forecast (blue line) compares to the test data (red dots)
4. Watch the **Training MAE** and **Test MAE** metrics to understand model performance

## Scenarios

| Scenario | Description | What to Learn |
|----------|-------------|---------------|
| **Baseline** | Linear trend + weekly seasonality + low noise | Default parameters work well |
| **Overfitting Trap** | High random noise | Regularization with low changepoint_prior_scale |
| **The Shock** | Sudden 50% jump | Changepoint detection sensitivity |
| **Saturating Growth** | Logistic S-curve | Using cap/floor with logistic growth |
| **Multiplicative Seasonality** | Seasonality scales with trend | Additive vs Multiplicative mode |

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

To regenerate precomputed data (requires Prophet):
```bash
pip install prophet scikit-learn
python precompute.py
```
