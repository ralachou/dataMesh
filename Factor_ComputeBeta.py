import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import ace_tools as tools

# Simulate factor returns (daily returns for IG and HY over 250 days)
np.random.seed(42)
n_days = 250
factor_returns = pd.DataFrame({
    "IG": np.random.normal(0, 0.01, n_days),
    "HY": np.random.normal(0, 0.015, n_days)
})

# Simulate how each rating's "shock" (Spread change) moves over time
# True beta coefficients (for simulation purpose)
true_betas = {
    "AAA": [0.2, 0.0],
    "AA": [0.3, 0.0],
    "A": [0.6, 0.0],
    "BBB": [0.7, 0.1],
    "IG CDS": [1.0, 0.0],
    "BB": [0.1, 0.7],
    "B": [0.0, 0.8],
    "CCC": [0.0, 1.0],
    "NR": [0.0, 0.9],
    "HY CDS": [0.2, 1.0]
}

# Generate rating return time series from factor returns + noise
rating_returns = pd.DataFrame(index=factor_returns.index)
for rating, betas in true_betas.items():
    rating_returns[rating] = (
        betas[0] * factor_returns["IG"] +
        betas[1] * factor_returns["HY"] +
        np.random.normal(0, 0.002, n_days)  # add noise
    )

# Regress each rating's return on IG and HY to estimate beta
estimated_betas = []
for rating in rating_returns.columns:
    model = LinearRegression().fit(factor_returns, rating_returns[rating])
    estimated_betas.append({
        "Rating": rating,
        "Beta_IG": model.coef_[0],
        "Beta_HY": model.coef_[1]
    })

betas_df = pd.DataFrame(estimated_betas)
tools.display_dataframe_to_user(name="Estimated Betas from Rating Regressions", dataframe=betas_df)
