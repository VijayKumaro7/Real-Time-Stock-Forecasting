"""
conftest.py
-----------
Shared pytest fixtures used across all test modules.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic price data
# ---------------------------------------------------------------------------

@pytest.fixture
def price_series():
    """A deterministic 200-day daily close price series (flat uptrend)."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
    # Start at 100, add small daily noise + slight uptrend
    returns = np.random.normal(loc=0.001, scale=0.01, size=n)
    prices = 100 * np.cumprod(1 + returns)
    return pd.DataFrame({"Close": prices}, index=dates)


@pytest.fixture
def price_df_short(price_series):
    """A shorter (100-row) price DataFrame, sufficient for most indicator tests."""
    return price_series.iloc[:100].copy()


@pytest.fixture
def predictions_df():
    """Three-column DataFrame mimicking multi-model prediction output."""
    np.random.seed(0)
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "LSTM_Prediction": np.linspace(100, 110, 10),
            "XGBoost_Prediction": np.linspace(100, 108, 10),
            "ARIMA_Prediction": np.linspace(100, 105, 10),
        },
        index=dates,
    )
