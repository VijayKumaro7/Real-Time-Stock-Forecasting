"""
forecasting.py
--------------
Pure business-logic functions extracted from the Streamlit dashboard so they
can be imported and tested independently of the UI layer.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def get_trend_indicator(current, previous):
    """Return (arrow, css_class, pct_change) for a price movement."""
    if current > previous:
        return "up", "bullish", (current - previous) / previous * 100
    elif current < previous:
        return "down", "bearish", (current - previous) / previous * 100
    else:
        return "flat", "neutral", 0.0


def generate_trading_signal(consensus_trend, confidence_avg, volatility):
    """
    Return a (signal_label, css_class) trading recommendation.

    Parameters
    ----------
    consensus_trend : float
        Percentage change predicted by the consensus of all models.
    confidence_avg : float
        Average confidence score across models (0-100).
    volatility : float
        Current market volatility percentage (unused in logic but kept for
        API compatibility).
    """
    if confidence_avg >= 70:
        if consensus_trend > 2:
            return "STRONG BUY", "alert-success"
        elif consensus_trend > 0.5:
            return "BUY", "alert-success"
        elif consensus_trend < -2:
            return "STRONG SELL", "alert-danger"
        elif consensus_trend < -0.5:
            return "SELL", "alert-danger"
        else:
            return "HOLD", "alert-warning"
    else:
        return "UNCERTAIN - Low Confidence", "alert-warning"


# ---------------------------------------------------------------------------
# Support / resistance
# ---------------------------------------------------------------------------

def calculate_support_resistance(df, window=20):
    """
    Return (support, resistance) price levels computed over a rolling window.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Close' column.
    window : int
        Look-back window in days.
    """
    highs = df["Close"].rolling(window=window).max()
    lows = df["Close"].rolling(window=window).min()
    resistance = highs.iloc[-window:].max()
    support = lows.iloc[-window:].min()
    return float(support), float(resistance)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def calculate_metrics(actual, predicted):
    """
    Compute regression error metrics.

    Parameters
    ----------
    actual : array-like
        Ground-truth values.
    predicted : array-like
        Model predictions.

    Returns
    -------
    dict with keys: MSE, RMSE, MAE, MAPE
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape}


# ---------------------------------------------------------------------------
# Model confidence
# ---------------------------------------------------------------------------

def get_model_confidence(predictions_df):
    """
    Derive confidence scores (0-100) from the inverse of each model's
    prediction volatility.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Columns must end with '_Prediction' (e.g. 'LSTM_Prediction').

    Returns
    -------
    dict  {model_name: confidence_score}
    """
    volatility = predictions_df.std()
    max_vol = volatility.max()

    confidence = {}
    for col in predictions_df.columns:
        if "Prediction" in col:
            model_name = col.replace("_Prediction", "")
            if max_vol == 0:
                conf_score = 100.0
            else:
                conf_score = max(0.0, 100.0 - (volatility[col] / max_vol * 100))
            confidence[model_name] = min(100.0, conf_score)

    return confidence


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def add_technical_indicators(df):
    """
    Add SMA_20, SMA_50, BB_Upper, BB_Lower, and RSI columns to a copy of df.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Close' column.
    """
    df = df.copy()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    rolling_mean = df["Close"].rolling(window=20).mean()
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = rolling_mean + (rolling_std * 2)
    df["BB_Lower"] = rolling_mean - (rolling_std * 2)

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


# ---------------------------------------------------------------------------
# XGBoost feature engineering
# ---------------------------------------------------------------------------

def create_xgb_features(df, lags=None):
    """
    Create lag and moving-average features for XGBoost.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Close' column.
    lags : list[int]
        Lag periods to include (default [1, 2, 3, 5, 7, 10]).

    Returns
    -------
    pd.DataFrame  (rows with NaN dropped)
    """
    if lags is None:
        lags = [1, 2, 3, 5, 7, 10]
    feat = df.copy()
    for lag in lags:
        feat[f"lag_{lag}"] = feat["Close"].shift(lag)
    feat["ma_5"] = feat["Close"].rolling(5).mean()
    feat["ma_10"] = feat["Close"].rolling(10).mean()
    return feat.dropna()


def train_xgb(df):
    """
    Train an XGBRegressor on the Close price using lag + MA features.

    Returns
    -------
    (model, feature_columns)
    """
    import xgboost as xgb

    df_feat = create_xgb_features(df)
    X = df_feat.drop(columns=["Close"])
    y = df_feat["Close"]
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y)
    return model, X.columns


# ---------------------------------------------------------------------------
# LSTM preprocessing
# ---------------------------------------------------------------------------

def prepare_lstm_data(df, sequence_length=60):
    """
    Scale and window the Close price series for LSTM training.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Close' column.
    sequence_length : int
        Number of time-steps per input sequence.

    Returns
    -------
    (X, y, scaler)
        X shape: (n_samples, sequence_length, 1)
        y shape: (n_samples,)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[["Close"]].values)
    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i - sequence_length: i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    if len(X) == 0:
        X = X.reshape((0, sequence_length, 1))
    else:
        X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler


def build_lstm(input_shape):
    """
    Build and compile a two-layer stacked LSTM model.

    Parameters
    ----------
    input_shape : tuple
        (sequence_length, n_features)
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
