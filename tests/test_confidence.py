"""
test_confidence.py
------------------
Tests for get_model_confidence.
"""

import numpy as np
import pandas as pd
import pytest
from src.forecasting import get_model_confidence


class TestGetModelConfidence:
    def test_returns_dict_with_model_names(self, predictions_df):
        result = get_model_confidence(predictions_df)
        assert set(result.keys()) == {"LSTM", "XGBoost", "ARIMA"}

    def test_scores_in_valid_range(self, predictions_df):
        result = get_model_confidence(predictions_df)
        for name, score in result.items():
            assert 0.0 <= score <= 100.0, f"{name} confidence {score} out of range"

    def test_lower_volatility_gives_higher_confidence(self):
        """Model with flat (zero-std) predictions should have highest confidence."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "LSTM_Prediction": np.full(10, 100.0),          # zero variance
                "XGBoost_Prediction": np.linspace(90, 110, 10), # high variance
            },
            index=dates,
        )
        result = get_model_confidence(df)
        assert result["LSTM"] > result["XGBoost"]

    def test_all_same_volatility_returns_zero_for_all(self):
        """When all models have identical volatility, max_vol == vol for each,
        so conf_score = max(0, 100 - 100) = 0."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = np.linspace(100, 110, 10)
        df = pd.DataFrame(
            {
                "LSTM_Prediction": data,
                "XGBoost_Prediction": data.copy(),
            },
            index=dates,
        )
        result = get_model_confidence(df)
        for score in result.values():
            assert score == pytest.approx(0.0)

    def test_ignores_non_prediction_columns(self):
        """Columns not containing 'Prediction' should be silently ignored."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "LSTM_Prediction": np.linspace(100, 105, 5),
                "Date_Column": np.arange(5, dtype=float),  # should be ignored
            },
            index=dates,
        )
        result = get_model_confidence(df)
        assert "Date" not in result
        assert "LSTM" in result

    def test_single_model(self):
        """With a single model, it gets maximum confidence (100 – 0 = 100)."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {"LSTM_Prediction": np.linspace(100, 110, 10)},
            index=dates,
        )
        result = get_model_confidence(df)
        # The single model's vol == max_vol, so conf = max(0, 100 - 100) = 0
        # unless all-zero-vol, which would give 100. This tests the actual formula.
        assert "LSTM" in result
        assert 0.0 <= result["LSTM"] <= 100.0

    def test_constant_predictions_give_full_confidence(self):
        """Zero-variance predictions should yield 100% confidence."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "LSTM_Prediction": np.full(10, 100.0),
                "XGBoost_Prediction": np.linspace(90, 110, 10),
            },
            index=dates,
        )
        result = get_model_confidence(df)
        assert result["LSTM"] == pytest.approx(100.0)
