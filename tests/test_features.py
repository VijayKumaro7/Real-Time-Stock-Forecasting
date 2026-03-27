"""
test_features.py
----------------
Tests for create_xgb_features, add_technical_indicators,
and calculate_support_resistance.
"""

import numpy as np
import pandas as pd
import pytest
from src.forecasting import (
    create_xgb_features,
    add_technical_indicators,
    calculate_support_resistance,
)


# ---------------------------------------------------------------------------
# create_xgb_features
# ---------------------------------------------------------------------------

class TestCreateXgbFeatures:
    def test_lag_columns_created(self, price_series):
        lags = [1, 2, 3]
        result = create_xgb_features(price_series, lags=lags)
        for lag in lags:
            assert f"lag_{lag}" in result.columns

    def test_ma_columns_created(self, price_series):
        result = create_xgb_features(price_series)
        assert "ma_5" in result.columns
        assert "ma_10" in result.columns

    def test_no_nan_after_dropna(self, price_series):
        result = create_xgb_features(price_series)
        assert result.isnull().sum().sum() == 0

    def test_row_count_after_dropna(self, price_series):
        # With default lags [1,2,3,5,7,10] and ma_10, the minimum lag needed
        # to avoid NaN is max(10, 10) = 10 rows dropped.
        result = create_xgb_features(price_series)
        assert len(result) == len(price_series) - 10

    def test_close_column_preserved(self, price_series):
        result = create_xgb_features(price_series)
        assert "Close" in result.columns

    def test_lag_1_values_match_shifted(self, price_series):
        result = create_xgb_features(price_series, lags=[1])
        # lag_1 at index i should equal Close at index i-1 in the original
        original = price_series["Close"]
        for idx in result.index:
            pos = price_series.index.get_loc(idx)
            expected = original.iloc[pos - 1]
            assert result.loc[idx, "lag_1"] == pytest.approx(expected)

    def test_ma_5_values_correct(self, price_series):
        result = create_xgb_features(price_series, lags=[1])
        expected_ma5 = price_series["Close"].rolling(5).mean()
        for idx in result.index:
            assert result.loc[idx, "ma_5"] == pytest.approx(expected_ma5.loc[idx])

    def test_custom_lags(self, price_series):
        result = create_xgb_features(price_series, lags=[2, 4])
        assert "lag_2" in result.columns
        assert "lag_4" in result.columns
        assert "lag_1" not in result.columns


# ---------------------------------------------------------------------------
# add_technical_indicators
# ---------------------------------------------------------------------------

class TestAddTechnicalIndicators:
    def test_returns_new_dataframe(self, price_series):
        result = add_technical_indicators(price_series)
        assert result is not price_series  # must be a copy

    def test_expected_columns_added(self, price_series):
        result = add_technical_indicators(price_series)
        for col in ["SMA_20", "SMA_50", "BB_Upper", "BB_Lower", "RSI"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_sma_20_matches_rolling_mean(self, price_series):
        result = add_technical_indicators(price_series)
        expected = price_series["Close"].rolling(20).mean()
        # Compare where both are not NaN
        mask = result["SMA_20"].notna()
        pd.testing.assert_series_equal(
            result.loc[mask, "SMA_20"],
            expected.loc[mask],
            check_names=False,
        )

    def test_bb_upper_above_bb_lower(self, price_series):
        result = add_technical_indicators(price_series)
        valid = result["BB_Upper"].notna() & result["BB_Lower"].notna()
        assert (result.loc[valid, "BB_Upper"] >= result.loc[valid, "BB_Lower"]).all()

    def test_rsi_in_valid_range(self, price_series):
        result = add_technical_indicators(price_series)
        rsi_valid = result["RSI"].dropna()
        assert (rsi_valid >= 0).all(), "RSI has values below 0"
        assert (rsi_valid <= 100).all(), "RSI has values above 100"

    def test_bollinger_band_width(self, price_series):
        """BB_Upper - BB_Lower should equal 4 * rolling std."""
        result = add_technical_indicators(price_series)
        rolling_std = price_series["Close"].rolling(20).std()
        expected_width = rolling_std * 4
        actual_width = result["BB_Upper"] - result["BB_Lower"]
        mask = expected_width.notna()
        pd.testing.assert_series_equal(
            actual_width.loc[mask],
            expected_width.loc[mask],
            check_names=False,
            rtol=1e-5,
        )

    def test_original_close_unchanged(self, price_series):
        result = add_technical_indicators(price_series)
        pd.testing.assert_series_equal(result["Close"], price_series["Close"])


# ---------------------------------------------------------------------------
# calculate_support_resistance
# ---------------------------------------------------------------------------

class TestCalculateSupportResistance:
    def test_support_lte_min_in_window(self, price_series):
        support, _ = calculate_support_resistance(price_series, window=20)
        recent_min = price_series["Close"].iloc[-20:].min()
        assert support <= recent_min + 1e-9  # rolling min over any window ≤ slice min

    def test_resistance_gte_max_in_window(self, price_series):
        _, resistance = calculate_support_resistance(price_series, window=20)
        recent_max = price_series["Close"].iloc[-20:].max()
        assert resistance >= recent_max - 1e-9

    def test_support_less_than_or_equal_to_resistance(self, price_series):
        support, resistance = calculate_support_resistance(price_series)
        assert support <= resistance

    def test_returns_floats(self, price_series):
        support, resistance = calculate_support_resistance(price_series)
        assert isinstance(support, float)
        assert isinstance(resistance, float)

    def test_constant_price_series(self):
        dates = pd.date_range("2023-01-01", periods=50)
        df = pd.DataFrame({"Close": np.full(50, 100.0)}, index=dates)
        support, resistance = calculate_support_resistance(df, window=20)
        assert support == pytest.approx(100.0)
        assert resistance == pytest.approx(100.0)
