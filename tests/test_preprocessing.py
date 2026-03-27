"""
test_preprocessing.py
---------------------
Tests for prepare_lstm_data (scaling + windowing).
"""

import numpy as np
import pytest
from src.forecasting import prepare_lstm_data


class TestPrepareLstmData:
    SEQ_LEN = 60

    def test_output_shapes(self, price_series):
        X, y, _ = prepare_lstm_data(price_series, sequence_length=self.SEQ_LEN)
        n = len(price_series) - self.SEQ_LEN
        assert X.shape == (n, self.SEQ_LEN, 1)
        assert y.shape == (n,)

    def test_x_values_in_unit_range(self, price_series):
        X, _, _ = prepare_lstm_data(price_series, sequence_length=self.SEQ_LEN)
        assert X.min() >= 0.0 - 1e-9
        assert X.max() <= 1.0 + 1e-9

    def test_y_values_in_unit_range(self, price_series):
        _, y, _ = prepare_lstm_data(price_series, sequence_length=self.SEQ_LEN)
        assert y.min() >= 0.0 - 1e-9
        assert y.max() <= 1.0 + 1e-9

    def test_scaler_inverse_transform_recovers_prices(self, price_series):
        X, y, scaler = prepare_lstm_data(price_series, sequence_length=self.SEQ_LEN)
        recovered = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        original = price_series["Close"].values[self.SEQ_LEN:]
        np.testing.assert_allclose(recovered, original, rtol=1e-5)

    def test_shorter_sequence_length(self, price_series):
        seq_len = 10
        X, y, _ = prepare_lstm_data(price_series, sequence_length=seq_len)
        n = len(price_series) - seq_len
        assert X.shape == (n, seq_len, 1)
        assert y.shape == (n,)

    def test_x_last_dim_is_one(self, price_series):
        X, _, _ = prepare_lstm_data(price_series, sequence_length=self.SEQ_LEN)
        assert X.shape[2] == 1

    def test_consecutive_windows_overlap_correctly(self, price_series):
        """X[1] should be X[0] shifted by one step."""
        X, _, _ = prepare_lstm_data(price_series, sequence_length=self.SEQ_LEN)
        np.testing.assert_array_equal(X[0, 1:, 0], X[1, :-1, 0])

    def test_raises_if_too_short(self):
        """DataFrame shorter than sequence_length should yield empty arrays."""
        import pandas as pd
        short_df = pd.DataFrame({"Close": [100.0] * 5})
        X, y, _ = prepare_lstm_data(short_df, sequence_length=60)
        assert len(X) == 0
        assert len(y) == 0
