"""
test_models.py
--------------
Smoke tests for model architecture (build_lstm) and training (train_xgb).
These tests verify that models can be constructed and trained without errors,
and that their outputs have the expected shapes and types.

Tests in this module are automatically skipped if TensorFlow or XGBoost are
not installed in the current environment (pytest.importorskip).
"""

import numpy as np
import pytest

from src.forecasting import create_xgb_features


# ---------------------------------------------------------------------------
# build_lstm  (requires TensorFlow)
# ---------------------------------------------------------------------------

class TestBuildLstm:
    @pytest.fixture(autouse=True)
    def require_tensorflow(self):
        pytest.importorskip("tensorflow", reason="TensorFlow not installed")

    def test_returns_sequential_model(self):
        from tensorflow.keras.models import Sequential
        from src.forecasting import build_lstm
        model = build_lstm(input_shape=(60, 1))
        assert isinstance(model, Sequential)

    def test_model_has_five_layers(self):
        from src.forecasting import build_lstm
        model = build_lstm(input_shape=(60, 1))
        # LSTM → Dropout → LSTM → Dropout → Dense
        assert len(model.layers) == 5

    def test_output_shape_is_one(self):
        from src.forecasting import build_lstm
        model = build_lstm(input_shape=(60, 1))
        assert model.output_shape == (None, 1)

    def test_model_is_compiled(self):
        from src.forecasting import build_lstm
        model = build_lstm(input_shape=(60, 1))
        assert model.optimizer is not None

    def test_model_accepts_correct_input_shape(self):
        from src.forecasting import build_lstm
        seq_len = 30
        model = build_lstm(input_shape=(seq_len, 1))
        dummy = np.zeros((4, seq_len, 1))
        output = model.predict(dummy, verbose=0)
        assert output.shape == (4, 1)

    def test_custom_input_shape(self):
        from src.forecasting import build_lstm
        model = build_lstm(input_shape=(10, 1))
        assert model.input_shape == (None, 10, 1)


# ---------------------------------------------------------------------------
# train_xgb  (requires XGBoost)
# ---------------------------------------------------------------------------

class TestTrainXgb:
    @pytest.fixture(autouse=True)
    def require_xgboost(self):
        pytest.importorskip("xgboost", reason="XGBoost not installed")

    def test_returns_model_and_feature_columns(self, price_series):
        from src.forecasting import train_xgb
        model, feat_cols = train_xgb(price_series)
        assert model is not None
        assert len(feat_cols) > 0

    def test_feature_columns_match_expected(self, price_series):
        from src.forecasting import train_xgb
        _, feat_cols = train_xgb(price_series)
        expected_lags = {f"lag_{l}" for l in [1, 2, 3, 5, 7, 10]}
        expected_mas = {"ma_5", "ma_10"}
        assert expected_lags.issubset(set(feat_cols))
        assert expected_mas.issubset(set(feat_cols))

    def test_close_not_in_feature_columns(self, price_series):
        from src.forecasting import train_xgb
        _, feat_cols = train_xgb(price_series)
        assert "Close" not in feat_cols

    def test_predictions_are_numeric(self, price_series):
        from src.forecasting import train_xgb
        model, feat_cols = train_xgb(price_series)
        feat_df = create_xgb_features(price_series)
        X = feat_df[feat_cols]
        preds = model.predict(X)
        assert np.issubdtype(preds.dtype, np.floating) or np.issubdtype(preds.dtype, np.integer)

    def test_predictions_shape_matches_input(self, price_series):
        from src.forecasting import train_xgb
        model, feat_cols = train_xgb(price_series)
        feat_df = create_xgb_features(price_series)
        X = feat_df[feat_cols]
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_predictions_in_reasonable_range(self, price_series):
        """Predictions should be in the same order of magnitude as Close prices."""
        from src.forecasting import train_xgb
        model, feat_cols = train_xgb(price_series)
        feat_df = create_xgb_features(price_series)
        X = feat_df[feat_cols]
        preds = model.predict(X)
        close_min = price_series["Close"].min()
        close_max = price_series["Close"].max()
        margin = (close_max - close_min) * 0.5
        assert preds.min() > close_min - margin
        assert preds.max() < close_max + margin
