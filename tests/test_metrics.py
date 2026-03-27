"""
test_metrics.py
---------------
Tests for calculate_metrics.
"""

import numpy as np
import pytest
from src.forecasting import calculate_metrics


class TestCalculateMetrics:
    def test_perfect_predictions_give_zero_errors(self):
        actual = np.array([100.0, 200.0, 300.0])
        predicted = actual.copy()
        m = calculate_metrics(actual, predicted)
        assert m["MSE"] == pytest.approx(0.0)
        assert m["RMSE"] == pytest.approx(0.0)
        assert m["MAE"] == pytest.approx(0.0)
        assert m["MAPE"] == pytest.approx(0.0)

    def test_rmse_is_sqrt_of_mse(self):
        actual = np.array([10.0, 20.0, 30.0])
        predicted = np.array([11.0, 21.0, 31.0])
        m = calculate_metrics(actual, predicted)
        assert m["RMSE"] == pytest.approx(np.sqrt(m["MSE"]))

    def test_known_values(self):
        # actual=[100, 200], predicted=[110, 180]
        # errors: 10, -20  → MAE=15, MSE=250, RMSE=sqrt(250)≈15.81
        # MAPE: |10/100|*100=10%, |-20/200|*100=10% → mean=10%
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 180.0])
        m = calculate_metrics(actual, predicted)
        assert m["MAE"] == pytest.approx(15.0)
        assert m["MSE"] == pytest.approx(250.0)
        assert m["RMSE"] == pytest.approx(np.sqrt(250.0))
        assert m["MAPE"] == pytest.approx(10.0)

    def test_mse_is_non_negative(self):
        actual = np.array([5.0, 10.0, 15.0])
        predicted = np.array([6.0, 8.0, 14.0])
        m = calculate_metrics(actual, predicted)
        assert m["MSE"] >= 0

    def test_mae_is_non_negative(self):
        actual = np.array([5.0, 10.0])
        predicted = np.array([3.0, 12.0])
        m = calculate_metrics(actual, predicted)
        assert m["MAE"] >= 0

    def test_returns_all_four_keys(self):
        m = calculate_metrics([1.0], [1.0])
        assert set(m.keys()) == {"MSE", "RMSE", "MAE", "MAPE"}

    def test_accepts_lists(self):
        """Should work with plain Python lists, not just numpy arrays."""
        m = calculate_metrics([100.0, 200.0], [100.0, 200.0])
        assert m["RMSE"] == pytest.approx(0.0)

    def test_symmetric_errors_mae(self):
        """MAE should be symmetric: over- and under-predict by same amount."""
        actual = np.array([100.0, 100.0])
        predicted = np.array([110.0, 90.0])
        m = calculate_metrics(actual, predicted)
        assert m["MAE"] == pytest.approx(10.0)
