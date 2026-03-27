"""
test_signals.py
---------------
Tests for get_trend_indicator and generate_trading_signal.
"""

import pytest
from src.forecasting import get_trend_indicator, generate_trading_signal


# ---------------------------------------------------------------------------
# get_trend_indicator
# ---------------------------------------------------------------------------

class TestGetTrendIndicator:
    def test_price_increase_returns_bullish(self):
        arrow, css_class, pct = get_trend_indicator(110, 100)
        assert css_class == "bullish"
        assert pct == pytest.approx(10.0)
        assert arrow == "up"

    def test_price_decrease_returns_bearish(self):
        arrow, css_class, pct = get_trend_indicator(90, 100)
        assert css_class == "bearish"
        assert pct == pytest.approx(-10.0)
        assert arrow == "down"

    def test_price_unchanged_returns_neutral(self):
        arrow, css_class, pct = get_trend_indicator(100, 100)
        assert css_class == "neutral"
        assert pct == pytest.approx(0.0)
        assert arrow == "flat"

    def test_small_increase(self):
        _, css_class, pct = get_trend_indicator(100.01, 100.0)
        assert css_class == "bullish"
        assert pct > 0

    def test_pct_change_formula(self):
        _, _, pct = get_trend_indicator(150, 100)
        assert pct == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# generate_trading_signal
# ---------------------------------------------------------------------------

class TestGenerateTradingSignal:
    """All six code paths inside generate_trading_signal."""

    def test_strong_buy(self):
        signal, css = generate_trading_signal(consensus_trend=5.0, confidence_avg=80, volatility=1.0)
        assert signal == "STRONG BUY"
        assert css == "alert-success"

    def test_buy(self):
        signal, css = generate_trading_signal(consensus_trend=1.0, confidence_avg=75, volatility=1.0)
        assert signal == "BUY"
        assert css == "alert-success"

    def test_strong_sell(self):
        signal, css = generate_trading_signal(consensus_trend=-3.0, confidence_avg=90, volatility=1.0)
        assert signal == "STRONG SELL"
        assert css == "alert-danger"

    def test_sell(self):
        signal, css = generate_trading_signal(consensus_trend=-1.0, confidence_avg=72, volatility=1.0)
        assert signal == "SELL"
        assert css == "alert-danger"

    def test_hold(self):
        signal, css = generate_trading_signal(consensus_trend=0.1, confidence_avg=85, volatility=0.5)
        assert signal == "HOLD"
        assert css == "alert-warning"

    def test_uncertain_low_confidence(self):
        signal, css = generate_trading_signal(consensus_trend=5.0, confidence_avg=50, volatility=1.0)
        assert signal == "UNCERTAIN - Low Confidence"
        assert css == "alert-warning"

    def test_boundary_confidence_exactly_70(self):
        """confidence_avg == 70 should use the high-confidence branch."""
        signal, _ = generate_trading_signal(consensus_trend=3.0, confidence_avg=70, volatility=1.0)
        assert signal == "STRONG BUY"

    def test_boundary_confidence_just_below_70(self):
        """confidence_avg < 70 must return UNCERTAIN regardless of trend."""
        signal, _ = generate_trading_signal(consensus_trend=10.0, confidence_avg=69.9, volatility=1.0)
        assert signal == "UNCERTAIN - Low Confidence"

    def test_boundary_trend_exactly_2(self):
        """consensus_trend == 2 does NOT qualify for STRONG BUY (requires > 2)."""
        signal, _ = generate_trading_signal(consensus_trend=2.0, confidence_avg=80, volatility=1.0)
        assert signal == "BUY"

    def test_boundary_trend_exactly_negative_2(self):
        """consensus_trend == -2 does NOT qualify for STRONG SELL (requires < -2)."""
        signal, _ = generate_trading_signal(consensus_trend=-2.0, confidence_avg=80, volatility=1.0)
        assert signal == "SELL"

    def test_volatility_does_not_affect_signal(self):
        """Volatility parameter is not used in the signal logic."""
        s1, _ = generate_trading_signal(3.0, 80, volatility=0)
        s2, _ = generate_trading_signal(3.0, 80, volatility=999)
        assert s1 == s2
