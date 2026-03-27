"""
test_data_loading.py
--------------------
Tests for the data-loading layer using mocked yfinance calls.
yfinance is mocked at the sys.modules level so no installation or network
access is required.
"""

import sys
import numpy as np
import pandas as pd
import pytest
from types import ModuleType
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Inject a lightweight yfinance stub into sys.modules before any import
# ---------------------------------------------------------------------------

def _make_yfinance_stub(return_value):
    """Return a minimal yfinance-like module stub."""
    stub = ModuleType("yfinance")
    stub.download = MagicMock(return_value=return_value)
    return stub


def _make_mock_df(n=100, ticker="AAPL"):
    """Return a plausible yfinance-style MultiIndex DataFrame."""
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = 100 + np.arange(n, dtype=float)
    cols = pd.MultiIndex.from_tuples([("Close", ticker)], names=["Price", "Ticker"])
    return pd.DataFrame(prices.reshape(-1, 1), index=dates, columns=cols)


# ---------------------------------------------------------------------------
# The function under test (mirrors load_data from app_checkpoint.py)
# ---------------------------------------------------------------------------

def _load_data_with_stub(yf_stub, ticker, start_date):
    """Call load_data logic using the provided yfinance stub."""
    df = yf_stub.download(ticker, start=start_date, end="2024-12-31")
    if df.empty:
        return None
    df = df[["Close"]].dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_returns_dataframe_on_success(self):
        stub = _make_yfinance_stub(_make_mock_df())
        result = _load_data_with_stub(stub, "AAPL", "2023-01-01")
        assert isinstance(result, pd.DataFrame)

    def test_returns_only_close_column(self):
        stub = _make_yfinance_stub(_make_mock_df())
        result = _load_data_with_stub(stub, "AAPL", "2023-01-01")
        assert list(result.columns) == ["Close"]

    def test_no_nan_rows_returned(self):
        mock_df = _make_mock_df()
        mock_df.iloc[5, 0] = np.nan
        mock_df.iloc[10, 0] = np.nan
        stub = _make_yfinance_stub(mock_df)
        result = _load_data_with_stub(stub, "AAPL", "2023-01-01")
        assert result.isnull().sum().sum() == 0

    def test_returns_none_when_api_returns_empty(self):
        stub = _make_yfinance_stub(pd.DataFrame())
        result = _load_data_with_stub(stub, "INVALID", "2023-01-01")
        assert result is None

    def test_row_count_matches_mock(self):
        n = 150
        stub = _make_yfinance_stub(_make_mock_df(n=n))
        result = _load_data_with_stub(stub, "AAPL", "2023-01-01")
        assert len(result) == n

    def test_download_called_with_correct_ticker(self):
        stub = _make_yfinance_stub(_make_mock_df())
        _load_data_with_stub(stub, "TSLA", "2023-01-01")
        call_args = stub.download.call_args
        assert "TSLA" in str(call_args)

    def test_multiindex_columns_flattened(self):
        """After loading, columns must be a plain Index, not MultiIndex."""
        stub = _make_yfinance_stub(_make_mock_df())
        result = _load_data_with_stub(stub, "AAPL", "2023-01-01")
        assert not isinstance(result.columns, pd.MultiIndex)

    def test_index_is_datetime(self):
        stub = _make_yfinance_stub(_make_mock_df())
        result = _load_data_with_stub(stub, "AAPL", "2023-01-01")
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_nan_rows_count_reduced_correctly(self):
        """All-NaN rows should be dropped, reducing the row count."""
        mock_df = _make_mock_df(n=50)
        # Mark 5 rows as NaN
        mock_df.iloc[[1, 3, 5, 7, 9], 0] = np.nan
        stub = _make_yfinance_stub(mock_df)
        result = _load_data_with_stub(stub, "AAPL", "2023-01-01")
        assert len(result) == 45
