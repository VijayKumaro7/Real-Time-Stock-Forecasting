# CLAUDE.md — AI Assistant Guide for Real-Time Stock Forecasting

## Project Overview

A real-time stock price forecasting dashboard built with Streamlit. It combines three forecasting models (LSTM, XGBoost, ARIMA) with technical indicators and trading signal generation. Supports US and Indian (NSE/BSE) stock symbols via the Yahoo Finance API.

---

## Repository Structure

```
Real-Time-Stock-Forecasting/
├── src/
│   ├── __init__.py
│   └── forecasting.py          # Pure business logic (no Streamlit deps) — testable
├── tests/
│   ├── conftest.py             # Shared pytest fixtures
│   ├── test_confidence.py      # Confidence scoring tests
│   ├── test_data_loading.py    # yfinance mocked data-loading tests
│   ├── test_features.py        # XGBoost feature engineering tests
│   ├── test_metrics.py         # Model performance metrics tests
│   ├── test_models.py          # LSTM & XGBoost architecture tests
│   ├── test_preprocessing.py   # LSTM windowing/scaling tests
│   └── test_signals.py         # Trading signal generation tests
├── src_model_code/             # Research/development notebooks (not production)
│   ├── Full_model_code.ipynb
│   └── LSTM_Model/lstm_pipeline.ipynb
├── Streamlit code/
│   ├── app.py                  # Simplified LSTM-only Streamlit app
│   └── app.ipynb
├── .github/workflows/
│   ├── python-package-conda.yml  # CI: lint + test on push
│   └── python-publish.yml        # CD: publish to PyPI on release
├── .devcontainer/devcontainer.json
├── app_checkpoint.py           # Main full-featured Streamlit app (966 lines)
├── Requirements.txt            # Streamlit app dependencies
├── requirements-dev.txt        # Dev/test dependencies
├── README/requirements.txt     # Core library dependencies
├── pytest.ini
└── README.md
```

**Key distinction:** `src/forecasting.py` contains pure functions with no Streamlit imports. All business logic that needs to be tested lives here. The Streamlit apps (`app_checkpoint.py`, `Streamlit code/app.py`) are the UI layer.

---

## Development Branch

Always develop on `claude/add-claude-documentation-8qlxH`. Never push to `main` directly.

```bash
git checkout claude/add-claude-documentation-8qlxH
git push -u origin claude/add-claude-documentation-8qlxH
```

---

## Running the Application

```bash
# Full-featured app (LSTM + XGBoost + ARIMA + technical indicators + alerts)
streamlit run app_checkpoint.py

# Simplified LSTM-only app
streamlit run "Streamlit code/app.py"
```

The dev container auto-launches the simplified app on port `8501`.

---

## Installing Dependencies

```bash
python -m venv myenv
source myenv/bin/activate            # Linux/macOS
pip install -r requirements-dev.txt  # Installs test deps (pytest, pytest-cov, etc.)
pip install -r README/requirements.txt  # Installs core app deps
```

---

## Testing

```bash
pytest                    # Run all tests (configured via pytest.ini)
pytest -v                 # Verbose output
pytest --cov=src          # With coverage report
pytest tests/test_signals.py  # Run a single test module
```

**pytest.ini config:**
```ini
[pytest]
testpaths = tests
addopts = -v --tb=short
```

**Test infrastructure:**
- 80+ tests across 8 modules
- `conftest.py` provides shared fixtures: `price_series` (200-day synthetic), `price_df_short` (100-row), `predictions_df` (multi-model DataFrame)
- `test_data_loading.py` mocks `yfinance.download` — no network calls in tests
- All tests are pure unit tests against `src/forecasting.py`

**When adding new business logic:** Add it to `src/forecasting.py` and write corresponding tests in `tests/`. Keep UI logic in the Streamlit app files.

---

## Linting

CI runs flake8 with these rules:

```bash
# Hard failures (stops CI)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Style warnings (non-blocking, max-line-length=127)
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

Max line length is **127 characters**. Max cyclomatic complexity per function is **10**.

---

## Architecture & Key Conventions

### Separation of Concerns

| Layer | File | Rule |
|---|---|---|
| Business logic | `src/forecasting.py` | No Streamlit imports. Pure functions only. |
| UI / dashboard | `app_checkpoint.py` | Imports from `src/forecasting.py`. |
| Tests | `tests/` | Import from `src/` only. Mock external calls. |

### Model Configurations

**LSTM:**
- Sequence window: 60 days
- Architecture: 2 stacked LSTM layers (50 units each) + Dropout(0.2)
- Optimizer: Adam, Loss: MSE
- Training: 5 epochs, batch_size=32
- Scaling: MinMaxScaler to [0, 1]

**XGBoost:**
- Lag features: [1, 2, 3, 5, 7, 10] days
- Moving averages: 5-day, 10-day
- Params: 200 estimators, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8
- Forecasting strategy: recursive multi-step

**ARIMA:**
- Order: (5, 1, 0)

### Technical Indicators

Computed and stored as DataFrame columns: `SMA_20`, `SMA_50`, Bollinger Bands (20-period, ±2σ), RSI (14-period), support/resistance (20-day rolling min/max).

### Trading Signals

Generated by `generate_trading_signal(consensus_trend, confidence_avg, volatility)` in `src/forecasting.py`:

| Signal | Condition |
|---|---|
| STRONG BUY | confidence >= 70% AND consensus_trend > 2% |
| BUY | confidence >= 70% AND consensus_trend > 0.5% |
| HOLD | confidence >= 70% AND -0.5% <= trend <= 0.5% |
| SELL | confidence >= 70% AND consensus_trend < -0.5% |
| STRONG SELL | confidence >= 70% AND consensus_trend < -2% |
| UNCERTAIN | confidence < 70% |

### Confidence Scoring

Formula: `max(0, 100 - (model_volatility / max_volatility * 100))` — normalized to [0, 100]. Lower volatility in predictions → higher confidence.

### Data Loading

- Source: `yfinance.download()` — no persistent database
- Caching: `@st.cache_data` in Streamlit apps for historical data
- Supported symbols: US stocks (AAPL, GOOGL, TSLA, MSFT, AMZN), Indian NSE (suffix `.NS`), Indian BSE (suffix `.BO`)

---

## Naming Conventions

- Functions: `snake_case` (e.g., `prepare_lstm_data`, `calculate_metrics`)
- Test classes: `PascalCase` (e.g., `TestBuildLstm`, `TestGenerateTradingSignal`)
- DataFrame columns: uppercase with underscores (e.g., `SMA_20`, `LSTM_Prediction`)
- All public functions in `src/forecasting.py` must have Google-style docstrings

---

## CI/CD Pipelines

| Workflow | Trigger | Actions |
|---|---|---|
| `python-package-conda.yml` | push to any branch | Python 3.10, flake8 lint, pytest |
| `python-publish.yml` | release published | Build package, publish to PyPI (OIDC) |

---

## What NOT to Do

- Do not import `streamlit` in `src/forecasting.py`
- Do not make real network calls in tests — mock `yfinance.download`
- Do not push directly to `main`
- Do not add features beyond what was asked — keep changes minimal
- Do not change LSTM hyperparameters (epochs, layers, window) without documenting the reason
- Do not exceed 127-character line length
- Do not exceed cyclomatic complexity of 10 per function
