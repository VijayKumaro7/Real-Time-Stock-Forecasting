<div align="center">

<h1>📈 Real-Time Stock Market Forecasting</h1>

<img src="https://img.shields.io/github/stars/vijaykumaro7/Real-Time-Stock-Forecasting?style=social">
<img src="https://img.shields.io/github/forks/vijaykumaro7/Real-Time-Stock-Forecasting?style=social">
<img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">

<br><br>

<strong>Forecast stock market trends in real-time using LSTM, XGBoost, and ARIMA models with an interactive Streamlit dashboard.</strong>

</div>

---

## 🌟 Features

- ⚡ **Real-time data retrieval** via Yahoo Finance API
- 🤖 **LSTM Neural Networks** for deep learning-based price prediction
- 🌲 **XGBoost** for gradient boosting-based forecasting
- 📉 **ARIMA** for classical statistical time-series analysis
- 📊 **Technical Indicators**: SMA, Bollinger Bands, RSI
- 🎯 **Trading Signals**: AI-powered Buy / Sell / Hold recommendations
- 📡 **Live Dashboard** with auto-refresh capability
- 🔔 **Price Alert System** with configurable upper/lower thresholds
- 📈 **Model Performance Metrics**: RMSE, MAE, MAPE, confidence scores
- 🌐 **Supports Indian & US stocks** (e.g., `TATAMOTORS.NS`, `AAPL`)

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10+ |
| **Web Framework** | Streamlit |
| **Deep Learning** | TensorFlow / Keras (LSTM) |
| **Machine Learning** | XGBoost, Scikit-learn |
| **Time Series** | Statsmodels (ARIMA) |
| **Data Source** | yfinance (Yahoo Finance API) |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Data Processing** | Pandas, NumPy |

---

## 📁 Project Structure

```
Real-Time-Stock-Forecasting/
├── app_checkpoint.py              # Main application (LSTM + XGBoost + ARIMA)
├── app-checkpoint.ipynb           # Jupyter notebook version of main app
├── Streamlit code/
│   ├── app.py                     # Simplified LSTM-only Streamlit app
│   ├── app-checkpoint.py          # Intermediate app (LSTM + Prophet)
│   └── app.ipynb                  # Notebook version
├── src_model_code/
│   ├── Full_model_code.ipynb      # Full model development notebook
│   └── LSTM_Model/
│       └── lstm_pipeline.ipynb    # Dedicated LSTM pipeline notebook
├── README/
│   └── requirements.txt           # Core dependencies list
├── .devcontainer/
│   └── devcontainer.json          # Dev container configuration
├── .github/
│   └── workflows/
│       ├── python-publish.yml     # PyPI publish workflow
│       └── python-package-conda.yml # Conda test workflow
└── Requirements.txt
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/vijaykumaro7/Real-Time-Stock-Forecasting.git
cd Real-Time-Stock-Forecasting
```

### 2. Create and activate a virtual environment

```bash
python -m venv myenv

# Linux / macOS
source myenv/bin/activate

# Windows
myenv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit yfinance pandas numpy plotly scikit-learn tensorflow xgboost statsmodels matplotlib seaborn
```

---

## ▶️ Running the Application

### Full-featured app (LSTM + XGBoost + ARIMA)

```bash
streamlit run app_checkpoint.py
```

### Simplified app (LSTM only)

```bash
streamlit run "Streamlit code/app.py"
```

The app runs on `http://localhost:8501` by default.

---

## 💡 How It Works

```
User Input (Ticker, Date Range, Forecast Days)
           ↓
Yahoo Finance API — Live & Historical Data
           ↓
Data Preprocessing & MinMax Scaling
           ↓
  ┌────────┬──────────┬────────┐
  │  LSTM  │ XGBoost  │ ARIMA  │
  └────────┴──────────┴────────┘
           ↓
Technical Analysis (SMA, Bollinger Bands, RSI)
           ↓
Performance Metrics & Confidence Scoring
           ↓
Trading Signal Generation (Buy / Sell / Hold)
           ↓
Interactive Streamlit Dashboard
```

---

## 🤖 Models Overview

### LSTM (Long Short-Term Memory)
- Uses a 60-day sequence window for input
- Two LSTM layers with 50 units each + Dropout (0.2)
- MinMaxScaler normalization to [0, 1] range
- Autoregressive prediction for future days

### XGBoost
- Lag features: 1, 2, 3, 5, 7, 10 days
- Moving averages: 5-day and 10-day
- 200 estimators, learning rate 0.05
- Recursive multi-step forecasting

### ARIMA
- Configuration: order (5, 1, 0)
- Classical autoregressive integrated moving average
- Powered by the Statsmodels library

---

## 📊 Application Preview

![App Screenshot 1](https://github.com/user-attachments/assets/346312d7-51ed-453e-bfa8-ca7ebe3db443)
![App Screenshot 2](https://github.com/user-attachments/assets/ca951c4c-3ec0-48e8-bafb-3146c246cbb3)
![App Screenshot 3](https://github.com/user-attachments/assets/85273046-9fe1-4f27-8f22-3ee221b17c8f)
![App Screenshot 4](https://github.com/user-attachments/assets/266dcf69-1e0a-4881-b5a9-fbb166becec7)
![App Screenshot 5](https://github.com/user-attachments/assets/af20c645-5b7b-4029-aba1-54291214f42c)
![App Screenshot 6](https://github.com/user-attachments/assets/520a0c5b-341c-41da-a700-209f9f7c33fb)
![App Screenshot 7](https://github.com/user-attachments/assets/a15abaf3-c981-4cfa-8604-7b62f22245a9)

---

## 🗂️ Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **LSTM** | Deep learning price forecast with confidence band |
| **XGBoost** | Gradient boosting forecast with feature importance |
| **ARIMA** | Statistical time-series forecast |
| **Technical Analysis** | SMA, Bollinger Bands, RSI charts |
| **Live Dashboard** | Real-time price with auto-refresh |

---

## 📦 Supported Stock Symbols

| Market | Example Symbols |
|--------|----------------|
| **US Stocks** | `AAPL`, `GOOGL`, `TSLA`, `MSFT`, `AMZN` |
| **Indian Stocks (NSE)** | `TATAMOTORS.NS`, `RELIANCE.NS`, `INFY.NS` |
| **Indian Stocks (BSE)** | `TATAMOTORS.BO`, `RELIANCE.BO` |

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

<div align="center">
  Made with ❤️ using Python & Streamlit
</div>
