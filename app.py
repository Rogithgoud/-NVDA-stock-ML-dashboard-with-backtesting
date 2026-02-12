import streamlit as st
import joblib
import pandas as pd
import numpy as np
from utils.data_loader import fetch_data

st.set_page_config(page_title="NVDA Stock Prediction Dashboard", layout="wide")
st.title("📈 NVIDIA Stock Price Prediction Dashboard")

# ==============================
# Load Model
# ==============================

model = joblib.load("model/model.pkl")

# ==============================
# Fetch Data
# ==============================

df = fetch_data("NVDA")
df = df.sort_values("timestamp")

# ==============================
# Feature Engineering
# ==============================

df["return"] = df["close"].pct_change()
df["volatility"] = df["return"].rolling(5).std()
df["ma_5"] = df["close"].rolling(5).mean()
df["ma_10"] = df["close"].rolling(10).mean()
df["ma_20"] = df["close"].rolling(20).mean()

df = df.dropna()

features = ["close", "volume", "volatility", "ma_5", "ma_10", "ma_20"]
X = df[features]

# ==============================
# Next Day Prediction (Top Metric)
# ==============================

latest_data = X.iloc[[-1]]
predicted_next_price = model.predict(latest_data)[0]

st.metric("Predicted Next Day Close", f"${predicted_next_price:.2f}")

# ==============================
# Full Dataset Predictions
# ==============================

df["Predicted Price"] = model.predict(X)
df["Actual Price"] = df["close"]

# ==============================
# Buy/Sell Signal
# ==============================

df["Signal"] = np.where(
    df["Predicted Price"] > df["Actual Price"], 1, 0
)

# ==============================
# Backtesting
# ==============================

df["Strategy Return"] = df["Signal"].shift(1) * df["return"]
df["Cumulative Strategy"] = (1 + df["Strategy Return"]).cumprod()

df["Buy Hold"] = (1 + df["return"]).cumprod()

# ==============================
# Charts
# ==============================

st.subheader("Actual vs Predicted Price")

chart_df = df[["Actual Price", "Predicted Price"]]
st.line_chart(chart_df)

st.subheader("Backtesting: Strategy vs Buy & Hold")

backtest_df = df[["Cumulative Strategy", "Buy Hold"]]
st.line_chart(backtest_df)

# ==============================
# Performance Metrics
# ==============================

strategy_return = df["Cumulative Strategy"].iloc[-1] - 1
buy_hold_return = df["Buy Hold"].iloc[-1] - 1

col1, col2 = st.columns(2)

col1.metric("Strategy Total Return", f"{strategy_return*100:.2f}%")
col2.metric("Buy & Hold Return", f"{buy_hold_return*100:.2f}%")
