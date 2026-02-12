import os
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID") or st.secrets["APCA_API_KEY_ID"]
API_SECRET = os.getenv("APCA_API_SECRET_KEY") or st.secrets["APCA_API_SECRET_KEY"]



def fetch_data(symbol="NVDA"):
    client = StockHistoricalDataClient(API_KEY, API_SECRET)

    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start="2022-01-01",
    )

    bars = client.get_stock_bars(request_params)
    df = bars.df.reset_index()
    df = df[df["symbol"] == symbol]
    df = df.sort_values("timestamp")

    return df
