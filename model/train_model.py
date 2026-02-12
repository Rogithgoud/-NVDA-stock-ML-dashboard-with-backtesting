import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from utils.data_loader import fetch_data

df = fetch_data()

df["return"] = df["close"].pct_change()
df["volatility"] = df["return"].rolling(5).std()
df["ma_5"] = df["close"].rolling(5).mean()
df["ma_10"] = df["close"].rolling(10).mean()
df["ma_20"] = df["close"].rolling(20).mean()

df["target"] = df["close"].shift(-1)

df = df.dropna()

features = ["close", "volume", "volatility", "ma_5", "ma_10", "ma_20"]

X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f"Model Trained. MAE: ${mae:.2f}")

joblib.dump(model, "model/model.pkl")
