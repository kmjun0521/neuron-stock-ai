
import os
import numpy as np
import yfinance as yf

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers

app = FastAPI(title="Neuron Stock AI")

# 우선 웹사이트 연결을 허용합니다.
# 배포 후 GitHub Pages 주소로 제한할 수 있습니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Neuron Stock AI is running"}

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/predict")
def predict(ticker: str = "005930.KS"):
    try:
        # 최근 1년의 주가와 거래량을 가져옵니다.
        df = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            return {"error": "주가 데이터를 찾을 수 없습니다."}

        # yfinance가 다중 열 이름을 반환하는 경우도 처리합니다.
        close = df["Close"]
        volume = df["Volume"]

        if getattr(close, "ndim", 1) > 1:
            close = close.iloc[:, 0]
        if getattr(volume, "ndim", 1) > 1:
            volume = volume.iloc[:, 0]

        data = np.column_stack([
            close.to_numpy(dtype=float),
            volume.to_numpy(dtype=float)
        ])

        data = data[np.isfinite(data).all(axis=1)]

        window = 10
        if len(data) < 60:
            return {"error": "학습에 필요한 주가 데이터가 부족합니다."}

        # 최근 10거래일의 종가와 거래량으로 다음 종가를 학습합니다.
        split = int(len(data) * 0.8)

        scaler = MinMaxScaler()
        scaler.fit(data[:split])

        scaled = scaler.transform(data)

        X, y = [], []
        for i in range(window, len(scaled)):
            X.append(scaled[i-window:i])
            y.append(scaled[i, 0])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        split_samples = split - window
        X_train, y_train = X[:split_samples], y[:split_samples]
        X_test, y_test = X[split_samples:], y[split_samples:]

        if len(X_test) == 0:
            return {"error": "평가용 데이터가 부족합니다."}

        # 인공신경망 모델을 구성합니다.
        keras.utils.set_random_seed(42)

        model = keras.Sequential([
            layers.Input(shape=(window, 2)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1)
        ])

        model.compile(optimizer="adam", loss="mse")

        model.fit(
            X_train,
            y_train,
            epochs=30,
            batch_size=16,
            verbose=0,
            shuffle=False
        )

        # 테스트 데이터에서 예측 성능을 평가합니다.
        test_pred_scaled = model.predict(X_test, verbose=0).reshape(-1)

        test_pred = (
            test_pred_scaled * scaler.scale_[0]
            + scaler.min_[0] * 0
        )

        # MinMaxScaler의 역변환을 정확하게 적용합니다.
        test_pred = (
            test_pred_scaled - scaler.min_[0]
        ) / scaler.scale_[0]

        actual = (
            y_test - scaler.min_[0]
        ) / scaler.scale_[0]

        mae = float(mean_absolute_error(actual, test_pred))
        rmse = float(np.sqrt(mean_squared_error(actual, test_pred)))

        # 마지막 10거래일로 다음 거래일 종가를 추정합니다.
        last_window = scaled[-window:].reshape(1, window, 2)
        next_scaled = float(model.predict(last_window, verbose=0)[0, 0])
        next_price = float(
            (next_scaled - scaler.min_[0]) / scaler.scale_[0]
        )

        return {
            "ticker": ticker,
            "last_close": round(float(data[-1, 0]), 2),
            "predicted_next_close": round(next_price, 2),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "history": [
                round(float(v), 2) for v in data[-60:, 0]
            ],
            "message": "예측은 과거 데이터 기반 추정치이며 투자 수익을 보장하지 않습니다."
        }

    except Exception as e:
        return {"error": str(e)}
