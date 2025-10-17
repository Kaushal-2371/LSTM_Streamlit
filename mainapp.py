import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import STL
from datetime import datetime

# ✅ Streamlit Page Setup
st.set_page_config(page_title="Stock Forecast (STL + LSTM)", layout="wide")
st.title("📈 Stock Price Forecast using STL + LSTM")

# ✅ Sidebar Inputs
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., HDFCBANK.NS)", "HDFCBANK.NS")
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date_forecast = st.sidebar.date_input("End Date", datetime(2025, 8, 6))
window_size = st.sidebar.slider("Window Size", 5, 60, 10)
epochs = st.sidebar.slider("Training Epochs", 5, 100, 10)

# ✅ Button to Run Forecast
if st.sidebar.button("Run Forecast"):
    with st.spinner("Fetching data and training model... ⏳"):
        # 1. Download Data
        data = yf.download(ticker, start=start_date, end=end_date_forecast)
        if data.empty:
            st.error("No data found. Please check the stock symbol or date range.")
        else:
            d_high = data["High"]
            st.subheader(f"📊 {ticker} Stock Data")
            st.dataframe(data.tail(5))

            # 2. STL Decomposition
            stl = STL(d_high, period=30)
            result = stl.fit()
            trend, seasonal = result.trend, result.seasonal

            # 3. Prepare LSTM Data
            def prepare_lstm_data(series, window_size):
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(series.values.reshape(-1, 1))
                X, y = [], []
                for i in range(len(scaled) - window_size):
                    X.append(scaled[i:i+window_size])
                    y.append(scaled[i+window_size])
                return np.array(X), np.array(y), scaler

            X_trend, y_trend, scaler_trend = prepare_lstm_data(trend, window_size)
            X_seasonal, y_seasonal, scaler_seasonal = prepare_lstm_data(seasonal, window_size)

            # 4. LSTM Model Builder
            def build_and_train_lstm(X, y, epochs):
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                    LSTM(50, return_sequences=True),
                    LSTM(50, return_sequences=True),
                    LSTM(50, return_sequences=False),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X, y, epochs=epochs, batch_size=10, verbose=0)
                return model

            # 5. Train Models
            model_trend = build_and_train_lstm(X_trend, y_trend, epochs)
            model_seasonal = build_and_train_lstm(X_seasonal, y_seasonal, epochs)

            # 6. Predictions
            y_trend_pred = model_trend.predict(X_trend)
            y_seasonal_pred = model_seasonal.predict(X_seasonal)

            trend_pred = scaler_trend.inverse_transform(y_trend_pred)
            seasonal_pred = scaler_seasonal.inverse_transform(y_seasonal_pred)

            final_pred = trend_pred.flatten() + seasonal_pred.flatten()
            actual = d_high.values[window_size:]

            # 7. Plot Actual vs Predicted
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(actual, label="Actual", color='blue')
            ax.plot(final_pred, label="Predicted (Trend + Seasonal)", color='orange')
            ax.set_title(f"LSTM Forecast on STL Components - {ticker}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # 8. RMSE
            rmse = np.sqrt(mean_squared_error(actual, final_pred))
            st.metric(label="📉 Final RMSE (Reconstructed)", value=f"{rmse:.4f}")

            # 9. Forecast Next Day
            def forecast_next(model, scaler, series):
                last_window = series.values[-window_size:].reshape(1, window_size, 1)
                last_scaled = scaler.transform(last_window.reshape(window_size, 1)).reshape(1, window_size, 1)
                next_scaled = model.predict(last_scaled)
                return scaler.inverse_transform(next_scaled)[0][0]

            next_trend = forecast_next(model_trend, scaler_trend, trend)
            next_seasonal = forecast_next(model_seasonal, scaler_seasonal, seasonal)
            next_day_forecast = next_trend + next_seasonal

            st.success(f"📅 **Next Day Forecasted High Price:** ₹{next_day_forecast:.2f}")

            # 10. Show Trend & Seasonal Components
            st.subheader("🧩 STL Decomposition")
            fig2, ax2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            ax2[0].plot(d_high, label="Original")
            ax2[0].legend()
            ax2[1].plot(trend, label="Trend", color="orange")
            ax2[1].legend()
            ax2[2].plot(seasonal, label="Seasonal", color="green")
            ax2[2].legend()
            st.pyplot(fig2)
else:
    st.info("👈 Adjust parameters and click **Run Forecast** to start.")
