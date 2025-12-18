import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="LSTM & CNN-LSTM Forecast", layout="wide")
st.title("📈 LSTM & CNN-LSTM Cuaca (Prediksi 7 & 14 Hari)")

# ===============================
# FUNCTIONS
# ===============================
def create_sequence(data, time_step=30):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

def inverse_tavg(pred, scaler, n_features):
    dummy = np.zeros((len(pred), n_features))
    dummy[:,0] = pred
    return scaler.inverse_transform(dummy)[:,0]

def predict_future(model, last_seq, n_days):
    future = []
    seq = last_seq.copy()
    for _ in range(n_days):
        pred = model.predict(seq.reshape(1, *seq.shape), verbose=0)
        future.append(pred[0,0])
        seq = np.roll(seq, -1, axis=0)
        seq[-1,0] = pred
    return np.array(future)

# ===============================
# UPLOAD CSV
# ===============================
uploaded_file = st.file_uploader("📂 Upload CSV Cuaca", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview Data")
    st.dataframe(df.head())

    # ===============================
    # PREPROCESSING
    # ===============================
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    df = df[['date','tavg','tmin','tmax','wspd','pres']]
    df = df.dropna()

    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear

    df[['tavg','tmin','tmax']] = df[['tavg','tmin','tmax']].rolling(7).mean()
    df = df.dropna()

    features = ['tavg','tmin','tmax','wspd','pres','month','dayofyear']

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    time_step = 30
    X, y = create_sequence(scaled, time_step)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ===============================
    # TRAIN BUTTON
    # ===============================
    if st.button("🚀 Train Model"):
        with st.spinner("Training models..."):
            # ===============================
            # LSTM
            # ===============================
            lstm = Sequential([
                LSTM(128, return_sequences=True, input_shape=(time_step, X.shape[2])),
                LSTM(64),
                Dense(1)
            ])
            lstm.compile(optimizer='adam', loss='mse')

            lstm.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
                verbose=0
            )

            # ===============================
            # CNN-LSTM
            # ===============================
            cnn_lstm = Sequential([
                Conv1D(64, 3, activation='relu', input_shape=(time_step, X.shape[2])),
                MaxPooling1D(2),
                LSTM(64),
                Dense(1)
            ])
            cnn_lstm.compile(optimizer='adam', loss='mse')

            cnn_lstm.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
                verbose=0
            )

        st.success("✅ Training selesai!")

        # ===============================
        # EVALUATION
        # ===============================
        y_pred_lstm = lstm.predict(X_test, verbose=0)
        y_pred_cnn = cnn_lstm.predict(X_test, verbose=0)

        y_test_real = inverse_tavg(y_test, scaler, len(features))
        y_pred_lstm_real = inverse_tavg(y_pred_lstm.flatten(), scaler, len(features))
        y_pred_cnn_real = inverse_tavg(y_pred_cnn.flatten(), scaler, len(features))

        rmse_lstm = np.sqrt(mean_squared_error(y_test_real, y_pred_lstm_real))
        rmse_cnn = np.sqrt(mean_squared_error(y_test_real, y_pred_cnn_real))

        col1, col2 = st.columns(2)
        col1.metric("RMSE LSTM", f"{rmse_lstm:.3f}")
        col2.metric("RMSE CNN-LSTM", f"{rmse_cnn:.3f}")

        # ===============================
        # PLOT TEST RESULT
        # ===============================
        st.subheader("📊 Hasil Prediksi Test Set")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(y_test_real, label="Actual")
        ax.plot(y_pred_lstm_real, label="LSTM")
        ax.plot(y_pred_cnn_real, label="CNN-LSTM")
        ax.legend()
        st.pyplot(fig)

        # ===============================
        # FUTURE PREDICTION
        # ===============================
        last_seq = X[-1]

        pred7_lstm = inverse_tavg(
            predict_future(lstm, last_seq, 7),
            scaler, len(features)
        )

        pred14_lstm = inverse_tavg(
            predict_future(lstm, last_seq, 14),
            scaler, len(features)
        )

        st.subheader("🔮 Prediksi 7 Hari (LSTM)")
        st.line_chart(pred7_lstm)

        st.subheader("🔮 Prediksi 14 Hari (LSTM)")
        st.line_chart(pred14_lstm)
