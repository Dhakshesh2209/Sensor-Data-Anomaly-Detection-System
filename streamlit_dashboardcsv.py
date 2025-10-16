import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# === Model Definition ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = torch.relu(self.fc1(hn[-1]))
        return self.fc2(x)

# === Load Model and Data ===
model = LSTMModel(input_size=21, hidden_size=64, output_size=1)
model.load_state_dict(torch.load("lstm_rul_model.pt"))
model.eval()

X = np.load("X_lstm.npy")
background = torch.tensor(X[:100], dtype=torch.float32)
explainer = shap.GradientExplainer(model, background)

iso_forest = IsolationForest(contamination=0.05)
iso_forest.fit(X.reshape(X.shape[0], -1))

# === Streamlit UI ===
st.set_page_config(page_title="RUL Dashboard", layout="wide")
st.title("🔧 RUL Prediction Dashboard")
st.markdown("Upload a CSV file with shape **30×21** to predict Remaining Useful Life (RUL), detect anomalies, and explain sensor impact.")

# === CSV Upload ===
uploaded_file = st.file_uploader("Upload sensor sequence (.csv)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, header=None)
        if df.shape != (30, 21):
            st.error(f"❌ Invalid shape: {df.shape}. Expected (30, 21).")
        else:
            sequence = df.to_numpy()
            sample = torch.tensor(sequence.reshape(1, 30, 21), dtype=torch.float32)

            # === Predict RUL ===
            with torch.no_grad():
                predicted_rul = model(sample).item()

            # === Anomaly Detection ===
            anomaly_flag = iso_forest.predict(sequence.reshape(1, -1))[0] == -1

            # === SHAP Explanation ===
            shap_values = explainer.shap_values(sample)
            sensor_shap = shap_values[0][0, -1, :]
            sensor_names = [f"Sensor_{i+1}" for i in range(21)]
            critical_sensors = np.argsort(np.abs(sensor_shap))[-3:]

            # === Adjust RUL ===
            if anomaly_flag:
                penalty = 0.2 * predicted_rul
                adjusted_rul = predicted_rul - penalty
                st.warning("⚠️ Anomaly detected. RUL penalized by 20%.")
                st.write(f"Critical sensors: {[sensor_names[i] for i in critical_sensors]}")
            else:
                adjusted_rul = predicted_rul
                st.success("✅ No anomaly detected.")

            # === Display Results ===
            col1, col2 = st.columns(2)
            col1.metric("Predicted RUL", f"{predicted_rul:.2f} cycles")
            col2.metric("Adjusted RUL", f"{adjusted_rul:.2f} cycles")

            # === SHAP Bar Chart ===
            st.subheader("📊 Sensor Importance (Final Timestep)")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.barh(sensor_names, sensor_shap)
            ax1.set_xlabel("SHAP Value (Impact on RUL)")
            ax1.set_title("Sensor Importance")
            plt.tight_layout()
            st.pyplot(fig1)

            # === Sensor Time-Series Plot ===
            st.subheader("📈 Sensor Readings Over Time")
            selected_sensors = st.multiselect("Select sensors to plot", sensor_names, default=sensor_names[:3])
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            for sensor in selected_sensors:
                idx = sensor_names.index(sensor)
                ax2.plot(range(30), sequence[:, idx], label=sensor)
            ax2.set_xlabel("Cycle")
            ax2.set_ylabel("Sensor Value")
            ax2.set_title("Sensor Trends Over 30 Cycles")
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)
    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
else:
    st.info("📥 Please upload a CSV file to begin.")
