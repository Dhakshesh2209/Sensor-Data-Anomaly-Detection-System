import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# === CONFIG ===
DATA_FILE = "anomaly_predictions_FD001.csv"

# === LOAD DATA ===
df = pd.read_csv(DATA_FILE)
engine_ids = sorted(df['unit'].unique())

# === SIDEBAR CONTROLS ===
st.sidebar.title("Engine Selector")
selected_engine = st.sidebar.selectbox("Choose Engine ID", engine_ids)

sensor_options = [col for col in df.columns if col.startswith("sensor_") and not col.endswith(("mean", "std"))]
selected_sensor = st.sidebar.selectbox("Choose Sensor", sensor_options)

# === FILTER DATA ===
df_engine = df[df['unit'] == selected_engine]

# === MAIN DASHBOARD ===
st.title("Sensor Data Anomaly Detection Dashboard")
st.subheader(f"Engine {selected_engine} - {selected_sensor}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_engine['time'], df_engine[selected_sensor], label='Sensor Reading', color='blue')
anomalies = df_engine[df_engine['anomaly_score'] == 1]
ax.scatter(anomalies['time'], anomalies[selected_sensor], color='red', label='Anomaly')
ax.set_xlabel("Cycle")
ax.set_ylabel("Sensor Value")
ax.legend()
st.pyplot(fig)

# === OPTIONAL: Show RUL and Stats ===
st.subheader("Engine Health Summary")
st.metric("Current RUL", int(df_engine['RUL'].iloc[-1]))
st.write(f"Total Anomalies Detected: {len(anomalies)}")
st.write("Recent Anomalies:")
st.dataframe(anomalies[['time', selected_sensor, 'RUL']].tail(5))
