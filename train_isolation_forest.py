import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# === CONFIG ===
DATA_FILE = "preprocessed_FD001.csv"
SAVE_PREDICTIONS_AS = "anomaly_predictions_FD001.csv"

# === STEP 1: LOAD PREPROCESSED DATA ===
df = pd.read_csv(DATA_FILE)

# === STEP 2: SELECT FEATURES FOR MODEL ===
# Use only sensor columns and rolling features
feature_cols = [col for col in df.columns if 'sensor_' in col and ('mean' in col or 'std' in col or col.endswith(tuple(str(i) for i in range(1, 22))) )]

X = df[feature_cols]

# === STEP 3: TRAIN ISOLATION FOREST ===
model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
df['anomaly_score'] = model.fit_predict(X)
df['anomaly_score'] = df['anomaly_score'].map({1: 0, -1: 1})  # 1 = anomaly

# === STEP 4: SAVE RESULTS ===
df.to_csv(SAVE_PREDICTIONS_AS, index=False)
print("✅ Anomaly detection complete. Results saved.")

# === STEP 5: OPTIONAL VISUALIZATION ===
# Plot anomalies for a sample engine
sample_unit = 1
df_sample = df[df['unit'] == sample_unit]

plt.figure(figsize=(12, 5))
plt.plot(df_sample['time'], df_sample['sensor_2'], label='Sensor 2')
plt.scatter(df_sample[df_sample['anomaly_score'] == 1]['time'],
            df_sample[df_sample['anomaly_score'] == 1]['sensor_2'],
            color='red', label='Anomaly')
plt.title(f"Engine {sample_unit} - Sensor 2 Anomalies")
plt.xlabel("Cycle")
plt.ylabel("Sensor Reading")
plt.legend()
plt.tight_layout()
plt.show()
