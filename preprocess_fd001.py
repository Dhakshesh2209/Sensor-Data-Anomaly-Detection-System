import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# === CONFIG ===
DATA_PATH = r"C:\Users\dhaks\OneDrive\Documents\Personal tries\Sensor Data Anomaly Detection System\6.+Turbofan+Engine+Degradation+Simulation+Data+Set\6. Turbofan Engine Degradation Simulation Data Set\CMAPSSData"
FILE_NAME = "train_FD001.txt"
SAVE_AS = "preprocessed_FD001.csv"

# === STEP 1: LOAD DATA ===
def load_data(path, filename):
    full_path = os.path.join(path, filename)
    cols = ['unit', 'time'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv(full_path, sep=' ', header=None)
    df = df.dropna(axis=1, how='all')  # remove empty columns
    df.columns = cols
    return df

# === STEP 2: NORMALIZE SENSOR DATA PER ENGINE ===
def normalize_sensors(df):
    sensor_cols = [col for col in df.columns if 'sensor_' in col]
    df_scaled = df.copy()

    def scale_group(group):
        scaler = MinMaxScaler()
        group[sensor_cols] = scaler.fit_transform(group[sensor_cols])
        return group

    df_scaled = df_scaled.groupby('unit').apply(scale_group).reset_index(drop=True)
    return df_scaled, sensor_cols

# === STEP 3: FEATURE ENGINEERING ===
def add_rolling_features(df, sensor_cols, window=5):
    for col in sensor_cols:
        df[f'{col}_mean'] = df.groupby('unit')[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        df[f'{col}_std'] = df.groupby('unit')[col].transform(lambda x: x.rolling(window=window, min_periods=1).std())
    return df

# === STEP 4: RUL ESTIMATION & LABELING ===
def add_rul_and_labels(df, threshold=20):
    df['RUL'] = df.groupby('unit')['time'].transform(lambda x: x.max() - x)
    df['anomaly_label'] = df['RUL'].apply(lambda x: 1 if x < threshold else 0)
    return df

# === MAIN PIPELINE ===
def preprocess_pipeline():
    print("🔄 Loading data...")
    df = load_data(DATA_PATH, FILE_NAME)

    print("📊 Normalizing sensor data...")
    df_scaled, sensor_cols = normalize_sensors(df)

    print("🧠 Adding rolling features...")
    df_features = add_rolling_features(df_scaled, sensor_cols)

    print("📉 Estimating RUL and labeling anomalies...")
    df_final = add_rul_and_labels(df_features)

    print(f"💾 Saving preprocessed data to {SAVE_AS}...")
    df_final.to_csv(SAVE_AS, index=False)
    print("✅ Preprocessing complete.")

if __name__ == "__main__":
    preprocess_pipeline()
