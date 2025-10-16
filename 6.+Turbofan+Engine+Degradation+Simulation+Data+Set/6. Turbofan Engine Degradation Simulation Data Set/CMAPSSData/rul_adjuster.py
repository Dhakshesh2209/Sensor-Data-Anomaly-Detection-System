import numpy as np
import torch
import torch.nn as nn
import shap
from sklearn.ensemble import IsolationForest

# === STEP 1: Load data and model ===
X = np.load("X_lstm.npy")
sample = torch.tensor(X[100].reshape(1, 30, 21), dtype=torch.float32)

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

model = LSTMModel(input_size=21, hidden_size=64, output_size=1)
model.load_state_dict(torch.load("lstm_rul_model.pt"))
model.eval()

# === STEP 2: Predict RUL ===
with torch.no_grad():
    predicted_rul = model(sample).item()

# === STEP 3: Run anomaly detection ===
iso_forest = IsolationForest(contamination=0.05)
iso_forest.fit(X.reshape(X.shape[0], -1))  # flatten for training
anomaly_flag = iso_forest.predict(sample.reshape(1, -1))[0] == -1

# === STEP 4: Run SHAP ===
background = torch.tensor(X[:100], dtype=torch.float32)
explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(sample)
sensor_shap = shap_values[0][0, -1, :]  # last timestep

# === STEP 5: Adjust RUL if anomaly detected ===
if anomaly_flag:
    critical_sensors = np.argsort(np.abs(sensor_shap))[-3:]  # top 3 sensors
    penalty = 0.2 * predicted_rul
    adjusted_rul = predicted_rul - penalty
    print(f"⚠️ Anomaly detected. Penalizing RUL by 20%.")
    print(f"Critical sensors: {[f'Sensor_{i+1}' for i in critical_sensors]}")
else:
    adjusted_rul = predicted_rul
    print("✅ No anomaly detected.")

# === STEP 6: Final Output ===
print(f"Predicted RUL: {predicted_rul:.2f}")
print(f"Adjusted RUL: {adjusted_rul:.2f}")
