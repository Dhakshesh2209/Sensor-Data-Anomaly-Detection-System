import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt

# === STEP 1: Load preprocessed data ===
X = np.load("X_lstm.npy")
y = np.load("y_lstm.npy")
sample = torch.tensor(X[100].reshape(1, 30, 21), dtype=torch.float32)

# === STEP 2: Define model architecture ===
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

# === STEP 3: Load trained model ===
model = LSTMModel(input_size=21, hidden_size=64, output_size=1)
model.load_state_dict(torch.load("lstm_rul_model.pt"))
model.eval()

# === STEP 4: Prepare SHAP GradientExplainer ===
background = torch.tensor(X[:100], dtype=torch.float32)
explainer = shap.GradientExplainer(model, background)

# === STEP 5: Run SHAP on one sequence ===
shap_values = explainer.shap_values(sample)

# === STEP 6: Extract SHAP values for last timestep
sensor_shap = shap_values[0][0, -1, :]  # shape: (21,)
sensor_names = [f"Sensor_{i+1}" for i in range(21)]

# === STEP 7: Plot SHAP bar chart ===
plt.figure(figsize=(10, 6))
plt.barh(sensor_names, sensor_shap)
plt.xlabel("SHAP Value (Impact on RUL)")
plt.title("Sensor Importance at Final Timestep")
plt.tight_layout()
plt.savefig("shap_sensor_importance.png")
print("✅ SHAP sensor importance saved as 'shap_sensor_importance.png'")
