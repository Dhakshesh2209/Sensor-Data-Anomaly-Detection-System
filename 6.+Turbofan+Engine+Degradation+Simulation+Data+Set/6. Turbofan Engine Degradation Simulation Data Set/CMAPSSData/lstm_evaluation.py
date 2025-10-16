import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# === STEP 1: Load preprocessed data ===
X = np.load("X_lstm.npy")
y = np.load("y_lstm.npy")

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

# === STEP 4: Predict on full dataset ===
X_tensor = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    predictions = model(X_tensor).squeeze().numpy()

# === STEP 5: Plot and save ===
plt.figure(figsize=(10, 6))
plt.plot(y, label="True RUL", alpha=0.6)
plt.plot(predictions, label="Predicted RUL", alpha=0.6)
plt.xlabel("Sample Index")
plt.ylabel("Remaining Useful Life")
plt.title("LSTM RUL Prediction vs Ground Truth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rul_prediction_plot.png")
plt.show()
