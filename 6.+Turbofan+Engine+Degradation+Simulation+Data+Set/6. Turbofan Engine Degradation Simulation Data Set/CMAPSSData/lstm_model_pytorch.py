import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# === STEP 1: Load preprocessed data ===
X = np.load("X_lstm.npy")
y = np.load("y_lstm.npy")

# === STEP 2: Convert to PyTorch tensors ===
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# === STEP 3: Train-test split ===
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# === STEP 4: Define LSTM model ===
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
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === STEP 5: Train the model ===
for epoch in range(10):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch+1}: Train Loss = {train_loss / len(train_loader):.4f}")

# === STEP 6: Save the model ===
torch.save(model.state_dict(), "lstm_rul_model.pt")
print("✅ PyTorch LSTM model trained and saved as 'lstm_rul_model.pt'")
