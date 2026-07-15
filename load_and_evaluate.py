import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# spyhtame model architecture needed to load weights
class HousePriceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)
    

model = HousePriceNet()
model.load_state_dict(torch.load('house_price_model.pth'))
model.eval()
print("model loaded")


data = fetch_california_housing()
scaler = StandardScaler()
X = torch.FloatTensor(scaler.fit_transform(data.data))
y = torch.FloatTensor(data.target).unsqueeze(1)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# evaluate
with torch.no_grad():
    predictions = model(X_test)
    mse  = torch.mean((predictions - y_test) ** 2).item()
    mae  = torch.mean(torch.abs(predictions - y_test)).item()

print(f"MSE:  {mse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"MAE in dollars: ${mae * 100000:.0f}")







