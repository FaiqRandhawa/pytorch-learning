import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

# data
data = fetch_california_housing()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.data)
X = torch.FloatTensor(X_scaled)
y = torch.FloatTensor(data.target).unsqueeze(1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model
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

model     = HousePriceNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training
for epoch in range(500):
    model.train()
    y_pred = model(X_train)
    loss   = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# save model what this does is that it saves the state dictionary of the trained model to a file named 'house_price_model.pth'. The state dictionary contains all the parameters (weights and biases) of the model, which can be later loaded to recreate the model with the same learned parameters. This is useful for saving the model after training so that it can be used for inference or further training without having to retrain from scratch.
torch.save(model.state_dict(), 'house_price_model.pth')
print("model saved")