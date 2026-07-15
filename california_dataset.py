import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.manual_seed(42)  # Set random seed for reproducibility.

#data
data=fetch_california_housing()  # Load the California housing dataset.
scaler=StandardScaler()  # Create a StandardScaler object for feature scaling.
X_scaled=scaler.fit_transform(data.data)  # Fit the scaler to the data and transform it.
X=torch.FloatTensor(X_scaled)  # Convert the scaled features to a PyTorch FloatTensor.
y=torch.FloatTensor(data.target).unsqueeze(1)  # Convert the target to a column vector.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split the dataset into training and testing sets.


##model

class HousePriceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 64)  # Define a linear layer with 8 input features and 64 hidden features.
        self.layer2 = nn.Linear(64, 32)  # Define a linear layer with 64 hidden features and 32 output features.
        self.layer3 = nn.Linear(32, 1)  # Define a linear layer with 32 output features and 1 final output feature.

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x=torch.relu(self.layer3(x))  # Define the forward pass of the network.
        return x

model=HousePriceNet()  # Create an instance of the HousePriceNet model.
criterion=nn.MSELoss()  # Define the Mean Squared Error loss function.
optimizer=torch.optim.Adam(model.parameters(), lr=0.01)  # Define the Adam optimizer with a learning rate of 0.01.

for epoch in range(1000):
    y_pred=model(X_train)  # Perform a forward pass to get predictions.
    loss=criterion(y_pred, y_train)  # Calculate the loss between predictions and true values.
    optimizer.zero_grad()  # Clear the gradients of all optimized tensors.
    loss.backward()  # Backpropagate the loss to compute gradients.
    optimizer.step()  # Update the model parameters based on the computed gradients.

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")  # Print the loss every 50 epochs.


# ── 5. INFERENCE (changes every project) ─────────────────────
model.eval()
with torch.no_grad():
    sample = X_test[0].unsqueeze(0)
    predicted = model(sample).item()
    actual    = y_test[0].item()
    print(f"\npredicted: {predicted:.4f} | actual: {actual:.4f}")