import torch
import torch.nn as nn
#torch.manual_seed(42) means that we are setting the random seed for PyTorch's random number generator to 42. This is done to ensure reproducibility of results. When you set a specific seed, the sequence of random numbers generated will be the same each time you run the code, which is useful for debugging and comparing results across different runs.
torch.manual_seed(42)

# ── data ─────────────────────────────────────────────────────
X = torch.randn(100, 3)
y = 2*X[:,0] + 1.5*X[:,1] - 0.5*X[:,2] + 0.1*torch.randn(100)
y = y.unsqueeze(1)

# ── model ─────────────────────────────────────────────────────
class HousePriceNet(nn.Module):
    #what def __init__(self): means that we are defining the constructor method for the HousePriceNet class. This method is called when an instance of the class is created. It initializes the neural network's layers and parameters. In this case, it sets up a single linear layer that takes 3 input features and produces 1 output feature, which is suitable for predicting house prices based on 3 input variables.

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 1)


##def forward just defines the forward pass of the neural network. It specifies how the input data (x) flows through the network layers to produce the output. In this case, it takes the input tensor x and passes it through layer1, which is a linear transformation, and returns the result as the output of the network.
    def forward(self, x):
        return self.layer1(x)

# ── training ──────────────────────────────────────────────────
model     = HousePriceNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#for every step of this loop what happens is that we are performing the training of the neural network model. In each iteration (step) of the loop, the following operations occur:
#1. Forward Pass: The input data X is passed through the model to obtain predictions (y_pred).
#2. Loss Calculation: The loss is computed by comparing the predicted values (y_pred) with the true target values (y). The loss function used here is the Mean Squared Error (MSE), which measures the average squared difference between the predicted and actual values.
#3. Gradient Reset: The gradients of the model's parameters are reset to zero using optimizer.zero_grad().
#4. Backward Pass: The loss is backpropagated through the network using loss.mean().backward(), which computes the gradients of the loss with respect to the model's parameters.

for step in range(500):
    y_pred = model(X)
    loss   = (y_pred - y) ** 2
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"step {step:3d} | loss: {loss.mean().item():.4f}")

# ── inference ─────────────────────────────────────────────────
test_house = torch.tensor([[1.5, 3.0, -0.5]])
with torch.no_grad():
    price = model(test_house).item()
    print(f"\npredicted price: {price:.4f}")