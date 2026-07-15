import torch

from nn import SimpleNet

# 1D tensor - like a list
a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# 2D tensor - like a matrix
b = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

# print shapes
print(a.shape)
print(b.shape)

# your task - do these operations and print results
print(a * 2)          # multiply every element by 2
print(b[0])           # first row of b
print(b[:, 1])        # second column of b
print(a.mean())       # mean of a
print(b.sum())        # sum of all elements in b


##training loop
import torch
import torch.nn as nn

model = SimpleNet()  # your network from before

# fake data — 1 sample, 3 inputs, target output is 1.0
x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([1.0])


##this is the training loop, we will use SGD optimizer and MSE loss function sgd means Stochastic Gradient Descent, which is a common optimization algorithm used to minimize the loss function during training of neural networks. It updates the model's parameters based on the gradients computed from the loss with respect to the parameters. and lr is the learning rate, which controls how much to change the model in response to the estimated error each time the model weights are updated. if lr is too high, the model may converge too quickly to a suboptimal solution, or even diverge. if lr is too low, the training process may be unnecessarily slow. and MSE loss function means Mean Squared Error loss function, which is a common loss function used for regression tasks. It measures the average of the squares of the errors—that is, the average squared difference between the estimated values (predictions) and the actual value (ground truth). The goal during training is to minimize this loss.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(100):
    y_pred = model(x)                      # forward pass
    loss = (y_pred - y_true) ** 2          # how wrong are we
    optimizer.zero_grad()                  # clear old gradients
    loss.backward()                        # compute new gradients
    optimizer.step()                       # update weights
    
    if step % 10 == 0:
        print(f"step {step} loss: {loss.item():.4f} pred: {y_pred.item():.4f}")
