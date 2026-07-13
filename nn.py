#b[:,1] mean from every row the second column

#now we implement our first neural network in torch


import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 4)
        self.layer2 = nn.Linear(4, 4)
        self.layer3 = nn.Linear(4, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

model = SimpleNet()
x = torch.tensor([1.0, 2.0, 3.0])
out = model(x)
print(out)
