import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform    = transforms.ToTensor()
train_data   = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_data    = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1   = nn.Linear(64*24*24, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model     = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss   = criterion(output, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in test_loader:
            predicted  = model(images).argmax(dim=1)
            correct   += (predicted == labels).sum().item()
            total     += labels.size(0)

    print(f"epoch {epoch+1} | accuracy: {correct/total:.2%}")