import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

##what we are doing here is that we are importing the necessary libraries for building and training a neural network using PyTorch. We import torch for tensor operations, torch.nn for building neural network layers, torchvision.datasets for loading the MNIST dataset, torchvision.transforms for data preprocessing, and torch.utils.data.DataLoader for creating data loaders to efficiently load the dataset in batches.

transform    = transforms.ToTensor()
train_data   = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_data    = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

# visualize first digit
image, label = train_data[0]
print(f"label: {label}")
image_2d = image.squeeze().numpy()
for row in image_2d:
    print(''.join(['##' if pixel > 0.5 else '  ' for pixel in row]))

# model
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

model     = MNISTNet()
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
            output    = model(images)
            predicted = output.argmax(dim=1)
            correct  += (predicted == labels).sum().item()
            total    += labels.size(0)

    print(f"epoch {epoch+1} | accuracy: {correct/total:.2%}")