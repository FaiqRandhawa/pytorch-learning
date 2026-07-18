# pytorch learning journey

this is my personal log of learning AI and deep learning from scratch. i'm a second year CS student at FAST NUCES doing this alongside a two month internship in islamabad. the goal is to get genuinely good at this, not just surface level. i want to understand what's happening under the hood, not just call APIs.

i'm documenting everything here

---

## session log

---

### july 6 to july 10 — micrograd from scratch

built an entire autograd engine in pure python with no libraries. this was the foundation of everything.

**what i built:**

started with a Value class that wraps a single number. the difference from a regular float is that it tracks how it was created and by which other values. every operation like addition and multiplication stores a reference to the parent values that produced the result. this builds a computation graph in memory.

```python
a = Value(2.0)
b = Value(3.0)
c = a * b   # c remembers it came from a and b via multiplication
```

then added gradient storage. every Value gets a grad attribute that starts at zero. this is where the gradient will be stored after backpropagation runs.

then implemented the backward pass. for each operation there is a local rule for how gradients flow backwards through it. for addition, gradient passes through unchanged to both inputs. for multiplication, each input's gradient equals the other input's value times the output gradient. these are just the chain rule from calculus applied to each operation.

```python
# for c = a * b
# grad of a = c.grad * b.data
# grad of b = c.grad * a.data
```

then built the full backward method that walks the entire computation graph in reverse order using topological sort and calls the local backward function at each node. this is exactly what pytorch does when you call loss.backward().

then used the Value class to build a Neuron (weighted sum of inputs plus bias), a Layer (multiple neurons side by side), and an MLP (multiple layers stacked). ended up with a complete neural network in about 150 lines of pure python with no libraries.

**the key insight:** pytorch's nn.Linear, nn.Sequential and loss.backward() are doing exactly what i built here, just optimized and scaled up. nothing in pytorch is magic anymore.

---

### july 11 to july 13 — pytorch fundamentals

moved to pytorch after understanding what it does under the hood.

**tensors:**

a tensor is like the Value class but wraps millions of numbers instead of one, and can run on GPU. same concept, industrial scale.

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(a.shape)   # torch.Size([3])
print(b.shape)   # torch.Size([2, 2])
```

learned vectorization properly. instead of looping over elements, operations run on entire arrays at once. the manual dot product without loops:

```python
dot = torch.sum(arr1 * arr2)
```

**neural network with nn.Module:**

same thing as the MLP i built from scratch, but using pytorch's production tools.

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 4)
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)
```

**the training loop:**

the most important concept in all of machine learning. every model ever trained runs this loop in some form.

```python
for epoch in range(500):
    y_pred = model(X)          # forward pass
    loss   = criterion(y_pred, y)  # measure error
    optimizer.zero_grad()      # clear old gradients
    loss.backward()            # compute new gradients
    optimizer.step()           # update weights
```

why zero_grad every step: pytorch accumulates gradients by default. if you don't clear them, new gradients stack on old ones and weights get updated with wrong values.

---

### july 14 to july 15 — real data pipeline

moved from fake random data to the california housing dataset. 20,640 real houses with 8 features each.

**why feature scaling matters:**

the raw features had wildly different scales. median income was around 8, longitude was around negative 122, population was in the hundreds. big numbers dominate training and small numbers get ignored. StandardScaler fixes this by making every feature have mean zero and standard deviation one.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.data)
```

after scaling all values sit between roughly negative 2 and positive 2. the model learns from all features equally.

**train/test split:**

never evaluate on data the model trained on. that's cheating. 80 percent of the data trains the model, 20 percent tests whether it actually learned or just memorized.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**why relu matters:**

stacking linear layers without activation functions is mathematically equivalent to a single linear layer. they collapse into one. relu adds non-linearity so the network can learn complex patterns. adding relu between layers cut the plateau loss from 0.51 to 0.24.

**evaluation with MAE:**

loss numbers are hard to interpret. mean absolute error in real units tells you something human readable.

```python
mae = torch.mean(torch.abs(predictions - y_test)).item()
print(f"average error: ${mae * 100000:.0f}")
```

the model was off by about 43,000 dollars on average. not perfect but a real working model trained in 30 seconds.

**saving and loading:**

train once, save, load anytime without retraining.

```python
torch.save(model.state_dict(), 'model.pth')   # save

model = HousePriceNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()   # load
```

state_dict is just a dictionary of all the learned weight values. pytorch saves the weights, not the architecture. you have to define the same architecture before loading.

---

### july 16 — classification and MNIST

completely different type of problem. regression predicts a number. classification predicts a category.

**what changes for classification:**

output layer becomes N neurons where N equals number of classes. for MNIST that's 10 (digits 0 through 9). loss function changes to CrossEntropyLoss which handles softmax internally and punishes confident wrong answers heavily. metric changes from MAE to accuracy.

```python
self.layer3 = nn.Linear(64, 10)   # 10 outputs, one per digit
criterion = nn.CrossEntropyLoss()
predicted = output.argmax(dim=1)   # highest probability wins
```

**DataLoader:**

can't feed 60,000 images at once, that crashes memory. DataLoader feeds data in batches.

```python
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

for epoch in range(5):
    for images, labels in train_loader:   # 64 images at a time
        # train on this batch
```

shuffle=True randomizes order every epoch so the model doesn't memorize the sequence.

**result:** 97.24 percent accuracy on 10,000 test images the model never saw during training. a 3 layer network reading handwriting.

---

### july 17 — CNNs

the flat MNIST network worked but threw away spatial information. flattening a 28x28 image destroys the neighborhood relationships between pixels. pixel at position (5,5) loses its connection to (5,6).

CNNs fix this by sliding filters across the image instead of flattening first.

**how convolution works:**

a 3x3 filter slides across every position in the image. at each position it does a dot product with the image patch underneath it. the result is one number representing "how much does this pattern exist here." sliding across the whole image produces a feature map showing where the pattern exists.

the network learns what the filters should be through backpropagation, same as it learns linear layer weights. after training some filters detect edges, some detect curves, some detect corners.

**the architecture:**

```python
self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # 32 filters on grayscale input
self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 64 filters on 32 feature maps
self.fc1   = nn.Linear(64*24*24, 128)           # combine all features
self.fc2   = nn.Linear(128, 10)                 # classify
```

conv1 detects low level patterns (edges). conv2 detects patterns of patterns (shapes made of edges). only then do you flatten and classify.

**why it's slower:** sliding filters across every position of every image in every batch is more computation than simple matrix multiplication. serious deep learning uses GPUs for this. on CPU it takes a few minutes.

**result:** epoch 1 already hits 97.98 percent, better than where the flat network ended after 5 full epochs. that's the power of spatial awareness.

---

## the reusable ML template

every project follows this structure. sections 1 and 2 change per project. sections 3 and 4 almost never change.

```python
# 1. data (changes every project)
# load, scale, split into train and test

# 2. model (changes every project)
class MyNet(nn.Module):
    def __init__(self): ...
    def forward(self, x): ...

# 3. setup (rarely changes)
model     = MyNet()
criterion = nn.MSELoss()          # or CrossEntropyLoss for classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. training loop (never changes)
for epoch in range(N):
    model.train()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. evaluate
model.eval()
with torch.no_grad():
    predictions = model(X_test)
```

---

## files in this repo

**basics.py** — tensor operations, vectorization, first pytorch steps

**nn.py** — first neural network with nn.Module, simple forward pass

**house_price_predictor.py** — regression on fake data, first training loop

**template.py** — full ML pipeline on california housing dataset, real data end to end

**real_house_prices.py** — data loading and preprocessing experiments

**save_load.py** — saving trained model to disk and loading back

**load_and_evaluate.py** — loading saved model and evaluating with MAE in dollars

**mnist_classifier.py** — flat neural network for digit classification, 97 percent accuracy

**cnn_mnist.py** — convolutional neural network on MNIST, better accuracy faster

---
