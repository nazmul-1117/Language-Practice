# Day_008 | ⚡ Neural Network Training on GPU in PyTorch

### 1\. The Need for GPU Acceleration

  * **Parallelism:** GPUs are designed with thousands of small, specialized cores, making them highly efficient at performing the same calculation simultaneously across vast amounts of data—perfect for matrix multiplications and additions, which are the core operations of neural networks.
  * **Speed:** Training on a GPU can reduce training time from days or weeks to hours, allowing for faster iteration and research.

### 2\. The Core Principle: Data Transfer

To utilize the GPU, the following three components **must** be explicitly moved from CPU (Host) memory to GPU (Device) memory:

1.  **The Model Parameters (Weights & Biases)**
2.  **The Input Data (Tensors)**
3.  **The Target Labels (Tensors)**

-----

## ⚙️ Step-by-Step PyTorch Implementation

### Step 1: Identify the Target Device

Before anything else, you must detect if a CUDA-enabled GPU is available.

```python
import torch

# Define the device variable globally
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Training on GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("Training on CPU (GPU not available)")
```

### Step 2: Move the Model to the Device

The entire model and all its internal parameters must be moved to the GPU before training starts.

```python
# Assuming 'model' is your nn.Module instance
model = model.to(device)
```

  * The `.to(device)` method internally iterates through every single parameter tensor (weights and biases) in the model and moves it to the specified GPU memory.

### Step 3: Move Data and Targets within the Training Loop

Inside the training loop, the input data and labels are loaded batch-by-batch from the CPU (via the `DataLoader`). They must be moved to the GPU **before** the forward pass.

```python
# Inside the for loop: for data, targets in train_loader:
# ...

# Move input tensors to the GPU
data = data.to(device)
targets = targets.to(device) 

# Now the model, data, and targets are all on the same device (GPU),
# allowing the matrix operations to be performed on the GPU.

# ...
scores = model(data)
loss = criterion(scores, targets)
# ...
```

### Step 4: Handle Loss and Backpropagation

The loss calculation also occurs on the GPU. When the `loss.backward()` operation is called, the automatic differentiation (Autograd) and gradient calculation are also heavily accelerated by the GPU.

-----

## 🚀 Optimization for GPU Performance

To get the most out of your GPU, leverage the following `DataLoader` parameters:

| Parameter | Recommended Setting (for GPU) | Why? |
| :--- | :--- | :--- |
| **`batch_size`** | As large as your GPU VRAM allows | Larger batches keep the GPU busy, increasing throughput. |
| **`num_workers`** | 2, 4, or 8 (based on CPU cores) | Allows the CPU to load the *next* batch of data in parallel while the GPU is processing the *current* batch. |
| **`pin_memory`** | `True` | Transfers data from CPU to GPU faster by using special "pinned" memory buffers. |

### Note on Tensor Operations

**Crucially, all tensors involved in a mathematical operation MUST reside on the same device.** If you try to add a CPU tensor to a GPU tensor, PyTorch will immediately raise a `RuntimeError`. This is a common debugging point when first setting up GPU training.

---

## 🚀 1. **Check if GPU is available**

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

Example output:

```
Using device: cuda
```

---

## 🧠 2. **Move your model to GPU**

```python
model = MyNeuralNetwork()
model = model.to(device)
```

or

```python
model.to(device)
```

---

## 📦 3. **Move your data to GPU inside the training loop**

You must send **features and labels** to GPU:

```python
for features, labels in train_loader:
    features = features.to(device)
    labels = labels.to(device)
```

---

## 🏋️‍♂️ 4. **Full GPU Training Loop (Working Example)**

```python
num_epochs = 10

for epoch in range(num_epochs):
    model.train()                     # training mode

    running_loss = 0.0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
```

---

## 🧪 5. **GPU Evaluation / Accuracy**

```python
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
```

---

## ⚡ 6. **Important: Softmax Models on GPU**

For 10-class classification (e.g., Fashion-MNIST), your model should output `[batch,10]`.

Use:

```python
_, predicted = torch.max(outputs, dim=1)
```

Works on GPU or CPU.

---

## 📌 7. **Common Mistakes to Avoid**

### ❌ Forgetting to move labels to GPU

```python
labels = labels.to(device)
```

### ❌ Forgetting to move model to GPU

```python
model.to(device)
```

### ❌ Comparing predictions on GPU with labels on CPU

Always keep both on same device.

### ❌ Using `.cuda()` everywhere

Prefer `.to(device)` — portable for CPU & GPU.

---

## 🧠 8. **Full Working Example (Fashion-MNIST + GPU) — Copy/Paste Ready**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------------------------------
# 1. Device
# ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ---------------------------------------
# 2. Data
# ---------------------------------------
transform = transforms.ToTensor()

train_ds = datasets.FashionMNIST(root="data", train=True, transform=transform, download=True)
test_ds = datasets.FashionMNIST(root="data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# ---------------------------------------
# 3. Model
# ---------------------------------------
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)   # CrossEntropyLoss = no softmax needed

model = ANN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------
# 4. Training
# ---------------------------------------
for epoch in range(5):
    model.train()
    total_loss = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# ---------------------------------------
# 5. Evaluation
# ---------------------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)

        outputs = model(features)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {correct / total:.4f}")
```

---