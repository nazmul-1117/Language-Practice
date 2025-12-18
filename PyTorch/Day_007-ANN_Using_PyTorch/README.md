# Day_007 | Building a ANN using PyTorch

## 🧱 Step 1: Define the ANN Architecture

You must define your network by subclassing `torch.nn.Module`. This example creates a simple network with one hidden layer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleANN, self).__init__()
        
        # Define the layers (Fully Connected / Linear)
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to Hidden layer
        self.fc2 = nn.Linear(hidden_size, output_size) # Hidden layer to Output layer

    def forward(self, x):
        # Define the computational flow (Forward Pass)
        # 1. Apply the first linear layer
        x = self.fc1(x)
        
        # 2. Apply a non-linear activation function (ReLU is common for hidden layers)
        x = F.relu(x)
        
        # 3. Apply the final linear layer
        x = self.fc2(x)
        
        # For a classification task, you might apply Softmax/Sigmoid here, 
        # but often the loss function (like CrossEntropyLoss) handles it internally.
        return x
```

### Key Components:

  * **`nn.Linear(in_features, out_features)`:** This is the PyTorch implementation of a fully connected layer. It handles the linear transformation: $y = xW^T + b$.
  * **`F.relu(x)`:** The Rectified Linear Unit is a non-linear activation function applied to the output of the hidden layer. It introduces the ability for the network to learn complex patterns.

## 🛠️ Step 2: Prepare for Training

Before running the training loop, you must initialize the model, loss function, and optimizer.

```python
# Hyperparameters (Example values)
INPUT_SIZE = 784  # e.g., for a flattened 28x28 image (like MNIST)
HIDDEN_SIZE = 128 
OUTPUT_SIZE = 10  # e.g., for 10 classification classes
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Instantiate the Model
model = SimpleANN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)

# 2. Define Loss Function (Criterion)
# CrossEntropyLoss is ideal for multi-class classification
criterion = nn.CrossEntropyLoss() 

# 3. Define Optimizer
# Adam is a popular, effective optimization algorithm
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

## 🔄 Step 3: Implement the Training Loop

The training loop iterates over the data multiple times (**epochs**) and, for each batch, executes the essential four steps:

```python
# Assuming you have a DataLoader set up (e.g., train_loader)
NUM_EPOCHS = 5

for epoch in range(NUM_EPOCHS):
    # Set the model to training mode
    model.train() 
    
    # Loop over all batches in the DataLoader
    for batch_idx, (data, targets) in enumerate(train_loader):
        
        # 1. Prepare Data and Device
        # Flatten the input data for the ANN (e.g., from [Batch, 1, 28, 28] to [Batch, 784])
        data = data.to(DEVICE).reshape(data.shape[0], -1) 
        targets = targets.to(DEVICE)

        # 2. Zero Gradients
        optimizer.zero_grad() # Clear previous gradients

        # 3. Forward Pass
        scores = model(data)
        loss = criterion(scores, targets)

        # 4. Backward Pass (Backpropagation)
        loss.backward() # Compute gradients
        optimizer.step() # Update model parameters

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')
```

---

## 🚀 1. **Install & Import PyTorch**

```bash
pip install torch torchvision torchaudio
```

### Imports

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
```

---

## 📦 2. **Understanding Key Components**

### ✔️ **Tensors**

The basic building block—similar to NumPy arrays but GPU-accelerated.

### ✔️ **Dataset & Dataloader**

For loading and batching data.

### ✔️ **Neural Network Modules (`nn.Module`)**

Defines the architecture.

### ✔️ **Loss Function**

Measures prediction error.

### ✔️ **Optimizer**

Updates weights (SGD, Adam, etc.)

### ✔️ **Training Loop**

Feeds data, computes gradients, updates weights.

---

## 📊 3. **Sample Toy Dataset**

Example: binary classification

```python
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# create dataset
X, y = make_classification(
    n_samples=1000, n_features=20, n_classes=2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
```

---

## 👷 4. **Create PyTorch Dataset & DataLoader**

```python
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = MyDataset(X_train, y_train)
test_ds = MyDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)
```

---

## 🏗️ 5. **Build an ANN Model**

### Simple 3-layer network:

```python
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))   # output 0-1
        return x

model = ANN()
```

---

## ⚙️ 6. **Define Loss and Optimizer**

```python
criterion = nn.BCELoss()  # binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---

## 🏋️‍♂️ 7. **Training Loop**

```python
num_epochs = 20

for epoch in range(num_epochs):
    for data, labels in train_loader:
        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
```

---

## 🧪 8. **Evaluation**

```python
with torch.no_grad():
    correct = 0
    total = 0

    for data, labels in test_loader:
        outputs = model(data)
        predicted = (outputs>0.5).float()

        total += labels.size(0)
        correct += (predicted == labels).sum()

    print("Accuracy:", correct.item()/total)
```

---

## 💾 9. **Saving & Loading a Model**

### Save:

```python
torch.save(model.state_dict(), "ann.pth")
```

### Load:

```python
model = ANN()
model.load_state_dict(torch.load("ann.pth"))
model.eval()
```

---

## ⚡ 10. **Training on GPU (Optional)**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for data, labels in train_loader:
    data, labels = data.to(device), labels.to(device)
```

---

## 📚 11. **Complete Working Code (Copy–Paste Ready)**

If you want, I can provide:

✅ A **single run-ready Python script**
✅ A **Jupyter Notebook version**
✅ A version using **MNIST / CIFAR-10**
✅ A **deep ANN** or **CNN / RNN / LSTM**
✅ A PyTorch Lightning version
✅ Explanation with diagrams

---

## 🔧 12. **Common Tips & Best Practices**

* Use **BatchNorm** for deeper networks.
* Use **Dropout** to reduce overfitting.
* Use **Adam** optimizer + a scheduler.
* Normalize input data (`StandardScaler`).
* For real datasets, always split into train/val/test.

---