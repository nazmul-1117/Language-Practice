# Day_012 | Transfer Learning using PyTorch

Transfer Learning is a powerful technique in deep learning where a model developed for a task is reused as the starting point for a model on a second, related task. In PyTorch, this is typically done using pre-trained models from the **`torchvision.models`** library.

This approach is highly effective in computer vision because the initial layers of a network (like a CNN) learn generic features (edges, textures) that are useful across many image-related tasks.

-----

## 🖼️ Two Main Strategies for Transfer Learning

When using a pre-trained model (usually trained on the massive **ImageNet** dataset), you have two primary methods to adapt it to your specific, smaller dataset:

### 1\. Feature Extractor (Recommended for Small Datasets)

  * **Goal:** Use the pre-trained convolutional base as a **fixed feature extractor**. Only the final classification head is trained.
  * **Process:** The weights in the original model's feature layers (the bulk of the network) are frozen.
  * **Benefit:** Prevents the learned, general features from being destroyed by training on a small, task-specific dataset, which could lead to overfitting.

### 2\. Fine-Tuning (Recommended for Large Datasets)

  * **Goal:** Use the pre-trained weights as a superior initialization point, then train the *entire* network (or the last few layers) on the new data.
  * **Process:** The convolutional base is usually **unfrozen**, and the entire model is trained, often with a **very low learning rate** to gently adjust the pre-trained weights.
  * **Benefit:** Allows the model to slightly specialize the high-level features learned by the base model to the nuances of the new dataset.

-----

## 🛠️ Step-by-Step Implementation in PyTorch

We will use the **Feature Extractor** approach with the popular **ResNet-18** model as an example.

### Step 1: Load the Pre-trained Model

Use `torchvision.models` to load a pre-trained model and move it to the correct device.

```python
import torch
import torch.nn as nn
import torchvision.models as models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ResNet-18 model pre-trained on ImageNet
model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model_ft = model_ft.to(DEVICE)
```

### Step 2: Freeze the Feature Extraction Layers

Iterate through the model's parameters and set their `requires_grad` attribute to `False`. This prevents Autograd from calculating gradients for these weights.

```python
# Freeze all parameters in the model
for param in model_ft.parameters():
    param.requires_grad = False
```

### Step 3: Replace the Classification Head

The original classification layer (`fc` for ResNet) maps features to 1,000 ImageNet classes. You must replace it with a new `nn.Linear` layer tailored to your specific number of classes.

```python
# Get the number of features feeding into the last FC layer
num_ftrs = model_ft.fc.in_features 
NEW_NUM_CLASSES = 5 # Example for a 5-class problem

# Replace the final layer with a new one that IS trainable by default
model_ft.fc = nn.Linear(num_ftrs, NEW_NUM_CLASSES)

# Crucially, the parameters of the NEW nn.Linear layer are automatically set to requires_grad=True
model_ft = model_ft.to(DEVICE)
```

### Step 4: Define Optimizer and Training

Since only the new final layer is trainable, you should optimize *only* those parameters.

```python
# Observe that only parameters of the final layer are optimized
optimizer = torch.optim.Adam(model_ft.fc.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

# The training loop proceeds as usual:
# for inputs, labels in dataloaders['train']:
#     ...
#     outputs = model_ft(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()
```

By providing an iterable of `model_ft.fc.parameters()` to the optimizer, you ensure that only the weights in the final classification layer are updated during `optimizer.step()`.

This video shows how you can implement transfer learning in PyTorch by modifying pre-trained models. [PyTorch Tutorial 15 - Transfer Learning - YouTube](https://www.youtube.com/watch?v=K0lWSB2QoIQ)

http://googleusercontent.com/youtube_content/3

---

### 1️⃣ Import Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import numpy as np
from PIL import Image
```

---

### 2️⃣ Load CSV and Preprocess

Assuming your CSV has **features in columns 0-727** and **label in the last column**:

```python
# Load CSV
df = pd.read_csv('fashion_mnist.csv')

# Split features and labels
X = df.iloc[:, :-1].values   # shape (60000, 728)
y = df.iloc[:, -1].values    # shape (60000,)

# Check the shape
print(X.shape, y.shape)
```

---

### 3️⃣ Reshape Data for VGG16

VGG16 expects **3-channel images of size 224x224**. Fashion-MNIST is **28x28 grayscale**. So we need to:

1. Reshape 728 → 28x26? (It's not exactly 28x28, so we can pad to 28x28)
2. Convert to 3 channels
3. Resize to 224x224

```python
class FashionMNISTDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Reshape 728 → 28x28 (pad 28x28 if necessary)
        img_array = self.X[idx]
        # If 728 → 28x26, pad 2 columns to make 28x28
        img_array = np.pad(img_array.reshape(28, 26), ((0,0),(0,2)), mode='constant')
        
        img = Image.fromarray(img_array.astype(np.uint8))
        img = img.convert("RGB")  # 3 channels
        
        if self.transform:
            img = self.transform(img)
        
        label = int(self.y[idx])
        return img, label
```

---

### 4️⃣ Define Transformations

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

---

### 5️⃣ Create Dataset and DataLoader

```python
dataset = FashionMNISTDataset(X, y, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

### 6️⃣ Load Pretrained VGG16 and Modify Classifier

Fashion-MNIST has **10 classes**:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained VGG16
vgg16 = models.vgg16(pretrained=True)

# Freeze convolutional layers
for param in vgg16.features.parameters():
    param.requires_grad = False

# Modify the classifier
vgg16.classifier[6] = nn.Linear(4096, 10)  # 10 classes
vgg16 = vgg16.to(device)
```

---

### 7️⃣ Define Loss and Optimizer

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.classifier.parameters(), lr=0.001)
```

---

### 8️⃣ Training Loop

```python
num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = vgg16(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}, Accuracy: {100*correct/total:.2f}%")
```

---

This setup:

* Handles CSV input
* Converts grayscale to 3-channel images
* Resizes to 224x224 for VGG16
* Freezes convolutional layers (transfer learning)
* Fine-tunes the classifier for 10 classes

---