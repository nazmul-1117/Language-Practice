# Day_011 | Building a CNN using PyTorch

Building a Convolutional Neural Network (CNN) in PyTorch is straightforward, as the framework provides dedicated modules for the key CNN operations: convolution, pooling, and flattening.

A typical CNN architecture for image classification consists of alternating **Convolutional layers** and **Pooling layers**, followed by **Fully Connected (Linear) layers**.

## 🧱 Step 1: Define the CNN Architecture

You build a CNN by creating a class that inherits from `torch.nn.Module` and defining the layers in `__init__` and the data flow in `forward`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 1. Convolutional Layer 1
        # Input: 3 channels (RGB)
        # Output: 6 feature maps
        # Kernel size: 5x5
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        
        # 2. Pooling Layer 1 (Max Pooling)
        # Reduces spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 3. Convolutional Layer 2
        # Input: 6 feature maps (output of conv1)
        # Output: 16 feature maps
        # Kernel size: 5x5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # 4. Fully Connected Layers (Dense)
        # NOTE: The input size (16 * 5 * 5) must be calculated based on your input image size
        # and the output size of the final pooling layer.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes) # Final output layer (e.g., 10 classes)

    def forward(self, x):
        # 1. Conv -> ReLU -> Pool
        # Initial image shape: [Batch, 3, 32, 32] (e.g., CIFAR-10)
        x = self.pool(F.relu(self.conv1(x)))
        
        # 2. Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))
        
        # 3. Flatten the feature maps into a single vector for the FC layers
        # The '-1' tells PyTorch to automatically calculate the batch size dimension.
        # The '1' specifies the starting dimension for flattening (i.e., flatten the channels, height, and width).
        x = torch.flatten(x, 1) 
        
        # 4. Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
```

## 🔬 Step 2: Key CNN Components Explained

### 1\. `nn.Conv2d` (Convolutional Layer)

This is the feature extraction heart of the CNN.

| Parameter | Purpose | Example |
| :--- | :--- | :--- |
| **`in_channels`** | The number of color channels in the input (e.g., 3 for RGB, 1 for grayscale). | `3` |
| **`out_channels`** | The number of filters/kernels applied, which determines the depth of the output feature map. | `6` |
| **`kernel_size`** | The size of the sliding window (filter) used to extract features (e.g., $3\times3$ or $5\times5$). | `5` |
| **`stride`** (Optional) | The step size the kernel takes across the input. Default is 1. | `1` |
| **`padding`** (Optional) | Adds a border of zeros around the input to preserve spatial dimensions. | `0` or `kernel_size // 2` |

The convolution operation extracts local features from the image.

### 2\. `nn.MaxPool2d` (Pooling Layer)

This layer downsamples the feature map, reducing its spatial dimensions.

| Parameter | Purpose | Example |
| :--- | :--- | :--- |
| **`kernel_size`** | The size of the window to take the maximum value over. | `2` |
| **`stride`** | The step size of the pooling window. If set to `kernel_size` (e.g., 2), it reduces the dimensions by half. | `2` |

Pooling helps make the feature detection invariant to small shifts and drastically reduces the number of parameters and computation required in subsequent layers.

### 3\. `torch.flatten()`

Before passing the final feature maps to the fully connected layers, you must convert the 3D feature map (Channels, Height, Width) into a 1D vector. This is done by `torch.flatten()`.

## 🏃 Step 3: Training the CNN

The training steps for a CNN are identical to those of an ANN or any other PyTorch model:

1.  **Instantiate:** `model = SimpleCNN().to(DEVICE)`
2.  **Criterion & Optimizer:** `criterion = nn.CrossEntropyLoss()`, `optimizer = optim.Adam(model.parameters(), lr=0.001)`
3.  **Training Loop:**
      * `optimizer.zero_grad()`
      * `outputs = model(inputs)` (Forward Pass)
      * `loss = criterion(outputs, labels)`
      * `loss.backward()` (Backpropagation)
      * `optimizer.step()` (Parameter Update)

You can see a practical example of the full PyTorch CNN pipeline, including data loading and transformations, in the following resource. [PyTorch CNN Tutorial: Build & Train Convolutional Neural Networks in Python](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3DD-V-fS8-t0g) is a video that walks through the steps of building and training a CNN in PyTorch, covering the essential components like `nn.Conv2d` and `nn.MaxPool2d`.