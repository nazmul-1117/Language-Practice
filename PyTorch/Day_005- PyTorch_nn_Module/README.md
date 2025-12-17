# Day_005 | torch.nn.Module

The **`torch.nn.Module`** class is arguably the most important component in PyTorch after the Tensor itself. It serves as the **base class** for all neural network modules, layers, and entire models.

If you are coming from Keras, the `nn.Module` is equivalent to the `tf.keras.Model` or `tf.keras.layers.Layer` base classes.

-----

## 🏗️ Anatomy of `torch.nn.Module`

Every component of a neural network—from a simple linear layer to a massive Transformer block—is built by subclassing `nn.Module`.

### 1\. The Core Structure

A custom PyTorch model or layer must implement two main methods:

| Method | Purpose | Description |
| :--- | :--- | :--- |
| **`__init__(self, *args)`** | **Constructor** | Used to define and register all the sub-modules (layers) and trainable parameters (weights, biases) that the module will use. |
| **`forward(self, x)`** | **Computation** | Defines the actual flow of computation. This method receives the input tensor(s) and returns the output tensor(s) after passing them through the defined layers. |

### 2\. Automatic Parameter Registration

When a module is initialized, PyTorch automatically registers any instances of `nn.Module` (like `nn.Linear` or a custom sub-module) assigned as an attribute within `__init__`.

This registration allows PyTorch to automatically:

  * Track all trainable parameters (tensors with `requires_grad=True`).
  * Move all parameters to a specific device (CPU/GPU) with a single command (`model.to(device)`).
  * Provide an iterable over all parameters via `model.parameters()`.

-----

## 📝 Key Functionality Inherited from `nn.Module`

Subclassing `nn.Module` provides access to a large set of essential deep learning utilities:

### 1\. Parameter and Device Management

| Method | Description |
| :--- | :--- |
| **`.to(device)`** | Moves all parameters and buffers of the module to the specified device (e.g., `'cuda'`, `'cpu'`). |
| **`.parameters()`** | Returns an iterator over the module's trainable parameters (used by the optimizer). |
| **`.state_dict()`** | Returns a Python dictionary containing the entire state of the module (all parameters and buffers), typically used for saving the model. |

### 2\. Training and Evaluation Modes

| Method | Description |
| :--- | :--- |
| **`.train()`** | Sets the module to **training mode**. This affects layers like `nn.Dropout` (which becomes active) and `nn.BatchNorm` (which updates running statistics). |
| **`.eval()`** | Sets the module to **evaluation mode**. This disables `nn.Dropout` and freezes the statistics in `nn.BatchNorm`. **Crucial for inference/testing.** |

### 3\. Layer Abstraction

The power of `nn.Module` is its ability to compose small modules into larger, complex modules (e.g., combining `nn.Linear` and `nn.ReLU` into a custom "DenseBlock"). This allows for highly modular and readable code.

-----

## 💡 Example: Creating a Simple Neural Network

Here is how you would use `nn.Module` to define a simple two-layer feedforward network:

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 1. Define the layers in __init__
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()                         # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size) # Output layer

    def forward(self, x):
        # 2. Define the computational flow in forward
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model
model = SimpleNet(input_size=784, hidden_size=128, output_size=10)
print(model)
```

In the example:

1.  `fc1`, `relu`, and `fc2` are registered as sub-modules in `__init__`.
2.  The `forward` method explicitly connects these sub-modules, defining the network's structure.
3.  When you call the model instance (e.g., `model(data)`), PyTorch automatically executes the `forward` method.