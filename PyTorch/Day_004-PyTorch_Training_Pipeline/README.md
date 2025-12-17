# Day_004 | 🏗️ PyTorch Training Pipeline: The 7 Steps

The entire workflow, from data to model update, is typically broken down into seven key phases:

### Phase 1: Data Preparation & Loading

The data needs to be organized and fed into the model efficiently.

  * **Dataset:** Your raw data (e.g., images, text, tables) is wrapped in a class that inherits from `torch.utils.data.Dataset`. This class defines how to retrieve a single data sample (`__getitem__`) and the total number of samples (`__len__`).
  * **DataLoader:** This utility wraps the `Dataset` and provides an **iterable** over the data. It handles crucial functions:
      * **Batching:** Grouping individual samples into batches (e.g., 32, 64, 128 samples).
      * **Shuffling:** Randomizing the data order for each epoch to prevent the model from learning the order.
      * **Multi-processing:** Loading data in parallel (`num_workers`) to prevent CPU bottlenecks.

### Phase 2: Device Configuration

You must define the target device (CPU or GPU) and move all necessary components to it.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Phase 3: Model Definition

You define your neural network architecture by creating a class that inherits from `torch.nn.Module`.

  * **`__init__`:** Instantiates all layers and sub-modules (e.g., `nn.Linear`, `nn.Conv2d`).
  * **`forward(self, x)`:** Defines the computation flow, specifying how the input tensor `x` passes through the layers to produce the output.

### Phase 4: Loss Function and Optimizer Setup

These components define *what* the model should minimize and *how* it should update its weights.

  * **Loss Function (`criterion`):** Measures the error between the model's prediction and the true label. It must be moved to the correct device.
      * *Examples:* `nn.CrossEntropyLoss()` (for classification), `nn.MSELoss()` (for regression).
  * **Optimizer:** Implements the specific gradient descent algorithm to update the parameters.
      * *Examples:* `optim.SGD`, `optim.Adam`, `optim.AdamW` (common for Transformers/LLMs).
      * **Crucial:** The optimizer must be initialized with the model's parameters: `optimizer = optim.Adam(model.parameters(), lr=0.001)`

-----

## 🔄 Phase 5: The Training Loop (Per Epoch)

The core of the PyTorch training pipeline is a nested loop: an outer loop for **epochs** and an inner loop for **batches** (iterations).

For `epoch` in `range(num_epochs)`:

1.  **Set Mode:** `model.train()` (Enables layers like Dropout and Batch Normalization).
2.  For `data, target` in `data_loader`:
      * **Move Data:** `data, target = data.to(device), target.to(device)`
      * **Zero Gradients:** `optimizer.zero_grad()`
          * This clears accumulated gradients from the *previous* batch's backpropagation.
      * **Forward Pass:** `output = model(data)`
          * The data flows through the model, generating predictions and building the computational graph.
      * **Calculate Loss:** `loss = criterion(output, target)`
      * **Backward Pass (Backpropagation):** `loss.backward()`
          * Autograd calculates $\frac{\partial \text{Loss}}{\partial \text{Parameter}}$ for every parameter and stores it in the `.grad` attribute.
      * **Update Parameters:** `optimizer.step()`
          * The optimizer uses the calculated gradients to adjust the parameters: $W = W - \text{learning\_rate} \cdot \nabla W$.

## 📊 Phase 6: The Evaluation Loop (Inference)

After each training epoch, the model is tested on unseen validation/test data to gauge its true performance.

  * **Set Mode:** `model.eval()` (Disables Dropout and uses fixed Batch Normalization stats).
  * **Disable Gradient Tracking:** `with torch.no_grad():`
      * This saves memory and computation time since parameter updates are not performed during evaluation.
  * For `data, target` in `test_loader`:
      * **Forward Pass:** `output = model(data)`
      * **Calculate Loss/Metrics:** Compute test loss and accuracy.

## 💾 Phase 7: Saving the Model

The final step is saving the trained model's parameters (weights and biases) for later use.

  * **Recommended Method:** Save the model's **state dictionary** (the learned parameters) rather than the entire model object.
    $$\text{torch.save}(\text{model.state\_dict(), 'model\_weights.pth'})$$

-----

[PyTorch Tutorial 06 - Training Pipeline: Model, Loss, and Optimizer](https://www.youtube.com/watch?v=VVDHU_TWwUg) provides a direct, code-based walkthrough of the key steps in the PyTorch training pipeline.

http://googleusercontent.com/youtube_content/0
