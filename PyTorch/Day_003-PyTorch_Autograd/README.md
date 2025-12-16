# Day_003 | ⚙️ PyTorch Autograd: The Engine of Deep Learning

![image](assets/1_7w9w0tGu5pPSBeS6FbbzEQ.png)

**Autograd** (short for Automatic Differentiation) is PyTorch's primary mechanism for calculating gradients. It is an essential component because deep learning models are trained by optimizing parameters using the **gradient descent** algorithm, which requires computing the derivative of the loss function with respect to every single parameter in the network.

Autograd works by recording all the operations performed on tensors that require a gradient, forming a dynamic structure called the **Computation Graph**.

-----

## 🧠 Core Concepts of Autograd

### 1\. The Computation Graph

  * **Dynamic Nature (Define-by-Run):** Unlike older frameworks, PyTorch builds the graph *on the fly* as code is executed. Every operation on a tensor is a node in the graph, and the tensors themselves are the edges.
  * **Function Nodes:** Each operation (e.g., addition, multiplication, ReLU) is represented by a `torch.autograd.Function` object. This object knows how to perform the **forward pass** (the calculation itself) and, crucially, how to calculate the **gradient for the backward pass**.

### 2\. The `requires_grad` Flag

This boolean attribute of a `torch.Tensor` determines whether Autograd should track the history of operations on that tensor.

  * **`requires_grad=True` (Model Parameters):** Typically set for all trainable parameters (weights and biases) in a neural network. Autograd will record every operation applied to these tensors.
  * **`requires_grad=False` (Input Data):** Usually set for the input data and targets, as we don't need to compute gradients with respect to the input data, only the parameters.

### 3\. The `.grad` Attribute

Every tensor with `requires_grad=True` will have a `.grad` attribute attached to it.

  * After the backward pass is executed, the **gradient of the scalar output (e.g., the loss) with respect to that tensor** is accumulated in its `.grad` attribute.

### 4\. The `.grad_fn` Attribute

Any tensor that is the result of an operation and has `requires_grad=True` will have a `.grad_fn` attribute.

  * This attribute references the `Function` object that created the tensor. This function node is the one responsible for computing the gradients for its inputs during the backward pass.

-----

## 👣 The Backward Pass Workflow

The entire Autograd process culminates in the backward pass, which happens in three main steps during a training iteration:

### Step 1: Zeroing Gradients

Before starting a new iteration, all existing gradients must be cleared. This is because gradients are **accumulated** by default.
$$\text{optimizer.zero\_grad()}$$

  * **Why?** If you don't zero the gradients, the gradients from the current batch would be added to the gradients computed from the previous batch, leading to incorrect parameter updates.

### Step 2: The Forward Pass

The model processes the input data to produce an output, and the loss is calculated (the final scalar output).
$$\text{output} = \text{model}(\text{input})$$
$$\text{loss} = \text{criterion}(\text{output}, \text{target})$$

### Step 3: Gradient Calculation (The Magic)

Calling `.backward()` on the final scalar loss initiates the reverse process.
$$\text{loss.backward()}$$

  * Autograd traverses the computation graph **backward**, from the loss tensor back to the input parameters.
  * At each node (`.grad_fn`), the chain rule of calculus is applied to compute the local gradient and propagate it back.
  * The computed gradient is stored in the **`.grad`** attribute of all tensors where `requires_grad=True`.

### Step 4: Parameter Update

Finally, the optimizer uses the calculated gradients to adjust the model's parameters.
$$\text{optimizer.step()}$$

  * This updates the weights and biases to reduce the loss in the next iteration.

-----

## 📝 Example Code Walkthrough

```python
# 1. Tensors requiring gradients (e.g., model weights)
x = torch.tensor(2.0, requires_grad=True)

# 2. Forward pass: y = 2x^2 + 5
y = x * x # y = 4.0. y.grad_fn is <PowBackward1>
z = 2 * y + 5 # z = 13.0. z.grad_fn is <AddBackward0>

# 3. Backward pass: dz/dx is calculated
# The final result should be dz/dx = 4x. At x=2, dz/dx = 8.0
z.backward() 

# 4. Check the gradient
print(x.grad) 
# Output: tensor(8.0)
```

In this example, calling `z.backward()` uses Autograd to compute $\frac{dz}{dx}$ by applying the chain rule: $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$.

  * $\frac{dy}{dx} = 2x$ (local gradient from the power function)
  * $\frac{dz}{dy} = 2$ (local gradient from the addition function)
  * Total gradient: $(2) \cdot (2x) = 4x$. Since $x=2$, $\frac{dz}{dx} = 8.0$.

<!-- ## Images -->
