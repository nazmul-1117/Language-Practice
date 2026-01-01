# Day_010 | 🚀 Optuna Hyperparameter Tuning Pipeline | Hyperparameter Tuning the ANN using Optuna

The process involves defining a single function (`objective`) that trains and evaluates the model for a given set of hyperparameters, and then letting Optuna search the space.

### Step 1: Install Optuna and Dependencies

```bash
pip install optuna torch torchvision
```

### Step 2: Define the `objective` Function

This function is the core of the tuning process. It takes a `trial` object from Optuna, uses it to suggest hyperparameters, builds a model, trains it, and returns the evaluation metric (e.g., validation accuracy or loss).

```python
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# Mock Data Setup (Replace with your actual DataLoader)
X = torch.randn(1000, 784) 
y = torch.randint(0, 10, (1000,))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Define the Objective Function
def objective(trial):
    # --- 1. Suggest Hyperparameters ---
    
    # a. Architecture Hyperparameters
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_size = trial.suggest_int('hidden_size', 64, 256, step=64)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    # b. Optimization Hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # --- 2. Build the Dynamic Model ---
    
    layers = []
    input_dim = 784 # Fixed input size
    output_dim = 10 # Fixed output size
    
    for i in range(n_layers):
        # Linear layer
        out_dim = hidden_size
        layers.append(nn.Linear(input_dim, out_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        input_dim = out_dim
        
    # Output layer
    layers.append(nn.Linear(input_dim, output_dim))
    
    model = nn.Sequential(*layers).to(DEVICE)
    
    # --- 3. Setup Training Components ---
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # --- 4. Training and Evaluation Loop ---
    
    N_TRAIN_EPOCHS = 5
    for epoch in range(N_TRAIN_EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
    # Evaluation (Validation Accuracy)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE)
            outputs = model(X_val_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_val_batch.size(0)
            correct += (predicted == y_val_batch).sum().item()
            
    accuracy = correct / total
    
    # --- 5. Report the Metric ---
    return accuracy
```

### Step 3: Run the Optuna Study

The study manages the search process, using efficient sampling to find the best trial quickly.

```python
# Create a study object and specify the direction (maximize accuracy)
study = optuna.create_study(direction='maximize')

# Run the optimization for a number of trials
study.optimize(objective, n_trials=50) 
```

### Step 4: Analyze Results

Optuna provides easy access to the best results and visualization tools.

```python
print("--- Optimization Finished ---")
print(f"Number of finished trials: {len(study.trials)}")
print(f"Best trial value (Accuracy): {study.best_value:.4f}")
print("Best hyperparameters found:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Access the best model configuration
best_params = study.best_params
# You would retrain your final model using these best_params on the full training data
```

-----

## ✨ Optuna Key Features

  * **TPE Sampler:** Optuna uses the **Tree-structured Parzen Estimator (TPE)** algorithm by default. This is a powerful Bayesian optimization technique that intelligently suggests the next best set of parameters based on the performance of previous trials.
  * **Pruning:** Optuna can stop unpromising trials early by checking the performance periodically (e.g., after every few epochs) using `trial.report()` and `trial.should_prune()`. This saves significant computational resources.
  * **Visualization:** Optuna offers built-in tools (requires `plotly`) to visualize the search process, parameter importance, and optimization history.