# Day_014 | Next Word Predictor using Pytorch | LSTM using PyTorch

That's a fantastic project\! Building a **Next Word Predictor** is a classic application of Recurrent Neural Networks (RNNs), and an **LSTM (Long Short-Term Memory)** is the ideal choice for this task due to its ability to capture long-range dependencies in language.

Here is a step-by-step guide on how to build a Next Word Predictor using an LSTM in PyTorch.

-----

## 🧠 Part 1: The Next Word Prediction Model Architecture

The task is to take a sequence of words (a phrase) and predict the single word that is most likely to follow it. This is a **sequence-to-sequence** or **sequence-to-one** problem, modeled as a **multi-class classification** task where the classes are all the words in your vocabulary.

### 1\. Model Structure (`NextWordLSTM`)

We will use an `nn.LSTM` layer, which provides both the hidden state ($h_t$) and the cell state ($c_t$).

```python
import torch
import torch.nn as nn

class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(NextWordLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. Embedding Layer: Converts token indices into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim) 

        # 2. LSTM Core: Processes the input sequence
        # batch_first=True makes input/output tensors: (Batch, Sequence, Features)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        
        # 3. Output Layer: Maps the final hidden state to the vocabulary size (the number of classes)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_state=None):
        # x shape: (batch_size, sequence_length)
        
        # 1. Embedding
        embedded = self.embedding(x)
        # embedded shape: (batch_size, sequence_length, embedding_dim)
        
        # 2. LSTM Forward Pass
        # rnn_out: output for every time step
        # h_n, c_n: final hidden and cell states (the memory)
        rnn_out, (h_n, c_n) = self.lstm(embedded, hidden_state)
        
        # 3. Final Prediction
        # For prediction, we only care about the output of the LAST time step.
        # rnn_out[:, -1, :] slices the last element (the predicted next word index)
        output = self.fc(rnn_out[:, -1, :]) 
        # output shape: (batch_size, vocab_size) - These are the raw logits
        
        return output, (h_n, c_n)

    def init_hidden(self, batch_size):
        # Initialize the hidden and cell states to zeros before the first sequence
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE))
```

-----

## 📚 Part 2: Data Preprocessing for Sequence Training

The training data must be prepared so that for every input sequence, there is a corresponding target word.

### 1\. Tokenization and Vocabulary

1.  **Tokenize:** Break down your text corpus (e.g., a book or dataset) into a list of words.
2.  **Vocabulary:** Build a mapping from every unique word to a unique integer ID (`word_to_idx`) and the inverse (`idx_to_word`). The size of this map is your **`vocab_size`**.

### 2\. Creating Input/Target Sequences

For a phrase like "The quick brown fox jumps over", the training pair is structured as follows:

| Component | Sequence | Tensor Shape (Example) |
| :--- | :--- | :--- |
| **Input (X)** | `[The, quick, brown, fox, jumps]` | `(Batch_Size, Sequence_Length)` |
| **Target (Y)** | `[over]` | `(Batch_Size)` |

### 3\. Training Data Setup (Code Sketch)

```python
# Assume sequence_length = 5
# X_data would contain the index sequences: [[idx(The), idx(quick), ...]]
# Y_data would contain the next word index: [idx(over)]

# Instantiate model
VOCAB_SIZE = len(word_to_idx)
EMBEDDING_DIM = 100
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NextWordLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)

# Setup Data Loaders (Example)
# train_data is a TensorDataset of (X_data, Y_data)
# train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
```

-----

## 🏃 Part 3: Training Loop

The training process for the LSTM is similar to other PyTorch models, using **CrossEntropyLoss** because this is a classification problem.

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS = 10

# Initialize hidden and cell states outside the batch loop for a new epoch
# or inside if treating each sequence as independent.
# For simplicity here, we treat each batch independently:
# hidden = model.init_hidden(batch_size=64) 

for epoch in range(NUM_EPOCHS):
    model.train()
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        
        # 1. Zero Gradients
        optimizer.zero_grad()
        
        # Initialize hidden state for the current batch (treats batches as separate sequences)
        hidden = model.init_hidden(batch_size=batch_x.size(0))
        
        # 2. Forward Pass
        # We only need the output logits; the final hidden state is ignored here
        # (unless doing continuous sequence training across batches).
        logits, _ = model(batch_x, hidden) 
        
        # 3. Calculate Loss
        # Logits shape (64, VOCAB_SIZE) vs Target shape (64)
        loss = criterion(logits, batch_y)
        
        # 4. Backward Pass & Update
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}')
```

## 🔮 Part 4: Inference (Predicting the Next Word)

To predict, you pass the input sequence through the trained model and take the word corresponding to the highest logit.

```python
def predict_next_word(model, input_text, word_to_idx, idx_to_word, sequence_length, device):
    model.eval()
    
    # 1. Preprocess Input
    tokens = input_text.lower().split()
    
    # Take the last 'sequence_length' tokens
    if len(tokens) > sequence_length:
        tokens = tokens[-sequence_length:]
        
    # Convert tokens to indices
    input_indices = [word_to_idx.get(word, word_to_idx['<unk>']) for word in tokens]
    
    # Convert to PyTorch Tensor (add batch dimension)
    input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)
    
    # 2. Run Forward Pass
    with torch.no_grad():
        # Initialize hidden state for inference (batch size 1)
        hidden = model.init_hidden(batch_size=1)
        
        logits, _ = model(input_tensor, hidden)
        
    # 3. Get Prediction
    # Take the index with the highest probability (max logit)
    predicted_index = torch.argmax(logits, dim=1).item()
    
    # Convert index back to word
    predicted_word = idx_to_word[predicted_index]
    
    return predicted_word

# Example usage:
# prediction = predict_next_word(model, "The quick brown fox", word_to_idx, idx_to_word, 5, DEVICE)
# print(f"Next predicted word: {prediction}")
```

---

## ✅ **Next Word Predictor Using PyTorch (LSTM)**

## **1. Install & Import Libraries**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
```

---

## **2. Prepare Dataset**

For simplicity, we’ll use a tiny text corpus. You should replace this text with a real dataset.

```python
text = """hello how are you hello how is everything hello how are we doing today"""
words = text.split()

# Build vocabulary
vocab = sorted(set(words))
word_to_idx = {w:i for i, w in enumerate(vocab)}
idx_to_word = {i:w for w,i in word_to_idx.items()}

encoded = [word_to_idx[w] for w in words]
```

---

## **3. Create Sequences for Training**

We’ll use each word to predict the next word.

Example:
`hello → how`, `how → are`, `are → you`, …

```python
sequence_length = 1
X = []
y = []

for i in range(len(encoded) - sequence_length):
    X.append(encoded[i:i+sequence_length])
    y.append(encoded[i+sequence_length])

X = torch.tensor(X)
y = torch.tensor(y)
```

---

## **4. Define Dataset & DataLoader**

```python
class WordDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = WordDataset(X, y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

---

## **5. Build the LSTM Model**

```python
class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=10, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
```

---

## **6. Train the Model**

```python
vocab_size = len(vocab)
model = NextWordLSTM(vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 200
for epoch in range(epochs):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
```

---

## **7. Predict Next Word**

```python
def predict_next_word(model, word):
    model.eval()
    idx = word_to_idx[word]
    x = torch.tensor([[idx]])
    with torch.no_grad():
        output = model(x)
        predicted_idx = torch.argmax(output, dim=1).item()
    return idx_to_word[predicted_idx]
```

---

## **8. Test the Predictor**

```python
print(predict_next_word(model, "hello"))
print(predict_next_word(model, "how"))
print(predict_next_word(model, "are"))
```

Expected outputs (based on tiny dataset):

* `"hello" → "how"`
* `"how" → "are"` or `"is"`
* `"are" → "you"` or `"we"`

---

## 🎉 Done!

You now have a functional next-word prediction model using PyTorch + LSTM.

---