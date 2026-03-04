# Day_013 | RNN using PyTorch | Question Answering System using PyTorch

While modern QA systems heavily rely on **Transformers (like BERT or T5)**, understanding the RNN foundation is crucial. We'll start with building a basic RNN and then outline how it can be adapted for a simple QA system (specifically, an **Extractive QA** approach based on a sequence-to-sequence structure).

-----

## 🔁 Part 1: Building a Basic RNN in PyTorch

The core of an RNN is the ability to process sequences one element at a time, maintaining a **hidden state** that summarizes the information processed so far.

### 1\. The RNN Architecture

PyTorch provides high-level modules for the common RNN types: `nn.RNN`, `nn.LSTM` (Long Short-Term Memory), and `nn.GRU` (Gated Recurrent Unit). **LSTMs and GRUs are preferred** because they mitigate the **vanishing gradient problem** that plagues simple RNNs.

```python
import torch
import torch.nn as nn

class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SequenceModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. Embedding Layer: Converts input indices (word IDs) into dense vectors.
        # This is essential for all NLP tasks.
        self.embedding = nn.Embedding(input_size, hidden_size) 

        # 2. RNN Core: Using GRU for better performance than simple RNN
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # 3. Output Layer: Maps the final hidden state to the desired output (e.g., a classification or the start/end indices).
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        
        # 1. Embedding
        embedded = self.embedding(x)
        # embedded shape: (batch_size, sequence_length, hidden_size)
        
        # 2. RNN Forward Pass
        # Initializes hidden state to zeros by default if none is provided
        # h_n holds the hidden state for the last time step for all layers
        rnn_out, h_n = self.rnn(embedded) 
        # rnn_out shape: (batch_size, sequence_length, hidden_size)
        # h_n shape: (num_layers, batch_size, hidden_size)

        # 3. Final Output (Using the output from the last sequence step)
        # We often use the final hidden state or the output of the last time step
        # For sequence classification, use rnn_out[:, -1, :] 
        # For sequence generation/tagging, use rnn_out
        
        # Here we'll use the output of the last time step for classification/tagging
        output = self.fc(rnn_out)
        # output shape: (batch_size, sequence_length, output_size)
        
        return output
```

-----

## ❓ Part 2: Question Answering System using RNN/LSTM

A simple QA system can be modeled as a **sequence labeling** task. Given a **Context Document** (a sequence) and a **Question** (another sequence), the goal is to predict the **Start** and **End** indices of the answer span within the Context. This is known as **Extractive QA**.

### 1\. The RNN-based QA Model Structure

We will combine the Context and the Question into a single input and use the RNN to process it. The output layer will have two heads to predict the start and end positions.

```python
# Assuming pre-tokenized and combined input: [CLS] Question [SEP] Context [SEP]
class SimpleQAModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # GRU processes the combined sequence
        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True) 
        # Use Bidirectional GRU for better context understanding
        
        # The output size of a bidirectional GRU is 2 * hidden_size
        rnn_output_size = hidden_size * 2 
        
        # Output Head 1: Predicts the START position probability for each token in the sequence
        self.start_head = nn.Linear(rnn_output_size, 1)
        
        # Output Head 2: Predicts the END position probability for each token in the sequence
        self.end_head = nn.Linear(rnn_output_size, 1)

    def forward(self, input_ids):
        # input_ids: (batch_size, max_seq_len)
        
        embedded = self.embedding(input_ids)
        # rnn_out: (batch_size, max_seq_len, 2 * hidden_size)
        rnn_out, _ = self.rnn(embedded)
        
        # Predict logits for start and end positions
        # start_logits: (batch_size, max_seq_len, 1)
        start_logits = self.start_head(rnn_out).squeeze(-1) 
        
        # end_logits: (batch_size, max_seq_len, 1)
        end_logits = self.end_head(rnn_out).squeeze(-1)
        
        return start_logits, end_logits
```

### 2\. Training the QA Model

The key difference in training QA is the **Loss Function**. Since we are predicting the probability of a position, we treat it as a two-part classification problem:

  * **Targets:** The labels will be two index tensors: `start_index` and `end_index` (the ground truth positions within the context).
  * **Loss:** We use **CrossEntropyLoss** for *each* head separately.

<!-- end list -->

```python
# Assuming model is SimpleQAModel, targets are start_positions, end_positions

# 1. Forward Pass
start_logits, end_logits = model(input_ids)

# 2. Calculate Loss for Start Position
# We compare the predicted logits (probabilities across all sequence indices) 
# against the single ground truth start index.
start_loss = nn.CrossEntropyLoss()(start_logits, start_positions)

# 3. Calculate Loss for End Position
end_loss = nn.CrossEntropyLoss()(end_logits, end_positions)

# 4. Total Loss
total_loss = start_loss + end_loss
```

The model is optimized to minimize the sum of the classification errors for both the start and end tokens simultaneously.

### 3\. Inference

During inference, the model generates two probability distributions (one for start, one for end) over all context tokens. The predicted answer span is often chosen as the pair $(\hat{s}, \hat{e})$ that maximizes the product of their probabilities, $P(\hat{s}) \cdot P(\hat{e})$, subject to constraints (e.g., $\hat{s} \le \hat{e}$, and the span must be within a reasonable maximum length).

---

## ✅ Part 1 — RNN Using PyTorch (Simple Character-Level RNN)

### **1. Define the RNN Model**

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
        # Fully connected output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hidden = self.rnn(x)   # out: [batch, seq, hidden]
        out = self.fc(out[:, -1, :])  # last time step
        return out
```

### **2. Create Model, Loss, Optimizer**

```python
input_size = 10
hidden_size = 20
output_size = 2  # binary classification

model = SimpleRNN(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### **3. Dummy Training Loop**

```python
for epoch in range(200):
    x = torch.randn(32, 5, input_size)     # (batch, seq_len, input_size)
    y = torch.randint(0, 2, (32,))         # labels
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.4f}")
```

👉 This gives you a fully-working RNN model template.

---

## ✅ Part 2 — Question Answering System in PyTorch

Using **HuggingFace Transformers** with PyTorch backend (most common modern approach).

We will build a QA system based on **BERT for Question Answering**.

---

## **1. Install Dependencies**

```bash
pip install transformers torch
```

---

## **2. Load Model & Tokenizer**

```python
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
```

---

## **3. Inference Function**

```python
def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    start = torch.argmax(start_scores)
    end = torch.argmax(end_scores) + 1
    
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
    )

    return answer
```

---

## **4. Test with a Context + Question**

```python
context = """
PyTorch is an open source machine learning framework that accelerates the path from research 
prototyping to production deployment.
"""

question = "What is PyTorch?"

print(answer_question(question, context))
```

**Output:**

```
an open source machine learning framework
```

---