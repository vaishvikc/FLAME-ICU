#!/usr/bin/env python3
"""Simplified LSTM training to identify bottlenecks"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
import json
import os
import time
from sklearn.metrics import roc_auc_score

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

print("Loading data...")
start = time.time()

# Load data
with open(config['data_split']['train_file'], 'rb') as f:
    train_data = pickle.load(f)
with open(config['data_split']['test_file'], 'rb') as f:
    test_data = pickle.load(f)

X_train = train_data['X']
y_train = train_data['y']
X_test = test_data['X']
y_test = test_data['y']

print(f"Data loaded in {time.time() - start:.2f} seconds")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Simple LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out

# Convert to tensors
print("Converting to tensors...")
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

# Create data loaders
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model
model = SimpleLSTM(input_size=X_train.shape[2])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train for just 2 epochs
print("\nTraining for 2 epochs...")
for epoch in range(2):
    start_epoch = time.time()
    model.train()
    train_loss = 0.0
    batch_count = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        batch_count += 1
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = train_loss / batch_count
    print(f"Epoch {epoch+1} completed in {time.time() - start_epoch:.2f}s, Avg Loss: {avg_loss:.4f}")

# Quick evaluation
print("\nEvaluating...")
model.eval()
y_pred = []

with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = torch.sigmoid(model(inputs))
        y_pred.extend(outputs.cpu().numpy())

y_pred = np.array(y_pred).ravel()
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nSimplified training completed successfully!")