import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.calibration import CalibrationDisplay

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load data
print("Loading data...")
df = pd.read_parquet('code/output/intermitted/by_hourly_wide_df.parquet')
print(f"Data shape: {df.shape}")
print(f"Number of unique hospitalization_ids: {df['hospitalization_id'].nunique()}")

# Extract relevant columns (all _max and _min columns, plus hospitalization_id and disposition)
max_min_cols = [col for col in df.columns if '_max' in col or '_min' in col]
required_cols = ['hospitalization_id', 'disposition', 'nth_hour'] + max_min_cols
df_filtered = df[required_cols]

print(f"Filtered data shape: {df_filtered.shape}")
print(f"Class distribution: \n{df_filtered['disposition'].value_counts()}")
for label, count in df_filtered['disposition'].value_counts().items():
    print(f"Class {label}: {count} ({100 * count / len(df_filtered):.2f}%)")

# Group by hospitalization_id and create 24-hour sequences
def prepare_sequences(df, feature_cols):
    """
    Transform the data into 24-hour sequences for each hospitalization_id
    Apply padding for hospitalization_ids with fewer than 24 hours
    Handle missing values by filling with zeros
    """
    # Dictionary to store sequences and targets
    sequences = {}
    targets = {}
    
    # Loop through each hospitalization_id
    for hosp_id, group in df.groupby('hospitalization_id'):
        # Sort by hour
        group = group.sort_values('nth_hour')
        
        # Get the target (same for all rows of the same hospitalization_id)
        target = group['disposition'].iloc[0]
        
        # Create sequence from features
        seq = group[feature_cols].values
        
        # Handle NaN values by replacing with zeros
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Pad if needed (if less than 24 hours)
        if len(seq) < 24:
            # Create padding (zeros)
            padding = np.zeros((24 - len(seq), len(feature_cols)))
            seq = np.vstack([seq, padding])
        
        # Store sequence and target
        sequences[hosp_id] = seq
        targets[hosp_id] = target
    
    return sequences, targets

print("Preparing sequences...")
feature_cols = max_min_cols
sequences, targets = prepare_sequences(df_filtered, feature_cols)

# Convert to numpy arrays for modeling
X = np.array(list(sequences.values()))
y = np.array(list(targets.values()))

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Normalize features
# We need to reshape for the scaler, then reshape back
n_samples, n_timesteps, n_features = X_train.shape
X_train_reshaped = X_train.reshape(-1, n_features)
X_test_reshaped = X_test.reshape(-1, n_features)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

# Reshape back
X_train_scaled = X_train_scaled.reshape(X_train.shape)
X_test_scaled = X_test_scaled.reshape(X_test.shape)

# Define LSTM model using PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # First LSTM layer - output is (batch_size, seq_len, hidden_size1)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        # Second LSTM layer - we only need the output from the last time step
        # output is (batch_size, seq_len, hidden_size2)
        x, _ = self.lstm2(x)
        # Get the last time step and apply dropout
        x = self.dropout2(x[:, -1, :])  # Shape becomes (batch_size, hidden_size2)
        # Dense layers
        x = self.relu(self.fc1(x))
        # Raw logits (no sigmoid) for BCEWithLogitsLoss
        x = self.fc2(x)
        return x

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

# Create DataLoaders for batch processing
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

# Calculate class weights for imbalanced data
pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
class_weights = torch.FloatTensor([1, pos_weight])

# Create data loaders
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Initialize model, loss function, and optimizer
model = LSTMModel(input_size=n_features)

# Initialize weights properly
def init_weights(m):
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

# Apply weight initialization
model.apply(init_weights)

# Add class weights to handle imbalanced data
pos_weight = torch.tensor([len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Use a smaller learning rate for better stability
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

# Train the model
print("Training model...")
num_epochs = 100
patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

# Lists to store metrics for plotting
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    # Track if we get NaN loss
    got_nan = False
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Apply gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"NaN loss detected in training batch, skipping update")
            got_nan = True
            continue
        
        train_loss += loss.item() * inputs.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct / train_total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_nan_detected = False
    
    with torch.no_grad():
        for inputs, labels in DataLoader(TensorDataset(X_train_tensor[-int(len(X_train_tensor)*0.2):], 
                                                   y_train_tensor[-int(len(y_train_tensor)*0.2):]), 
                                     batch_size=batch_size):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected in validation batch, skipping")
                val_nan_detected = True
                continue
                
            val_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / (len(train_loader.dataset) * 0.2)
    val_acc = val_correct / val_total
    
    # Store metrics
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')
    
    # Check for NaN values
    if got_nan or val_nan_detected or np.isnan(train_loss) or np.isnan(val_loss):
        print(f"NaN values detected in epoch {epoch+1}, stopping training")
        if best_model_state is not None:
            print("Loading last best model state")
            model.load_state_dict(best_model_state)
        else:
            print("No good model state found, reinitializing model")
            model.apply(init_weights)
        break
        
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict().copy()
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

# Load best model
if best_model_state:
    model.load_state_dict(best_model_state)

# Evaluate model
print("Evaluating model...")
model.eval()
y_pred_proba = []

with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(outputs)
        # Handle any potential NaN values
        probs = torch.nan_to_num(probs, nan=0.5)
        y_pred_proba.extend(probs.cpu().numpy())

y_pred_proba = np.array(y_pred_proba).ravel()
# Make sure there are no NaN values before calculating metrics
y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5)
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_history.png')
plt.close()

# Create calibration plot
plt.figure(figsize=(10, 6))
disp = CalibrationDisplay.from_predictions(y_test, y_pred_proba, n_bins=10, name='LSTM')
plt.title('Calibration Plot')
plt.savefig('calibration_plot.png')
plt.close()

# Create model directory if it doesn't exist
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')
os.makedirs(model_dir, exist_ok=True)

# Save the trained model
model_path = os.path.join(model_dir, 'lstm_icu_mortality_model.pt')
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Save the scaler for preprocessing new data
scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {scaler_path}")

# Save feature column names for inference
feature_cols_path = os.path.join(model_dir, 'feature_columns.pkl')
with open(feature_cols_path, 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"Feature columns saved to {feature_cols_path}")

print("Evaluation completed. Results saved in training_history.png and calibration_plot.png")
