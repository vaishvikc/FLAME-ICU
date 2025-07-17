import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.calibration import CalibrationDisplay

# Define the same LSTM model architecture used in training
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size2, 16)
        self.fc2 = nn.Linear(16, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # First LSTM layer - output is (batch_size, seq_len, hidden_size1)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        # Second LSTM layer - output is (batch_size, seq_len, hidden_size2)
        x, _ = self.lstm2(x)
        # Get the last time step and apply dropout
        x = self.dropout2(x[:, -1, :])  # Shape becomes (batch_size, hidden_size2)
        # Dense layers
        x = self.relu(self.fc1(x))
        # Raw logits (no sigmoid) for BCEWithLogitsLoss
        x = self.fc2(x)
        return x

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
        
        # Check if disposition column exists
        if 'disposition' in group.columns:
            # Get the target (same for all rows of the same hospitalization_id)
            target = group['disposition'].iloc[0]
            targets[hosp_id] = target
        else:
            print(f"Warning: No disposition column for hospitalization_id {hosp_id}")
            targets[hosp_id] = None
        
        # Create sequence from features
        seq = group[feature_cols].values
        
        # Handle NaN values by replacing with zeros
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Pad if needed (if less than 24 hours)
        if len(seq) < 24:
            # Create padding (zeros)
            padding = np.zeros((24 - len(seq), len(feature_cols)))
            seq = np.vstack([seq, padding])
        # Truncate if more than 24 hours
        elif len(seq) > 24:
            seq = seq[:24]
        
        # Store sequence and target
        sequences[hosp_id] = seq
    
    return sequences, targets

def load_model_and_metadata(model_dir):
    """Load saved model, scaler, and feature columns"""
    model_path = os.path.join(model_dir, 'lstm_icu_mortality_model.pt')
    scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
    feature_cols_path = os.path.join(model_dir, 'feature_columns.pkl')
    
    # Load feature columns
    with open(feature_cols_path, 'rb') as f:
        feature_cols = pickle.load(f)
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Initialize model with correct input size
    input_size = len(feature_cols)
    model = LSTMModel(input_size=input_size)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    
    return model, scaler, feature_cols

def transfer_learning(data_path, model_dir, output_dir, batch_size=8, num_epochs=50, patience=10):
    """
    Perform transfer learning on a new hospital system dataset
    
    Parameters:
    -----------
    data_path : str
        Path to the new hospital data file (parquet format)
    model_dir : str
        Directory containing the original saved model and metadata
    output_dir : str
        Directory to save the transfer-learned model and results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create graphs directory if it doesn't exist
    graphs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'final', 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Load pre-trained model and metadata
    model, scaler, feature_cols = load_model_and_metadata(model_dir)
    
    # Load new hospital data
    print(f"Loading new hospital data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Data shape: {df.shape}")
    
    # Check for required columns
    required_cols = ['hospitalization_id', 'nth_hour', 'disposition'] + feature_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        print("Available columns:", df.columns.tolist())
        return None
    
    # Prepare sequences
    print("Preparing sequences...")
    sequences, targets = prepare_sequences(df, feature_cols)
    
    # Exclude samples without targets
    valid_samples = [(hosp_id, seq) for hosp_id, seq in sequences.items() if targets[hosp_id] is not None]
    if len(valid_samples) < len(sequences):
        print(f"Warning: {len(sequences) - len(valid_samples)} samples excluded due to missing disposition values")
    
    valid_hosp_ids = [item[0] for item in valid_samples]
    X = np.array([item[1] for item in valid_samples])
    y = np.array([targets[hosp_id] for hosp_id in valid_hosp_ids])
    
    print(f"Dataset has {len(X)} patients with {sum(y)} positive cases ({sum(y)/len(y)*100:.2f}%)")
    
    # Split data 50/50 for transfer learning and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    
    print(f"Transfer learning set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Reshape data for scaling
    X_train_reshape = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshape = X_test.reshape(-1, X_test.shape[-1])
    
    # Apply the same scaling as the original model
    X_train_scaled = scaler.transform(X_train_reshape)
    X_test_scaled = scaler.transform(X_test_reshape)
    
    # Reshape back to 3D
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set model to training mode
    model.train()
    
    # Create a class-weighted loss function to handle imbalance
    pos_weight = torch.tensor([len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Use a smaller learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Early stopping setup
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    # Transfer learning
    print("Starting transfer learning...")
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
        
        if train_total > 0:
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
        else:
            train_loss = float('nan')
            train_acc = float('nan')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_nan_detected = False
        
        with torch.no_grad():
            for inputs, labels in test_loader:
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
        
        if val_total > 0:
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
        else:
            val_loss = float('nan')
            val_acc = float('nan')
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')
        
        # Check for NaN values
        if got_nan or val_nan_detected or np.isnan(train_loss) or np.isnan(val_loss):
            print(f"NaN values detected in epoch {epoch+1}, stopping training")
            if best_model_state is not None:
                print("Loading last best model state")
                model.load_state_dict(best_model_state)
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
    
    # If we have a best model state, use it
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss History')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy History')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'lstm_transfer_learning_history.png'))
    
    # Evaluate on test set
    print("\nEvaluating transfer-learned model...")
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
    y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Create calibration plot
    plt.figure(figsize=(10, 6))
    disp = CalibrationDisplay.from_predictions(y_test, y_pred_proba, n_bins=5, name='Transfer-Learned LSTM')
    plt.title('Calibration Plot')
    plt.savefig(os.path.join(graphs_dir, 'lstm_transfer_learning_calibration.png'))
    
    # Save the transfer-learned model
    transfer_model_path = os.path.join(output_dir, 'transfer_lstm_model.pt')
    torch.save(model.state_dict(), transfer_model_path)
    print(f"Transfer-learned model saved to {transfer_model_path}")
    
    # Copy the scaler and feature columns to the new directory
    scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    feature_cols_path = os.path.join(output_dir, 'feature_columns.pkl')
    with open(feature_cols_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    
    # Compare with original model performance on the same test set
    print("\nComparing with original model performance...")
    original_model, _, _ = load_model_and_metadata(model_dir)
    original_model.eval()
    
    original_y_pred_proba = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = original_model(inputs)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            # Handle any potential NaN values
            probs = torch.nan_to_num(probs, nan=0.5)
            original_y_pred_proba.extend(probs.cpu().numpy())
    
    original_y_pred_proba = np.array(original_y_pred_proba).ravel()
    original_y_pred_proba = np.nan_to_num(original_y_pred_proba, nan=0.5)
    original_y_pred = (original_y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics for original model
    original_accuracy = accuracy_score(y_test, original_y_pred)
    original_roc_auc = roc_auc_score(y_test, original_y_pred_proba)
    
    print(f"Original Model Accuracy: {original_accuracy:.4f}")
    print(f"Original Model ROC AUC: {original_roc_auc:.4f}")
    
    print("\nOriginal Model Classification Report:")
    print(classification_report(y_test, original_y_pred))
    
    print("Original Model Confusion Matrix:")
    print(confusion_matrix(y_test, original_y_pred))
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'ROC AUC'],
        'Original Model': [original_accuracy, original_roc_auc],
        'Transfer-Learned Model': [accuracy, roc_auc],
        'Improvement': [accuracy - original_accuracy, roc_auc - original_roc_auc]
    })
    
    print("\nModel Comparison:")
    print(comparison)
    
    # Save comparison to CSV
    comparison.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    return {
        'model': model,
        'original_metrics': {
            'accuracy': original_accuracy,
            'roc_auc': original_roc_auc
        },
        'transfer_metrics': {
            'accuracy': accuracy,
            'roc_auc': roc_auc
        }
    }

def main():
    """Main function for transfer learning"""
    # Get paths
    code_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(code_dir)
    model_dir = os.path.join(project_dir, 'model')
    output_dir = os.path.join(project_dir, 'model', 'transfer_learning')
    
    # For demonstration, using the original data as "new hospital data"
    # In a real scenario, this would be data from a different hospital
    data_path = os.path.join(project_dir, 'output', 'intermitted', 'by_hourly_wide_df.parquet')
    
    # Check if original model exists
    if not os.path.exists(os.path.join(model_dir, 'lstm_icu_mortality_model.pt')):
        print(f"Error: Original model not found in {model_dir}")
        return
    
    # Perform transfer learning
    results = transfer_learning(data_path, model_dir, output_dir)
    
    if results is None:
        print("Transfer learning could not be completed.")
        return
    
    print("\nTransfer learning completed!")
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main()
