#!/usr/bin/env python3
"""Fast LSTM training with all metrics but optimized for speed"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Load configurations
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_preprocessing_config():
    preprocessing_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'preprocessing', 'config_demo.json'
    )
    try:
        with open(preprocessing_config_path, 'r') as f:
            preprocessing_config = json.load(f)
        return preprocessing_config
    except FileNotFoundError:
        return {"site": "unknown"}

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, num_layers=2, output_size=1, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        
        if num_layers == 1:
            self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
            self.dropout1 = nn.Dropout(dropout_rate)
            self.fc1 = nn.Linear(hidden_size1, 16)
        else:
            self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
            self.dropout1 = nn.Dropout(dropout_rate)
            self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
            self.dropout2 = nn.Dropout(dropout_rate)
            self.fc1 = nn.Linear(hidden_size2, 16)
        
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        if self.num_layers == 2:
            x, _ = self.lstm2(x)
            x = self.dropout2(x)
        
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def train_lstm_fast():
    """Fast LSTM training with all metrics"""
    print("Starting Fast LSTM Training...")
    start_time = time.time()
    
    # Load configuration
    config = load_config()
    model_params = config['model_params']
    training_config = config['training_config']
    output_config = config['output_config'].copy()
    
    # Get site name
    preprocessing_config = load_preprocessing_config()
    site_name = preprocessing_config.get('site', 'unknown')
    print(f"Training LSTM model for site: {site_name}")
    
    # Update output paths
    for key in output_config:
        if isinstance(output_config[key], str):
            output_config[key] = output_config[key].replace(
                'lstm_icu_mortality_model', f'lstm_{site_name}_icu_mortality_model'
            ).replace(
                'feature_scaler', f'{site_name}_feature_scaler'
            ).replace(
                'feature_columns', f'{site_name}_feature_columns'
            ).replace(
                'metrics.json', f'{site_name}_metrics.json'
            )
    
    # Create output directories
    os.makedirs(os.path.dirname(output_config['model_path']), exist_ok=True)
    os.makedirs(output_config['plots_dir'], exist_ok=True)
    
    # Load data
    print("Loading data...")
    with open(config['data_split']['train_file'], 'rb') as f:
        train_data = pickle.load(f)
    with open(config['data_split']['test_file'], 'rb') as f:
        test_data = pickle.load(f)
    
    X_train = train_data['X']
    y_train = train_data['y']
    feature_cols = train_data['feature_cols']
    X_test = test_data['X']
    y_test = test_data['y']
    
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    n_samples, n_timesteps, n_features = X_train.shape
    
    # Normalize features (skip for speed if not critical)
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_test_reshaped = X_test.reshape(-1, n_features)
    
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # Create data loaders
    batch_size = model_params['batch_size']
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    num_layers = config.get('optimization_results', {}).get('num_layers', 2)
    model = LSTMModel(
        input_size=n_features,
        hidden_size1=model_params['hidden_size1'],
        hidden_size2=model_params['hidden_size2'],
        num_layers=num_layers,
        dropout_rate=model_params['dropout_rate']
    )
    
    # Loss and optimizer
    pos_weight = torch.tensor([len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=training_config['learning_rate'], 
        weight_decay=training_config['weight_decay']
    )
    
    # Train for fewer epochs for speed
    num_epochs = min(training_config['num_epochs'], 20)  # Cap at 20 epochs
    print(f"\nTraining for {num_epochs} epochs...")
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['gradient_clip_value'])
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Simple validation on subset
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # Use only first 20% for validation
            val_size = int(len(X_train_tensor) * 0.2)
            val_outputs = model(X_train_tensor[:val_size])
            val_loss = criterion(val_outputs, y_train_tensor[:val_size]).item()
        
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} ({time.time() - epoch_start:.1f}s) - '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
    
    print(f"\nTraining completed in {time.time() - start_time:.1f} seconds")
    
    # Evaluation
    print("\nEvaluating model...")
    model.eval()
    y_pred_proba = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = torch.sigmoid(model(inputs))
            y_pred_proba.extend(outputs.cpu().numpy())
    
    y_pred_proba = np.array(y_pred_proba).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    brier_score = brier_score_loss(y_test, y_pred_proba)
    ece = expected_calibration_error(y_test, y_pred_proba)
    
    print(f"\nResults:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    print(f"Expected Calibration Error: {ece:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Save model and results
    torch.save(model.state_dict(), output_config['model_path'])
    print(f"\nModel saved to {output_config['model_path']}")
    
    with open(output_config['scaler_path'], 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(output_config['feature_cols_path'], 'wb') as f:
        pickle.dump(feature_cols, f)
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'brier_score': float(brier_score),
        'expected_calibration_error': float(ece),
        'confusion_matrix': cm.tolist(),
        'training_time_seconds': time.time() - start_time
    }
    
    with open(output_config['metrics_path'], 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nTotal time: {time.time() - start_time:.1f} seconds")
    print(f"Results saved to {output_config['plots_dir']}")
    
    return model, scaler, feature_cols

if __name__ == "__main__":
    train_lstm_fast()