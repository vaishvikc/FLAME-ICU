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

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Load preprocessing configuration to get site name
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
        print(f"Warning: Preprocessing config not found at {preprocessing_config_path}")
        return {"site": "unknown"}

# Define LSTM model using PyTorch with flexible architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, num_layers=2, output_size=1, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        
        if num_layers == 1:
            # Single LSTM layer
            self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
            self.dropout1 = nn.Dropout(dropout_rate)
            self.fc1 = nn.Linear(hidden_size1, 16)
        else:
            # Two LSTM layers
            self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
            self.dropout1 = nn.Dropout(dropout_rate)
            self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
            self.dropout2 = nn.Dropout(dropout_rate)
            self.fc1 = nn.Linear(hidden_size2, 16)
        
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)
        
    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        if self.num_layers == 2:
            # Second LSTM layer
            x, _ = self.lstm2(x)
            x = self.dropout2(x)
        
        # Get the last time step
        x = x[:, -1, :]
        
        # Dense layers
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

def load_presplit_lstm_data():
    """Load pre-split LSTM data"""
    config = load_config()
    
    # Load training data
    train_file = config['data_split']['train_file']
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    
    # Load test data
    test_file = config['data_split']['test_file']
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)
    
    return train_data, test_data

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

def train_lstm_model():
    """Train LSTM model for ICU mortality prediction using pre-split data"""
    start_time = time.time()
    
    # Load configuration
    config = load_config()
    model_params = config['model_params']
    training_config = config['training_config']
    output_config = config['output_config'].copy()  # Make a copy to modify
    
    # Load preprocessing config to get site name
    preprocessing_config = load_preprocessing_config()
    site_name = preprocessing_config.get('site', 'unknown')
    print(f"Training LSTM model for site: {site_name}")
    
    # Update output paths with site name
    for key in output_config:
        if isinstance(output_config[key], str):
            # Replace the model name with site-specific name
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
    
    # Load pre-split data
    print("Loading pre-split data...")
    train_data, test_data = load_presplit_lstm_data()
    
    X_train = train_data['X']
    y_train = train_data['y']
    feature_cols = train_data['feature_cols']
    
    X_test = test_data['X']
    y_test = test_data['y']
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training mortality rate: {y_train.mean():.3f}")
    print(f"Test mortality rate: {y_test.mean():.3f}")
    print(f"Number of features per timestep: {len(feature_cols)}")
    print(f"Sequence length: {X_train.shape[1]}")
    
    # Check for NaN values
    print(f"\nNaN values in training set: {np.isnan(X_train).sum()}")
    print(f"NaN values in test set: {np.isnan(X_test).sum()}")

    # Normalize features
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_test_reshaped = X_test.reshape(-1, n_features)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)

    # Reshape back
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)

    # Split training data into train and validation
    val_split = int(len(X_train_scaled) * 0.8)
    X_train_split = X_train_scaled[:val_split]
    y_train_split = y_train[:val_split]
    X_val_split = X_train_scaled[val_split:]
    y_val_split = y_train[val_split:]

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_split)
    X_val_tensor = torch.FloatTensor(X_val_split)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.FloatTensor(y_train_split).reshape(-1, 1)
    y_val_tensor = torch.FloatTensor(y_val_split).reshape(-1, 1)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

    # Create DataLoaders for batch processing
    batch_size = model_params['batch_size']
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model with optimized architecture
    num_layers = config.get('optimization_results', {}).get('num_layers', 2)
    model = LSTMModel(
        input_size=n_features,
        hidden_size1=model_params['hidden_size1'],
        hidden_size2=model_params['hidden_size2'],
        num_layers=num_layers,
        dropout_rate=model_params['dropout_rate']
    )

    # Apply weight initialization
    model.apply(init_weights)

    # Add class weights to handle imbalanced data
    pos_weight = torch.tensor([len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Use a smaller learning rate for better stability
    optimizer = optim.Adam(
        model.parameters(), 
        lr=training_config['learning_rate'], 
        weight_decay=training_config['weight_decay']
    )

    # Train the model
    print("Training model...")
    num_epochs = training_config['num_epochs']
    patience = training_config['patience']
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
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Apply gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=training_config['gradient_clip_value']
            )
            
            optimizer.step()
            
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
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) | '
              f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')
        
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

    print(f"\nTraining completed in {time.time() - start_time:.1f} seconds")

    # Evaluate model
    print("\nEvaluating model...")
    model.eval()
    y_pred_proba = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            y_pred_proba.extend(probs.cpu().numpy())

    y_pred_proba = np.array(y_pred_proba).ravel()
    
    # Try different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print("\nEvaluating different thresholds:")
    for threshold in thresholds:
        y_pred_t = (y_pred_proba > threshold).astype(int)
        acc = accuracy_score(y_test, y_pred_t)
        print(f"Threshold {threshold}: accuracy = {acc:.4f}, predictions sum = {y_pred_t.sum()}")
    
    # Use the standard threshold for the main evaluation
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    brier_score = brier_score_loss(y_test, y_pred_proba)
    ece = expected_calibration_error(y_test, y_pred_proba)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    print(f"Expected Calibration Error: {ece:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Create enhanced confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Survived', 'Died'], 
                yticklabels=['Survived', 'Died'])
    plt.title(f'Confusion Matrix - LSTM {site_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_config['plots_dir'], f'lstm_{site_name}_confusion_matrix.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

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
    plt.savefig(os.path.join(output_config['plots_dir'], f'lstm_{site_name}_training_history.png'))
    plt.close()

    # Enhanced calibration analysis (matching XGBoost)
    try:
        print("\n=== Calibration Analysis ===")
        
        # Create reliability diagram
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: Calibration curve (reliability diagram)
        plt.subplot(1, 3, 1)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=10
        )
        
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"LSTM (ECE={ece:.4f})")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Reliability Diagram")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Prediction histogram
        plt.subplot(1, 3, 2)
        plt.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, label='Survived', density=True)
        plt.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, label='Died', density=True)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Density")
        plt.title("Prediction Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Brier score decomposition
        plt.subplot(1, 3, 3)
        # Calculate Brier score components
        reliability = np.sum((fraction_of_positives - mean_predicted_value) ** 2 * 
                           np.histogram(y_pred_proba, bins=10)[0] / len(y_pred_proba))
        resolution = np.sum((fraction_of_positives - np.mean(y_test)) ** 2 * 
                          np.histogram(y_pred_proba, bins=10)[0] / len(y_pred_proba))
        uncertainty = np.mean(y_test) * (1 - np.mean(y_test))
        
        components = ['Reliability', 'Resolution', 'Uncertainty', 'Brier Score']
        values = [reliability, -resolution, uncertainty, brier_score]  # Resolution is subtracted in Brier
        colors = ['red', 'green', 'blue', 'orange']
        
        bars = plt.bar(components, values, color=colors, alpha=0.7)
        plt.title("Brier Score Decomposition")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_config['plots_dir'], f'lstm_{site_name}_calibration_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Enhanced calibration analysis saved successfully.")
        
        # Print calibration summary
        print(f"Calibration Summary:")
        print(f"  Expected Calibration Error (ECE): {ece:.4f}")
        print(f"  Brier Score: {brier_score:.4f}")
        print(f"  Reliability: {reliability:.4f}")
        print(f"  Resolution: {resolution:.4f}")
        print(f"  Uncertainty: {uncertainty:.4f}")
        
    except Exception as e:
        print(f"Warning: Could not create calibration analysis: {e}")

    # Feature importance using gradient-based method
    print("\n=== Computing Feature Importance (Gradient-based) ===")
    try:
        model.eval()
        feature_importance = np.zeros(n_features)
        
        # Use a subset of test data for efficiency
        n_samples_importance = min(100, len(X_test_tensor))  # Reduced for speed
        importance_loader = DataLoader(
            TensorDataset(X_test_tensor[:n_samples_importance], y_test_tensor[:n_samples_importance]),
            batch_size=1,
            shuffle=False
        )
        
        for inputs, labels in importance_loader:
            inputs.requires_grad_(True)
            outputs = model(inputs)
            
            # Get gradient with respect to inputs
            outputs.backward(torch.ones_like(outputs))
            
            # Average absolute gradients across time steps
            gradients = inputs.grad.abs().mean(dim=1).squeeze().numpy()
            feature_importance += gradients
        
        # Normalize
        feature_importance = feature_importance / n_samples_importance
        
        # Sort features by importance
        feature_indices = np.argsort(feature_importance)[::-1]
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        top_k = min(20, len(feature_cols))
        top_features = [feature_cols[i] for i in feature_indices[:top_k]]
        top_importance = feature_importance[feature_indices[:top_k]]
        
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_importance)
        plt.yticks(y_pos, top_features)
        plt.xlabel('Feature Importance (Average Gradient Magnitude)')
        plt.title(f'Top {top_k} Features - LSTM Feature Importance (Gradient-based)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_config['plots_dir'], f'lstm_{site_name}_feature_importance_gradient.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nTop 10 Features (Gradient-based):")  
        for i in range(min(10, len(feature_cols))):
            idx = feature_indices[i]
            print(f"{i+1:2d}. {feature_cols[idx]}: {feature_importance[idx]:.4f}")
            
        # Also save feature importance to metrics
        feature_importance_dict = {
            feature_cols[i]: float(feature_importance[i]) 
            for i in range(len(feature_cols))
        }
        
    except Exception as e:
        print(f"Warning: Could not compute feature importance: {e}")
        feature_importance_dict = {}

    # Save the trained model
    torch.save(model.state_dict(), output_config['model_path'])
    print(f"\nModel saved to {output_config['model_path']}")

    # Save the scaler for preprocessing new data
    with open(output_config['scaler_path'], 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {output_config['scaler_path']}")

    # Save feature column names for inference
    with open(output_config['feature_cols_path'], 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"Feature columns saved to {output_config['feature_cols_path']}")

    # Save comprehensive metrics
    metrics = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'brier_score': float(brier_score),
        'expected_calibration_error': float(ece),
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance_dict,
        'training_history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'val_acc': [float(x) for x in history['val_acc']]
        },
        'model_architecture': {
            'input_size': n_features,
            'hidden_size1': model_params['hidden_size1'],
            'hidden_size2': model_params['hidden_size2'],
            'num_layers': num_layers,
            'dropout_rate': model_params['dropout_rate'],
            'sequence_length': X_train.shape[1]
        },
        'training_time_seconds': time.time() - start_time
    }
    
    with open(output_config['metrics_path'], 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_config['metrics_path']}")

    print(f"\nTotal training time: {time.time() - start_time:.1f} seconds")
    print(f"Evaluation completed. Results saved in {output_config['plots_dir']}")
    
    return model, scaler, feature_cols

if __name__ == "__main__":
    train_lstm_model()