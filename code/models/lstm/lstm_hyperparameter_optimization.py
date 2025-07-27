import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_config():
    """Load configuration from config file"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config):
    """Save configuration back to config file"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_preprocessing_config():
    """Load preprocessing configuration to get site name"""
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

# Define LSTM model with flexible architecture
class OptimizableLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, num_layers=2, output_size=1, dropout_rate=0.2):
        super(OptimizableLSTMModel, self).__init__()
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

def init_weights(m):
    """Initialize weights properly"""
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

def load_presplit_lstm_data():
    """Load pre-split LSTM training data for optimization"""
    print("Loading pre-split LSTM training data...")
    
    # Load configuration
    config = load_config()
    preprocessing_config = load_preprocessing_config()
    site_name = preprocessing_config.get('site', 'unknown')
    
    # Get train file path from config
    train_file = config['data_split']['train_file']
    
    # Load training data
    print(f"Loading training data from: {train_file}")
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    
    X_train = train_data['X']
    y_train = train_data['y']
    feature_cols = train_data['feature_cols']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    print(f"Training mortality rate: {y_train.mean():.3f}")
    print(f"Number of features per timestep: {len(feature_cols)}")
    
    return X_train, y_train, feature_cols, site_name

def objective(trial):
    """Optuna objective function for LSTM hyperparameter optimization"""
    global X_train_global, y_train_global, n_features
    
    # Suggest hyperparameters
    params = {
        # Architecture parameters
        'hidden_size1': trial.suggest_categorical('hidden_size1', [32, 64, 128, 256]),
        'hidden_size2': trial.suggest_categorical('hidden_size2', [16, 32, 64, 128]),
        'num_layers': trial.suggest_categorical('num_layers', [1, 2]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        
        # Training parameters
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
        
        # Sequence parameter
        'sequence_length': trial.suggest_categorical('sequence_length', [12, 24, 48, 72])
    }
    
    # Adjust sequence length if needed
    sequence_length = params['sequence_length']
    if sequence_length != X_train_global.shape[1]:
        # Truncate or pad sequences
        if sequence_length < X_train_global.shape[1]:
            X_adjusted = X_train_global[:, :sequence_length, :]
        else:
            # Pad with zeros
            padding_size = sequence_length - X_train_global.shape[1]
            padding = np.zeros((X_train_global.shape[0], padding_size, X_train_global.shape[2]))
            X_adjusted = np.concatenate([X_train_global, padding], axis=1)
    else:
        X_adjusted = X_train_global
    
    # 5-fold stratified cross-validation
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_adjusted, y_train_global)):
        try:
            X_fold_train, X_fold_val = X_adjusted[train_idx], X_adjusted[val_idx]
            y_fold_train, y_fold_val = y_train_global[train_idx], y_train_global[val_idx]
            
            # Normalize features
            n_samples, n_timesteps, n_features = X_fold_train.shape
            X_fold_train_reshaped = X_fold_train.reshape(-1, n_features)
            X_fold_val_reshaped = X_fold_val.reshape(-1, n_features)
            
            scaler = StandardScaler()
            X_fold_train_scaled = scaler.fit_transform(X_fold_train_reshaped)
            X_fold_val_scaled = scaler.transform(X_fold_val_reshaped)
            
            # Reshape back
            X_fold_train_scaled = X_fold_train_scaled.reshape(X_fold_train.shape)
            X_fold_val_scaled = X_fold_val_scaled.reshape(X_fold_val.shape)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_fold_train_scaled)
            X_val_tensor = torch.FloatTensor(X_fold_val_scaled)
            y_train_tensor = torch.FloatTensor(y_fold_train).reshape(-1, 1)
            y_val_tensor = torch.FloatTensor(y_fold_val).reshape(-1, 1)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
            
            # Initialize model
            model = OptimizableLSTMModel(
                input_size=n_features,
                hidden_size1=params['hidden_size1'],
                hidden_size2=params['hidden_size2'],
                num_layers=params['num_layers'],
                dropout_rate=params['dropout_rate']
            )
            
            # Apply weight initialization
            model.apply(init_weights)
            
            # Set up loss and optimizer
            pos_weight = torch.tensor([len(y_fold_train[y_fold_train == 0]) / max(1, len(y_fold_train[y_fold_train == 1]))])
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
            
            # Train for limited epochs for optimization
            num_epochs = 20  # Reduced for faster optimization
            best_val_loss = float('inf')
            patience = 5
            epochs_no_improve = 0
            
            for epoch in range(num_epochs):
                # Training
                model.train()
                train_loss = 0.0
                
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Check for NaN
                    if torch.isnan(loss):
                        raise ValueError("NaN loss detected")
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                val_probs = []
                val_labels = []
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        if torch.isnan(loss):
                            raise ValueError("NaN loss detected in validation")
                        
                        val_loss += loss.item()
                        probs = torch.sigmoid(outputs)
                        val_probs.extend(probs.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())
                
                val_loss /= len(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break
            
            # Calculate ROC-AUC
            val_probs = np.array(val_probs).ravel()
            val_labels = np.array(val_labels).ravel()
            
            # Handle edge cases
            if len(np.unique(val_labels)) < 2:
                roc_auc = 0.5
            else:
                roc_auc = roc_auc_score(val_labels, val_probs)
            
            cv_scores.append(roc_auc)
            
        except Exception as e:
            print(f"Error in fold {fold}: {e}")
            cv_scores.append(0.5)  # Return neutral score on error
    
    return np.mean(cv_scores)

def lstm_hyperparameter_optimization():
    """Main function for LSTM hyperparameter optimization"""
    global X_train_global, y_train_global, n_features
    
    print("=== LSTM Hyperparameter Optimization ===")
    
    # Load pre-split training data
    X_train_global, y_train_global, feature_cols, site_name = load_presplit_lstm_data()
    n_features = X_train_global.shape[2]
    
    print(f"Training set shape: {X_train_global.shape}")
    print(f"Starting hyperparameter optimization with {n_features} features...")
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize hyperparameters
    print("Running Optuna optimization (this may take several minutes)...")
    study.optimize(objective, n_trials=50, timeout=1800)  # 30 minutes timeout, 50 trials
    
    print(f"Best ROC-AUC: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best parameters to config
    config = load_config()
    
    # Update model_params and training_config with best parameters
    best_params = study.best_params.copy()
    
    # Split parameters into appropriate config sections
    config['model_params'].update({
        'hidden_size1': best_params['hidden_size1'],
        'hidden_size2': best_params['hidden_size2'],
        'dropout_rate': best_params['dropout_rate'],
        'batch_size': best_params['batch_size']
    })
    
    config['training_config'].update({
        'learning_rate': best_params['learning_rate'],
        'weight_decay': best_params['weight_decay']
    })
    
    config['data_config'].update({
        'sequence_length': best_params['sequence_length']
    })
    
    # Add optimization metadata
    config['optimization_results'] = {
        'best_roc_auc': float(study.best_value),
        'n_trials': len(study.trials),
        'optimization_date': pd.Timestamp.now().isoformat(),
        'num_layers': best_params['num_layers']
    }
    
    save_config(config)
    print(f"✅ Best parameters saved to config.json")
    print(f"Best cross-validation ROC-AUC: {study.best_value:.4f}")
    
    # Create output directory for optimization plots
    plots_dir = "../../protected_outputs/models/lstm/optimization_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot optimization history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Optimization History')
    plt.savefig(os.path.join(plots_dir, f'lstm_optimization_history_{site_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot parameter importance
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('Parameter Importance')
    plt.savefig(os.path.join(plots_dir, f'lstm_param_importance_{site_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ LSTM Optimization complete!")
    print(f"Plots saved to: {plots_dir}")
    print(f"Updated config saved with best parameters")
    print(f"Run training.py to train the final LSTM model with optimized parameters")
    
    return study.best_params

if __name__ == "__main__":
    lstm_hyperparameter_optimization()