#!/usr/bin/env python3
"""
LSTM Architecture and Hyperparameter Optimization
Explores different LSTM architectures and optimizes hyperparameters
Uses only training data with cross-validation
"""

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
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
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
        return {"site": "unknown"}

# LSTM Architecture Classes
class BasicLSTM(nn.Module):
    """Basic LSTM with configurable layers"""
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1, dropout_rate=0.2):
        super(BasicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc(out)
        return out

class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM"""
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1, dropout_rate=0.2):
        super(BidirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True,
                           dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc(out)
        return out

class LSTMWithAttention(nn.Module):
    """LSTM with attention mechanism"""
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1, dropout_rate=0.2):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        out = self.dropout(attended_output)
        out = self.fc(out)
        return out

class StackedLSTM(nn.Module):
    """Stacked LSTM with multiple hidden sizes"""
    def __init__(self, input_size, hidden_sizes, output_size=1, dropout_rate=0.2):
        super(StackedLSTM, self).__init__()
        self.layers = nn.ModuleList()
        
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(nn.LSTM(prev_size, hidden_size, batch_first=True))
            if i < len(hidden_sizes) - 1:
                self.layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
        
    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            else:
                x = layer(x)
        
        last_output = x[:, -1, :]
        out = self.fc(last_output)
        return out

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

def load_training_data():
    """Load pre-split LSTM training data"""
    config = load_config()
    train_file = config['data_split']['train_file']
    
    print(f"Loading training data from: {train_file}")
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    
    return train_data['X'], train_data['y'], train_data['feature_cols']

def evaluate_architecture(model_class, model_params, X_train, y_train, config, num_epochs=30):
    """Evaluate a single architecture using cross-validation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cv_scores = {'roc_auc': [], 'pr_auc': [], 'brier_score': []}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Normalize features
        n_samples, n_timesteps, n_features = X_fold_train.shape
        scaler = StandardScaler()
        
        X_fold_train_flat = X_fold_train.reshape(-1, n_features)
        X_fold_val_flat = X_fold_val.reshape(-1, n_features)
        
        X_fold_train_scaled = scaler.fit_transform(X_fold_train_flat).reshape(X_fold_train.shape)
        X_fold_val_scaled = scaler.transform(X_fold_val_flat).reshape(X_fold_val.shape)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_fold_train_scaled).to(device)
        y_train_tensor = torch.FloatTensor(y_fold_train).unsqueeze(1).to(device)
        X_val_tensor = torch.FloatTensor(X_fold_val_scaled).to(device)
        y_val_tensor = torch.FloatTensor(y_fold_val).unsqueeze(1).to(device)
        
        # Create data loaders
        batch_size = config.get('model_params', {}).get('batch_size', 32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        model = model_class(**model_params).to(device)
        model.apply(init_weights)
        
        # Loss and optimizer
        pos_weight = torch.tensor([len(y_fold_train[y_fold_train == 0]) / max(1, len(y_fold_train[y_fold_train == 1]))]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Training
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # Validate
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    all_preds.extend(probs.flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    break
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        if len(np.unique(all_labels)) > 1:
            cv_scores['roc_auc'].append(roc_auc_score(all_labels, all_preds))
            cv_scores['pr_auc'].append(average_precision_score(all_labels, all_preds))
            cv_scores['brier_score'].append(brier_score_loss(all_labels, all_preds))
    
    return {
        'roc_auc': np.mean(cv_scores['roc_auc']),
        'pr_auc': np.mean(cv_scores['pr_auc']),
        'brier_score': np.mean(cv_scores['brier_score']),
        'roc_auc_std': np.std(cv_scores['roc_auc']),
        'pr_auc_std': np.std(cv_scores['pr_auc']),
        'brier_score_std': np.std(cv_scores['brier_score'])
    }

def explore_architectures(X_train, y_train, config):
    """Explore different LSTM architectures"""
    print("\n=== Phase 1: Architecture Exploration ===")
    
    n_features = X_train.shape[2]
    architectures = [
        {
            'name': 'Basic LSTM (1 layer, 64 units)',
            'class': BasicLSTM,
            'params': {'input_size': n_features, 'hidden_size': 64, 'num_layers': 1, 'dropout_rate': 0.2}
        },
        {
            'name': 'Basic LSTM (2 layers, 64 units)',
            'class': BasicLSTM,
            'params': {'input_size': n_features, 'hidden_size': 64, 'num_layers': 2, 'dropout_rate': 0.3}
        },
        {
            'name': 'Basic LSTM (1 layer, 128 units)',
            'class': BasicLSTM,
            'params': {'input_size': n_features, 'hidden_size': 128, 'num_layers': 1, 'dropout_rate': 0.2}
        },
        {
            'name': 'Bidirectional LSTM (64 units)',
            'class': BidirectionalLSTM,
            'params': {'input_size': n_features, 'hidden_size': 64, 'num_layers': 1, 'dropout_rate': 0.2}
        },
        {
            'name': 'LSTM with Attention (64 units)',
            'class': LSTMWithAttention,
            'params': {'input_size': n_features, 'hidden_size': 64, 'num_layers': 1, 'dropout_rate': 0.2}
        },
        {
            'name': 'Stacked LSTM (64->32)',
            'class': StackedLSTM,
            'params': {'input_size': n_features, 'hidden_sizes': [64, 32], 'dropout_rate': 0.2}
        }
    ]
    
    results = []
    for i, arch in enumerate(architectures):
        print(f"\n[{i+1}/{len(architectures)}] Evaluating {arch['name']}...")
        scores = evaluate_architecture(arch['class'], arch['params'], X_train, y_train, config)
        
        result = {
            'architecture': arch['name'],
            'model_class': arch['class'].__name__,
            'params': arch['params'],
            **scores
        }
        results.append(result)
        
        print(f"  ROC-AUC: {scores['roc_auc']:.4f} ± {scores['roc_auc_std']:.4f}")
        print(f"  PR-AUC: {scores['pr_auc']:.4f} ± {scores['pr_auc_std']:.4f}")
    
    # Find best architecture
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('roc_auc', ascending=False)
    best_arch = results_df.iloc[0]
    
    print(f"\nBest architecture: {best_arch['architecture']}")
    print(f"ROC-AUC: {best_arch['roc_auc']:.4f}")
    
    return best_arch, results_df

def hyperparameter_optimization(best_arch, X_train, y_train, n_trials=50):
    """Optimize hyperparameters for the best architecture"""
    print("\n=== Phase 2: Hyperparameter Optimization ===")
    print(f"Optimizing hyperparameters for: {best_arch['architecture']}")
    
    global X_train_global, y_train_global, best_model_class, best_arch_params
    X_train_global = X_train
    y_train_global = y_train
    best_model_class = globals()[best_arch['model_class']]
    best_arch_params = best_arch['params']
    
    def objective(trial):
        # Hyperparameters to optimize
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'gradient_clip': trial.suggest_float('gradient_clip', 0.5, 5.0)
        }
        
        # Update architecture params with trial dropout
        arch_params = best_arch_params.copy()
        arch_params['dropout_rate'] = params['dropout_rate']
        
        # 5-fold CV
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cv_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_global, y_train_global)):
            X_fold_train, X_fold_val = X_train_global[train_idx], X_train_global[val_idx]
            y_fold_train, y_fold_val = y_train_global[train_idx], y_train_global[val_idx]
            
            # Normalize
            scaler = StandardScaler()
            n_samples, n_timesteps, n_features = X_fold_train.shape
            
            X_fold_train_scaled = scaler.fit_transform(X_fold_train.reshape(-1, n_features)).reshape(X_fold_train.shape)
            X_fold_val_scaled = scaler.transform(X_fold_val.reshape(-1, n_features)).reshape(X_fold_val.shape)
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_fold_train_scaled).to(device),
                torch.FloatTensor(y_fold_train).unsqueeze(1).to(device)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_fold_val_scaled).to(device),
                torch.FloatTensor(y_fold_val).unsqueeze(1).to(device)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
            
            # Model
            model = best_model_class(**arch_params).to(device)
            model.apply(init_weights)
            
            # Training
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
            
            for epoch in range(20):  # Reduced epochs for optimization
                model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient_clip'])
                    optimizer.step()
            
            # Evaluate
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = torch.sigmoid(model(inputs))
                    all_preds.extend(outputs.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
            
            if len(np.unique(all_labels)) > 1:
                cv_scores.append(roc_auc_score(all_labels, all_preds))
        
        return np.mean(cv_scores)
    
    # Create study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=1800)  # 30 min timeout
    
    print(f"\nBest ROC-AUC: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params, study

def save_results(best_arch, hyperparams, results_df, study, site_name):
    """Save optimization results and update config"""
    print("\n=== Saving Results ===")
    
    # Create output directories
    arch_dir = "../../protected_outputs/models/lstm/architecture_exploration"
    opt_dir = "../../protected_outputs/models/lstm/optimization_plots"
    os.makedirs(arch_dir, exist_ok=True)
    os.makedirs(opt_dir, exist_ok=True)
    
    # Save architecture exploration results
    results_df.to_csv(os.path.join(arch_dir, f'{site_name}_architecture_results.csv'), index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # ROC-AUC comparison
    plt.subplot(2, 2, 1)
    y_pos = np.arange(len(results_df))
    plt.barh(y_pos, results_df['roc_auc'], xerr=results_df['roc_auc_std'], capsize=5)
    plt.yticks(y_pos, results_df['architecture'])
    plt.xlabel('ROC-AUC')
    plt.title('ROC-AUC by Architecture')
    plt.gca().invert_yaxis()
    
    # PR-AUC comparison
    plt.subplot(2, 2, 2)
    plt.barh(y_pos, results_df['pr_auc'], xerr=results_df['pr_auc_std'], capsize=5, color='orange')
    plt.yticks(y_pos, results_df['architecture'])
    plt.xlabel('PR-AUC')
    plt.title('PR-AUC by Architecture')
    plt.gca().invert_yaxis()
    
    # Optimization history
    plt.subplot(2, 2, 3)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Hyperparameter Optimization History')
    
    # Parameter importance
    plt.subplot(2, 2, 4)
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('Hyperparameter Importance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(opt_dir, f'{site_name}_optimization_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Update config
    config = load_config()
    
    # Update model parameters
    config['model_params']['hidden_size1'] = best_arch['params'].get('hidden_size', 
                                                                     best_arch['params'].get('hidden_sizes', [64])[0])
    if 'hidden_sizes' in best_arch['params'] and len(best_arch['params']['hidden_sizes']) > 1:
        config['model_params']['hidden_size2'] = best_arch['params']['hidden_sizes'][1]
    
    config['model_params']['dropout_rate'] = hyperparams['dropout_rate']
    config['model_params']['batch_size'] = hyperparams['batch_size']
    
    # Update training parameters
    config['training_config']['learning_rate'] = hyperparams['learning_rate']
    config['training_config']['weight_decay'] = hyperparams['weight_decay']
    config['training_config']['gradient_clip_value'] = hyperparams['gradient_clip']
    
    # Add optimization metadata
    config['optimization_results'] = {
        'best_architecture': best_arch['architecture'],
        'best_model_class': best_arch['model_class'],
        'architecture_cv_roc_auc': float(best_arch['roc_auc']),
        'optimized_cv_roc_auc': float(study.best_value),
        'optimization_date': pd.Timestamp.now().isoformat(),
        'n_trials': len(study.trials)
    }
    
    save_config(config)
    
    print(f"✅ Results saved to:")
    print(f"  Architecture results: {arch_dir}")
    print(f"  Optimization plots: {opt_dir}")
    print(f"  Updated config: config.json")

def main():
    """Main optimization pipeline"""
    print("=== LSTM Architecture & Hyperparameter Optimization ===\n")
    
    # Load data and config
    preprocessing_config = load_preprocessing_config()
    site_name = preprocessing_config.get('site', 'unknown')
    
    print(f"Site: {site_name}")
    
    X_train, y_train, feature_cols = load_training_data()
    config = load_config()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training mortality rate: {y_train.mean():.3f}")
    
    # Phase 1: Architecture exploration
    best_arch, results_df = explore_architectures(X_train, y_train, config)
    
    # Phase 2: Hyperparameter optimization
    hyperparams, study = hyperparameter_optimization(best_arch, X_train, y_train)
    
    # Save results
    save_results(best_arch, hyperparams, results_df, study, site_name)
    
    print("\n✅ Optimization complete!")
    print(f"Best architecture: {best_arch['architecture']}")
    print(f"Final CV ROC-AUC: {study.best_value:.4f}")
    print("\nRun training.py to train the final model with optimized parameters")

if __name__ == "__main__":
    main()