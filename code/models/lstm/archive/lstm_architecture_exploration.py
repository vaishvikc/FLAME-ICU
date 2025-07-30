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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def load_config():
    """Load configuration from config file"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

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

# Define different LSTM architectures
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
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional
        
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
        
        # Attention mechanism
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
        
        # Build stacked LSTM layers
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(nn.LSTM(prev_size, hidden_size, batch_first=True))
            if i < len(hidden_sizes) - 1:  # Don't add dropout after the last layer
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

def load_presplit_lstm_data():
    """Load pre-split LSTM training data"""
    print("Loading pre-split LSTM training data...")
    
    config = load_config()
    preprocessing_config = load_preprocessing_config()
    site_name = preprocessing_config.get('site', 'unknown')
    
    train_file = config['data_split']['train_file']
    
    print(f"Loading training data from: {train_file}")
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    
    X_train = train_data['X']
    y_train = train_data['y']
    feature_cols = train_data['feature_cols']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    print(f"Training mortality rate: {y_train.mean():.3f}")
    
    return X_train, y_train, feature_cols, site_name

def evaluate_architecture(model_class, model_params, X_train, y_train, config):
    """Evaluate a single architecture using cross-validation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 5-fold stratified cross-validation
    cv_scores = {'roc_auc': [], 'pr_auc': [], 'brier_score': []}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Convert to tensors
        X_fold_train_tensor = torch.FloatTensor(X_fold_train)
        y_fold_train_tensor = torch.FloatTensor(y_fold_train).unsqueeze(1)
        X_fold_val_tensor = torch.FloatTensor(X_fold_val)
        y_fold_val_tensor = torch.FloatTensor(y_fold_val).unsqueeze(1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_fold_train_tensor, y_fold_train_tensor)
        val_dataset = TensorDataset(X_fold_val_tensor, y_fold_val_tensor)
        
        batch_size = config['model_params']['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = model_class(**model_params).to(device)
        model.apply(init_weights)
        
        # Loss function with class weights
        pos_weight = torch.tensor([len(y_fold_train[y_fold_train == 0]) / max(1, len(y_fold_train[y_fold_train == 1]))]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), 
                             lr=config['training_config']['learning_rate'],
                             weight_decay=config['training_config']['weight_decay'])
        
        # Training
        num_epochs = 30  # Reduced for exploration
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training_config']['gradient_clip_value'])
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    # Store predictions
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    all_preds.extend(probs.flatten())
                    all_labels.extend(batch_y.cpu().numpy().flatten())
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
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

def explore_architectures():
    """Main function to explore different LSTM architectures"""
    print("=== LSTM Architecture Exploration ===")
    
    # Load data and config
    X_train, y_train, feature_cols, site_name = load_presplit_lstm_data()
    config = load_config()
    
    n_features = X_train.shape[2]
    
    # Define architectures to test
    architectures = [
        {
            'name': 'Basic LSTM (1 layer, 64 units)',
            'class': BasicLSTM,
            'params': {'input_size': n_features, 'hidden_size': 64, 'num_layers': 1, 'dropout_rate': 0.2}
        },
        {
            'name': 'Basic LSTM (2 layers, 64 units)',
            'class': BasicLSTM,
            'params': {'input_size': n_features, 'hidden_size': 64, 'num_layers': 2, 'dropout_rate': 0.2}
        },
        {
            'name': 'Basic LSTM (1 layer, 128 units)',
            'class': BasicLSTM,
            'params': {'input_size': n_features, 'hidden_size': 128, 'num_layers': 1, 'dropout_rate': 0.2}
        },
        {
            'name': 'Basic LSTM (2 layers, 128 units)',
            'class': BasicLSTM,
            'params': {'input_size': n_features, 'hidden_size': 128, 'num_layers': 2, 'dropout_rate': 0.3}
        },
        {
            'name': 'Bidirectional LSTM (1 layer, 64 units)',
            'class': BidirectionalLSTM,
            'params': {'input_size': n_features, 'hidden_size': 64, 'num_layers': 1, 'dropout_rate': 0.2}
        },
        {
            'name': 'Bidirectional LSTM (2 layers, 64 units)',
            'class': BidirectionalLSTM,
            'params': {'input_size': n_features, 'hidden_size': 64, 'num_layers': 2, 'dropout_rate': 0.3}
        },
        {
            'name': 'LSTM with Attention (64 units)',
            'class': LSTMWithAttention,
            'params': {'input_size': n_features, 'hidden_size': 64, 'num_layers': 1, 'dropout_rate': 0.2}
        },
        {
            'name': 'LSTM with Attention (128 units)',
            'class': LSTMWithAttention,
            'params': {'input_size': n_features, 'hidden_size': 128, 'num_layers': 1, 'dropout_rate': 0.2}
        },
        {
            'name': 'Stacked LSTM (64->32)',
            'class': StackedLSTM,
            'params': {'input_size': n_features, 'hidden_sizes': [64, 32], 'dropout_rate': 0.2}
        },
        {
            'name': 'Stacked LSTM (128->64->32)',
            'class': StackedLSTM,
            'params': {'input_size': n_features, 'hidden_sizes': [128, 64, 32], 'dropout_rate': 0.3}
        },
        {
            'name': 'Stacked LSTM (256->128->64)',
            'class': StackedLSTM,
            'params': {'input_size': n_features, 'hidden_sizes': [256, 128, 64], 'dropout_rate': 0.3}
        }
    ]
    
    # Evaluate each architecture
    results = []
    
    for i, arch in enumerate(architectures):
        print(f"\nEvaluating {arch['name']}...")
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
        print(f"  Brier Score: {scores['brier_score']:.4f} ± {scores['brier_score_std']:.4f}")
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Sort by ROC-AUC
    results_df = results_df.sort_values('roc_auc', ascending=False)
    
    # Save results
    output_dir = "../../protected_outputs/models/lstm/architecture_exploration"
    os.makedirs(output_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(output_dir, f'{site_name}_architecture_exploration_results.csv'), index=False)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: ROC-AUC comparison
    plt.subplot(2, 2, 1)
    y_pos = np.arange(len(results_df))
    plt.barh(y_pos, results_df['roc_auc'], xerr=results_df['roc_auc_std'], capsize=5)
    plt.yticks(y_pos, results_df['architecture'])
    plt.xlabel('ROC-AUC')
    plt.title('ROC-AUC Score by Architecture')
    plt.gca().invert_yaxis()
    
    # Plot 2: PR-AUC comparison
    plt.subplot(2, 2, 2)
    plt.barh(y_pos, results_df['pr_auc'], xerr=results_df['pr_auc_std'], capsize=5, color='orange')
    plt.yticks(y_pos, results_df['architecture'])
    plt.xlabel('PR-AUC')
    plt.title('PR-AUC Score by Architecture')
    plt.gca().invert_yaxis()
    
    # Plot 3: Brier Score comparison (lower is better)
    plt.subplot(2, 2, 3)
    plt.barh(y_pos, results_df['brier_score'], xerr=results_df['brier_score_std'], capsize=5, color='green')
    plt.yticks(y_pos, results_df['architecture'])
    plt.xlabel('Brier Score (lower is better)')
    plt.title('Brier Score by Architecture')
    plt.gca().invert_yaxis()
    
    # Plot 4: Combined metrics
    plt.subplot(2, 2, 4)
    x = np.arange(len(results_df))
    width = 0.25
    
    plt.bar(x - width, results_df['roc_auc'], width, label='ROC-AUC', yerr=results_df['roc_auc_std'], capsize=5)
    plt.bar(x, results_df['pr_auc'], width, label='PR-AUC', yerr=results_df['pr_auc_std'], capsize=5)
    plt.bar(x + width, 1 - results_df['brier_score'], width, label='1 - Brier Score', yerr=results_df['brier_score_std'], capsize=5)
    
    plt.xlabel('Architecture')
    plt.ylabel('Score')
    plt.title('All Metrics Comparison')
    plt.xticks(x, [f"Arch {i+1}" for i in range(len(results_df))], rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{site_name}_architecture_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n=== Architecture Exploration Summary ===")
    print("\nTop 5 architectures by ROC-AUC:")
    print(results_df[['architecture', 'roc_auc', 'pr_auc', 'brier_score']].head())
    
    # Save best architecture info
    best_arch = results_df.iloc[0]
    best_config = {
        'architecture': best_arch['architecture'],
        'model_class': best_arch['model_class'],
        'params': best_arch['params'],
        'cv_scores': {
            'roc_auc': best_arch['roc_auc'],
            'pr_auc': best_arch['pr_auc'],
            'brier_score': best_arch['brier_score']
        }
    }
    
    with open(os.path.join(output_dir, f'{site_name}_best_architecture.json'), 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\nBest architecture: {best_arch['architecture']}")
    print(f"ROC-AUC: {best_arch['roc_auc']:.4f}")
    print(f"Results saved to: {output_dir}")
    
    return results_df

if __name__ == "__main__":
    explore_architectures()