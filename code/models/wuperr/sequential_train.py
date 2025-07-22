import argparse
import os
import logging
import json
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .sequential_wuperr import SequentialWUPERR
from .git_model_manager import GitModelManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """LSTM model for ICU mortality prediction - same as original implementation."""
    
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
        # First LSTM layer
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        # Second LSTM layer
        x, _ = self.lstm2(x)
        # Get the last time step and apply dropout
        x = self.dropout2(x[:, -1, :])
        # Dense layers
        x = self.relu(self.fc1(x))
        # Raw logits for BCEWithLogitsLoss
        x = self.fc2(x)
        return x

def prepare_sequences(df, feature_cols):
    """
    Transform data into 24-hour sequences - same as original implementation.
    """
    sequences = {}
    targets = {}
    
    for hosp_id, group in df.groupby('hospitalization_id'):
        group = group.sort_values('nth_hour')
        target = group['disposition'].iloc[0]
        
        # Create sequence from features
        seq = group[feature_cols].values
        
        # Handle NaN values
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Pad if needed (if less than 24 hours)
        if len(seq) < 24:
            padding = np.zeros((24 - len(seq), len(feature_cols)))
            seq = np.vstack([seq, padding])
        elif len(seq) > 24:
            seq = seq[:24]
        
        sequences[hosp_id] = seq
        targets[hosp_id] = target
    
    return sequences, targets

def load_site_data(site_id: int, data_dir: str = "data/sites"):
    """
    Load data for a specific site.
    
    Args:
        site_id: ID of the site
        data_dir: Directory containing site data
    
    Returns:
        Tuple of (features, targets, feature_columns)
    """
    site_data_path = os.path.join(data_dir, f"site_{site_id}_data.parquet")
    
    if not os.path.exists(site_data_path):
        raise FileNotFoundError(f"Site data not found at {site_data_path}")
    
    logger.info(f"Loading data for site {site_id}")
    df = pd.read_parquet(site_data_path)
    
    # Extract feature columns
    max_min_cols = [col for col in df.columns if '_max' in col or '_min' in col]
    required_cols = ['hospitalization_id', 'disposition', 'nth_hour'] + max_min_cols
    df_filtered = df[required_cols]
    
    # Prepare sequences
    sequences, targets = prepare_sequences(df_filtered, max_min_cols)
    
    # Convert to numpy arrays
    X = np.array(list(sequences.values()))
    y = np.array(list(targets.values()))
    
    logger.info(f"Site {site_id} data loaded: {X.shape[0]} samples, {X.shape[2]} features")
    return X, y, max_min_cols

def create_data_loaders(X, y, batch_size=16, test_size=0.2):
    """Create train and validation data loaders."""
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Scale features
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_val_reshaped = X_val.reshape(-1, n_features)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_val_scaled = scaler.transform(X_val_reshaped)
    
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler

def train_site_model(site_id: int, round_num: int, config: dict):
    """
    Train model for a specific site using WUPERR algorithm.
    
    Args:
        site_id: ID of the site
        round_num: Training round number
        config: Configuration dictionary
    """
    logger.info(f"Starting training for Site {site_id}, Round {round_num}")
    
    # Initialize components
    git_manager = GitModelManager()
    wuperr = SequentialWUPERR(
        regularization_lambda=config.get('regularization_lambda', 0.01),
        update_threshold=config.get('update_threshold', 0.001),
        ewc_lambda=config.get('ewc_lambda', 0.4)
    )
    
    # Pull latest model from repository
    if not git_manager.pull_latest_model():
        logger.error("Failed to pull latest model")
        return False
    
    # Load site data
    try:
        X, y, feature_cols = load_site_data(site_id)
        input_size = len(feature_cols)
    except FileNotFoundError:
        logger.error(f"Site data not found for site {site_id}")
        return False
    
    # Create data loaders
    train_loader, val_loader, scaler = create_data_loaders(
        X, y, batch_size=config.get('batch_size', 16)
    )
    
    # Initialize or load model
    model = git_manager.load_model(LSTMModel)
    if model is None:
        logger.info("No existing model found, initializing new model")
        model = LSTMModel(input_size=input_size)
        
        # Initialize weights
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
        
        model.apply(init_weights)
    
    # Set up training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Calculate class weights
    pos_weight = torch.tensor([len(y[y == 0]) / max(1, len(y[y == 1]))])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Calculate weight importance (if we have previous training)
    if git_manager.load_metadata() is not None:
        logger.info("Calculating weight importance for WUPERR...")
        # Use a subset of training data for Fisher calculation
        fisher_loader = DataLoader(
            train_loader.dataset, 
            batch_size=config.get('batch_size', 16),
            shuffle=True
        )
        wuperr.calculate_weight_importance(model, fisher_loader, device)
        
        # Load previous importance weights if available
        if os.path.exists(git_manager.importance_path):
            wuperr.load_importance_weights(git_manager.importance_path)
    
    # Training loop
    num_epochs = config.get('num_epochs', 20)
    best_val_loss = float('inf')
    patience = config.get('patience', 5)
    epochs_no_improve = 0
    
    training_history = []
    
    for epoch in range(num_epochs):
        # Training phase
        train_metrics = wuperr.train_step(model, train_loader, optimizer, criterion, device)
        
        # Validation phase
        val_metrics = wuperr.evaluate_model(model, val_loader, criterion, device)
        
        # Calculate additional metrics
        model.eval()
        y_pred_proba = []
        y_true = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                probs = torch.sigmoid(output)
                y_pred_proba.extend(probs.cpu().numpy().flatten())
                y_true.extend(target.cpu().numpy().flatten())
        
        auc_score = roc_auc_score(y_true, y_pred_proba)
        y_pred = (np.array(y_pred_proba) > 0.5).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Log metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['total_loss'],
            'train_standard_loss': train_metrics['standard_loss'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'auc': auc_score,
            'accuracy': accuracy
        }
        
        training_history.append(epoch_metrics)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {train_metrics['total_loss']:.4f}, "
                   f"Val Loss: {val_metrics['loss']:.4f}, "
                   f"Val Acc: {val_metrics['accuracy']:.4f}, "
                   f"AUC: {auc_score:.4f}")
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            epochs_no_improve = 0
            
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            logger.info(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model and metadata
    metadata = {
        'site_id': site_id,
        'round_num': round_num,
        'input_size': input_size,
        'feature_columns': feature_cols,
        'training_history': training_history,
        'final_metrics': epoch_metrics,
        'timestamp': datetime.now().isoformat(),
        'config': config
    }
    
    git_manager.save_model(model, metadata)
    
    # Save importance weights
    wuperr.save_importance_weights(git_manager.importance_path)
    
    # Update site contributions
    git_manager.update_site_contribution(site_id, round_num, epoch_metrics)
    
    # Commit and push to repository
    success = git_manager.commit_and_push_model(site_id, round_num, epoch_metrics)
    
    if success:
        logger.info(f"Successfully completed training for Site {site_id}, Round {round_num}")
    else:
        logger.error(f"Failed to commit changes for Site {site_id}, Round {round_num}")
    
    return success

def main():
    """Main function to run WUPERR sequential training."""
    parser = argparse.ArgumentParser(description='WUPERR Sequential Training')
    parser.add_argument('--site_id', type=int, required=True, help='Site ID (1-8)')
    parser.add_argument('--round_num', type=int, default=None, help='Round number (auto-detect if not provided)')
    parser.add_argument('--config', type=str, default='config_wuperr.json', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    # Try relative path first, then absolute
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), args.config)
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Determine round number if not provided
    if args.round_num is None:
        git_manager = GitModelManager()
        current_round = git_manager.get_current_round()
        last_site = git_manager.get_last_training_site()
        
        if last_site is None:
            round_num = 1
        elif last_site == 8:  # Last site completed, start new round
            round_num = current_round + 1
        else:
            round_num = current_round
    else:
        round_num = args.round_num
    
    # Validate site ID
    if args.site_id < 1 or args.site_id > config.get('num_sites', 8):
        logger.error(f"Invalid site ID: {args.site_id}")
        sys.exit(1)
    
    # Start training
    success = train_site_model(args.site_id, round_num, config)
    
    if success:
        logger.info("Training completed successfully")
        sys.exit(0)
    else:
        logger.error("Training failed")
        sys.exit(1)

if __name__ == "__main__":
    main()