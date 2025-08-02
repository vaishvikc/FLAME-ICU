import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import json
import pickle


class ICUMortalityDataset(Dataset):
    """PyTorch Dataset for ICU mortality prediction"""
    
    def __init__(self, X, y):
        """
        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
            y: Target labels (numpy array or pandas Series)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_config():
    """Load configuration from config file"""
    # Try models/nn/config.json first
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'nn', 'config.json')
    if not os.path.exists(config_path):
        # Fallback to preprocessing directory
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_preprocessing_config():
    """Load preprocessing configuration to get site name"""
    # Try top-level config_demo.json first (new location)
    preprocessing_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'config_demo.json'
    )
    
    if not os.path.exists(preprocessing_config_path):
        # Fallback to old location
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


def load_data():
    """
    Load pre-split data from XGBoost pipeline.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_columns, site_name
    """
    config = load_config()
    preprocessing_config = load_preprocessing_config()
    site_name = preprocessing_config.get('site', 'unknown')
    
    # Load pre-split data files
    train_file = config['data_split']['train_file']
    test_file = config['data_split']['test_file']
    
    # Convert relative paths to absolute paths
    # Get project root directory (2 levels up from preprocessing/)
    script_dir = os.path.dirname(os.path.abspath(__file__))  # code/preprocessing
    code_dir = os.path.dirname(script_dir)  # code
    project_root = os.path.dirname(code_dir)  # project root
    
    if not os.path.isabs(train_file):
        # Handle paths that are relative to models/nn/ directory
        if train_file.startswith('../../../'):
            # Path is relative to models/nn/, convert to absolute from project root
            train_file = os.path.join(project_root, train_file.replace('../../../', ''))
        else:
            train_file = os.path.abspath(os.path.join(script_dir, train_file))
    
    if not os.path.isabs(test_file):
        # Handle paths that are relative to models/nn/ directory
        if test_file.startswith('../../../'):
            # Path is relative to models/nn/, convert to absolute from project root
            test_file = os.path.join(project_root, test_file.replace('../../../', ''))
        else:
            test_file = os.path.abspath(os.path.join(script_dir, test_file))
    
    print(f"Loading training data from: {train_file}")
    train_df = pd.read_parquet(train_file)
    print(f"Training data shape: {train_df.shape}")
    
    print(f"Loading test data from: {test_file}")
    test_df = pd.read_parquet(test_file)
    print(f"Test data shape: {test_df.shape}")
    
    # Prepare features and targets
    feature_columns = [col for col in train_df.columns if col not in ['hospitalization_id', 'disposition']]
    
    X_train = train_df[feature_columns]
    y_train = train_df['disposition']
    X_test = test_df[feature_columns]
    y_test = test_df['disposition']
    
    print(f"Training set: {X_train.shape} features, {len(y_train)} patients")
    print(f"Test set: {X_test.shape} features, {len(y_test)} patients")
    print(f"Training mortality rate: {y_train.mean():.3f}")
    print(f"Test mortality rate: {y_test.mean():.3f}")
    
    # Handle missing values - replace with median for neural networks
    print("Handling missing values...")
    print(f"NaN values in training set before imputation: {X_train.isna().sum().sum()}")
    print(f"NaN values in test set before imputation: {X_test.isna().sum().sum()}")
    
    # Calculate medians, but handle columns that are entirely NaN
    train_medians = X_train.median()
    # For columns that are entirely NaN, use 0 as fallback
    train_medians = train_medians.fillna(0)
    
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)
    
    print(f"NaN values in training set after imputation: {X_train.isna().sum().sum()}")
    print(f"NaN values in test set after imputation: {X_test.isna().sum().sum()}")
    
    # Check for infinite values
    print(f"Infinite values in training set: {np.isinf(X_train.values).sum()}")
    print(f"Infinite values in test set: {np.isinf(X_test.values).sum()}")
    
    # Replace infinite values with large finite values
    X_train = X_train.replace([np.inf, -np.inf], [1e10, -1e10])
    X_test = X_test.replace([np.inf, -np.inf], [1e10, -1e10])
    
    # Final check - replace any remaining NaN with 0
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    print(f"Final NaN check - Training: {X_train.isna().sum().sum()}, Test: {X_test.isna().sum().sum()}")
    
    return X_train, X_test, y_train, y_test, feature_columns, site_name


def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=64, 
                       use_scaling=True, scaler=None):
    """
    Create PyTorch DataLoaders for training and testing.
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Target labels
        batch_size: Batch size for training
        use_scaling: Whether to apply StandardScaler
        scaler: Pre-fitted scaler (optional)
        
    Returns:
        train_loader, test_loader, scaler
    """
    # Apply scaling if requested
    if use_scaling:
        if scaler is None:
            print("Applying StandardScaler to features...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values if hasattr(X_train, 'values') else X_train
        X_test_scaled = X_test.values if hasattr(X_test, 'values') else X_test
        scaler = None
    
    # Create datasets
    train_dataset = ICUMortalityDataset(X_train_scaled, y_train)
    test_dataset = ICUMortalityDataset(X_test_scaled, y_test)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Created DataLoaders with batch size {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, test_loader, scaler


def save_preprocessing_artifacts(scaler, feature_columns, output_config, site_name):
    """Save scaler and feature columns for inference"""
    # Update paths to use site-specific directory structure
    scaler_path = output_config['scaler_path'].replace('/{SITE_NAME}/', f'/{site_name}/')
    feature_cols_path = output_config['feature_cols_path'].replace('/{SITE_NAME}/', f'/{site_name}/')
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    # Save scaler
    if scaler is not None:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Saved scaler to: {scaler_path}")
    
    # Save feature columns
    with open(feature_cols_path, 'wb') as f:
        pickle.dump(feature_columns, f)
    print(f"Saved feature columns to: {feature_cols_path}")