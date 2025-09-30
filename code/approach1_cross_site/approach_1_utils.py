"""
Utility functions for Approach 1: Cross-Site Model Validation
Author: FLAME-ICU Team
Description: Helper functions for training and inference in Approach 1
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, brier_score_loss, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_config(config_path=None):
    """Load approach 1 configuration"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'approach_1_config.json')

    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_and_preprocess_data(config):
    """
    Load consolidated features and preprocess for modeling

    Returns:
        dict: Contains train, val, test splits with features and targets
    """
    print("Loading consolidated features...")

    # Get absolute path to data
    script_dir = os.path.dirname(os.path.dirname(__file__))  # Go up from models/ to code/
    project_root = os.path.dirname(script_dir)  # Go up from code/ to project root
    data_path = os.path.join(project_root, config['data_config']['consolidated_features_path'])

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_parquet(data_path)
    print(f"Loaded data shape: {df.shape}")

    # Check split distribution
    print("Split distribution:")
    print(df['split_type'].value_counts())

    # Check target distribution
    print("Target distribution:")
    print(df[config['data_config']['target_column']].value_counts())

    # Prepare features
    features_df, feature_names = prepare_features(df, config)

    # Split data
    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_mask = df['split_type'] == split_name
        splits[split_name] = {
            'features': features_df[split_mask].reset_index(drop=True),
            'target': df.loc[split_mask, config['data_config']['target_column']].reset_index(drop=True),
            'hospitalization_id': df.loc[split_mask, 'hospitalization_id'].reset_index(drop=True)
        }
        print(f"{split_name} split: {len(splits[split_name]['features'])} samples")

    return splits, feature_names


def prepare_features(df, config):
    """
    Prepare features by selecting columns and handling missing values
    """
    print("Preparing features...")

    # Get feature columns (exclude specified columns)
    exclude_cols = config['data_config']['exclude_columns']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"Total features selected: {len(feature_cols)}")

    # Extract features
    features_df = df[feature_cols].copy()

    # Handle columns with excessive missing values
    missing_pct = features_df.isnull().sum() / len(features_df)
    high_missing_cols = missing_pct[missing_pct > config['preprocessing']['missing_threshold']].index.tolist()

    if high_missing_cols:
        print(f"Dropping {len(high_missing_cols)} columns with >{config['preprocessing']['missing_threshold']*100}% missing values")
        features_df = features_df.drop(columns=high_missing_cols)

    # Note: Model-specific missing value handling will be applied at runtime:
    # - XGBoost: no imputation needed (handles NaN natively)
    # - Neural Network: will fill with -1 during training/inference

    print(f"Final feature shape: {features_df.shape}")
    return features_df, features_df.columns.tolist()


def apply_missing_value_handling(X, config, model_type):
    """
    Apply model-specific missing value handling

    Args:
        X: Feature dataframe
        config: Configuration dictionary
        model_type: 'xgboost' or 'nn'

    Returns:
        Processed feature dataframe
    """
    missing_strategy = config['preprocessing']['handle_missing'][model_type]

    if missing_strategy == 'none':
        # XGBoost - no imputation needed, handles NaN natively
        print(f"Using {model_type} native missing value handling")
        return X
    elif missing_strategy == 'fill_negative_one':
        # Neural Network - fill with -1
        print(f"Filling missing values with -1 for {model_type}")
        return X.fillna(-1)
    else:
        raise ValueError(f"Unknown missing value strategy: {missing_strategy}")



def train_xgboost_approach1(splits, config, feature_names):
    """
    Train XGBoost model for Approach 1
    """
    print("Training XGBoost model...")

    # Prepare data with model-specific missing value handling
    X_train = apply_missing_value_handling(splits['train']['features'], config, 'xgboost')
    y_train = splits['train']['target']
    X_val = apply_missing_value_handling(splits['val']['features'], config, 'xgboost')
    y_val = splits['val']['target']

    # Calculate class weights if enabled
    params = config['xgboost_params'].copy()
    if params.get('use_class_weights', True):
        scale_pos_weight = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))
        params['scale_pos_weight'] = scale_pos_weight
        print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

    # Remove non-XGBoost parameters
    params.pop('use_class_weights', None)
    num_rounds = params.pop('num_rounds')
    early_stopping_rounds = params.pop('early_stopping_rounds')

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Train model
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_rounds,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=50
    )

    print(f"Best iteration: {model.best_iteration}")
    return model, evals_result


def train_nn_approach1(splits, config, feature_names):
    """
    Train Neural Network model for Approach 1
    """
    print("Training Neural Network model...")

    # Set random seed
    torch.manual_seed(config['nn_params']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['nn_params']['seed'])

    # Prepare data with model-specific missing value handling
    X_train_processed = apply_missing_value_handling(splits['train']['features'], config, 'nn')
    X_val_processed = apply_missing_value_handling(splits['val']['features'], config, 'nn')

    X_train = torch.FloatTensor(X_train_processed.values)
    y_train = torch.FloatTensor(splits['train']['target'].values).unsqueeze(1)
    X_val = torch.FloatTensor(X_val_processed.values)
    y_val = torch.FloatTensor(splits['val']['target'].values).unsqueeze(1)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    batch_size = config['nn_params']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_size = X_train.shape[1]
    hidden_sizes = config['nn_params']['hidden_sizes']
    dropout_rate = config['nn_params']['dropout_rate']
    activation = config['nn_params']['activation']
    batch_norm = config['nn_params']['batch_norm']

    model = ICUMortalityNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        activation=activation,
        batch_norm=batch_norm
    )

    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Calculate class weights if enabled
    if config['nn_params'].get('use_class_weights', True):
        class_counts = np.bincount(splits['train']['target'])
        class_weights = len(splits['train']['target']) / (2 * class_counts)
        pos_weight = torch.FloatTensor([class_weights[1] / class_weights[0]]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using class weights with pos_weight: {pos_weight.item():.2f}")
    else:
        criterion = nn.BCELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['nn_params']['learning_rate'],
        weight_decay=config['nn_params']['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

    # Training loop
    best_val_auc = 0
    patience_counter = 0
    epochs = config['nn_params']['epochs']
    patience = config['nn_params']['early_stopping_patience']
    gradient_clip = config['nn_params']['gradient_clip']

    training_history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                outputs = model(batch_x, return_logits=True)
            else:
                outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    outputs_logits = model(batch_x, return_logits=True)
                    outputs = torch.sigmoid(outputs_logits)
                    loss = criterion(outputs_logits, batch_y)
                else:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_targets, val_preds)

        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_auc'].append(val_auc)

        scheduler.step(val_auc)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(best_model_state)
    print(f"Best validation AUC: {best_val_auc:.4f}")

    return model, training_history


def evaluate_model(model, splits, config, model_type='xgboost'):
    """
    Evaluate model and calculate comprehensive metrics
    """
    results = {}

    for split_name in ['val', 'test']:
        print(f"Evaluating on {split_name} set...")

        X_raw = splits[split_name]['features']
        y_true = splits[split_name]['target'].values

        # Apply model-specific missing value handling
        X = apply_missing_value_handling(X_raw, config, model_type)

        # Get predictions
        if model_type == 'xgboost':
            dtest = xgb.DMatrix(X)
            y_pred_proba = model.predict(dtest)
        else:  # neural network
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X.values)
                if torch.cuda.is_available():
                    X_tensor = X_tensor.cuda()
                    model = model.cuda()
                y_pred_proba = model(X_tensor).cpu().numpy().flatten()

        # Calculate metrics
        threshold = config['evaluation_config']['threshold']
        y_pred = (y_pred_proba > threshold).astype(int)

        metrics = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'pr_auc': average_precision_score(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba)
        }

        # Calculate calibration error
        n_bins = config['evaluation_config']['calibration_bins']
        if len(np.unique(y_true)) > 1:  # Only if both classes present
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
                )
                calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                metrics['calibration_error'] = calibration_error
            except:
                metrics['calibration_error'] = np.nan
        else:
            metrics['calibration_error'] = np.nan

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        results[split_name] = {
            'metrics': metrics
        }

        # Print key metrics
        print(f"{split_name.upper()} Results:")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")

    return results


def save_model_artifacts(model, transformer, feature_names, config, model_type, results=None):
    """
    Save model, transformer, and metadata
    """
    print(f"Saving {model_type} model artifacts...")

    # Create output directory
    script_dir = os.path.dirname(os.path.dirname(__file__))
    project_root = os.path.dirname(script_dir)
    model_dir = os.path.join(project_root, config['output_paths']['models_dir'], model_type)
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    if model_type == 'xgboost':
        model_path = os.path.join(model_dir, 'xgb_model.json')
        model.save_model(model_path)
    else:  # neural network
        model_path = os.path.join(model_dir, 'nn_model.pth')
        torch.save(model.state_dict(), model_path)

    # Save transformer
    if transformer is not None:
        transformer_path = os.path.join(model_dir, 'transformer.pkl')
        with open(transformer_path, 'wb') as f:
            pickle.dump(transformer, f)

    # Save feature names
    feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)

    # Save config
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Save results if provided
    if results is not None:
        results_path = os.path.join(model_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"Model artifacts saved to: {model_dir}")
    return model_dir


class ICUMortalityNN(nn.Module):
    """
    Neural Network for ICU mortality prediction (simplified version)
    """
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.3, activation='relu', batch_norm=True):
        super(ICUMortalityNN, self).__init__()

        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if batch_norm else None
        self.dropout_layers = nn.ModuleList()

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        # Build layers
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            if batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_size))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(prev_size, 1)

    def forward(self, x, return_logits=False):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.bn_layers:
                x = self.bn_layers[i](x)
            x = self.activation(x)
            x = self.dropout_layers[i](x)

        x = self.output_layer(x)

        if return_logits:
            return x
        else:
            x = torch.sigmoid(x)
            x = torch.clamp(x, min=1e-7, max=1-1e-7)
            return x