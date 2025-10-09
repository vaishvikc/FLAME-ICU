"""
Utility functions for Approach 2: Transfer Learning with Main Model Initialization
Author: FLAME-ICU Team
Description: Helper functions for loading base models and fine-tuning
"""

import os
import sys
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
from datetime import datetime

# Add approach1 to path to import utilities
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)
approach1_dir = os.path.join(code_dir, 'approach1_cross_site')
sys.path.insert(0, approach1_dir)

from approach_1_utils import (
    ICUMortalityNN,
    FocalLoss,
    apply_missing_value_handling,
    prepare_features
)


def load_config(config_path=None):
    """Load approach 2 configuration"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'approach_2_config.json')

    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_site_config():
    """Load site configuration from clif_config.json"""
    script_dir = os.path.dirname(os.path.dirname(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'clif_config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Site configuration not found: {config_path}")

    with open(config_path, 'r') as f:
        site_config = json.load(f)

    site_name = site_config.get('site', 'unknown')
    return site_name


def load_and_preprocess_data(config):
    """
    Load consolidated features and preprocess for transfer learning

    Returns:
        dict: Contains train, val, test splits with features and targets
        list: Feature names
    """
    print("Loading consolidated features...")

    # Get absolute path to data
    script_dir = os.path.dirname(__file__)
    code_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(code_dir)
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


def load_base_models(config):
    """
    Load pre-trained models from Approach 1

    Returns:
        tuple: (xgb_model, nn_model, base_feature_names)
    """
    print("Loading pre-trained base models from Approach 1...")

    # Get path to base models
    script_dir = os.path.dirname(__file__)
    code_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(code_dir)
    base_models_path = os.path.join(project_root, config['transfer_learning']['base_models_path'])

    # Load XGBoost model
    xgb_dir = os.path.join(base_models_path, 'xgboost')
    xgb_model_path = os.path.join(xgb_dir, 'xgb_model.json')
    if not os.path.exists(xgb_model_path):
        raise FileNotFoundError(f"Base XGBoost model not found: {xgb_model_path}")

    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_model_path)
    print(f"✅ Loaded XGBoost base model from: {xgb_model_path}")

    # Load XGBoost feature names
    xgb_features_path = os.path.join(xgb_dir, 'feature_names.pkl')
    with open(xgb_features_path, 'rb') as f:
        xgb_feature_names = pickle.load(f)

    # Load Neural Network model
    nn_dir = os.path.join(base_models_path, 'nn')
    nn_model_path = os.path.join(nn_dir, 'nn_model.pth')
    if not os.path.exists(nn_model_path):
        raise FileNotFoundError(f"Base Neural Network model not found: {nn_model_path}")

    # Load NN config to reconstruct architecture
    nn_config_path = os.path.join(nn_dir, 'model_config.json')
    with open(nn_config_path, 'r') as f:
        base_config = json.load(f)

    # Reconstruct NN architecture
    input_size = len(xgb_feature_names)
    nn_model = ICUMortalityNN(
        input_size=input_size,
        hidden_sizes=base_config['nn_params']['hidden_sizes'],
        dropout_rate=base_config['nn_params']['dropout_rate'],
        activation=base_config['nn_params']['activation'],
        batch_norm=base_config['nn_params']['batch_norm']
    )

    # Load pre-trained weights
    nn_model.load_state_dict(torch.load(nn_model_path, map_location='cpu'))
    print(f"✅ Loaded Neural Network base model from: {nn_model_path}")

    return xgb_model, nn_model, xgb_feature_names


def fine_tune_xgboost(base_model, splits, config, feature_names):
    """
    Fine-tune XGBoost model using transfer learning

    Args:
        base_model: Pre-trained XGBoost Booster
        splits: Data splits dictionary
        config: Configuration dictionary
        feature_names: List of feature names

    Returns:
        Fine-tuned XGBoost model and evaluation results
    """
    print("Fine-tuning XGBoost model with transfer learning...")

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

    # Apply learning rate multiplier for transfer learning
    lr_multiplier = config['transfer_learning']['learning_rate_multiplier']
    params['eta'] = params['eta'] * lr_multiplier
    print(f"Reduced learning rate to: {params['eta']:.6f} (multiplier: {lr_multiplier})")

    # Remove non-XGBoost parameters
    params.pop('use_class_weights', None)
    num_rounds = params.pop('num_rounds')
    early_stopping_rounds = params.pop('early_stopping_rounds')

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Fine-tune model (continue training from base model)
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_rounds,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=20,
        xgb_model=base_model  # Continue from base model
    )

    print(f"Best iteration: {model.best_iteration}")
    return model, evals_result


def fine_tune_nn(base_model, splits, config, feature_names):
    """
    Fine-tune Neural Network model using transfer learning

    Args:
        base_model: Pre-trained Neural Network
        splits: Data splits dictionary
        config: Configuration dictionary
        feature_names: List of feature names

    Returns:
        Fine-tuned model and training history
    """
    print("Fine-tuning Neural Network model with transfer learning...")

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

    # Use pre-trained model (already loaded with base weights)
    model = base_model

    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Setup loss function
    if config['nn_params'].get('use_focal_loss', False):
        focal_alpha = config['nn_params'].get('focal_alpha', 0.75)
        focal_gamma = config['nn_params'].get('focal_gamma', 2.0)
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        print(f"Using Focal Loss with alpha: {focal_alpha}, gamma: {focal_gamma}")
    elif config['nn_params'].get('use_class_weights', True):
        class_counts = np.bincount(splits['train']['target'])
        sqrt_inv_freq = np.sqrt(len(splits['train']['target']) / (2 * class_counts))
        pos_weight = torch.FloatTensor([sqrt_inv_freq[1] / sqrt_inv_freq[0]]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using class weights with pos_weight: {pos_weight.item():.2f}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Apply learning rate multiplier for transfer learning
    lr_multiplier = config['transfer_learning']['learning_rate_multiplier']
    fine_tune_lr = config['nn_params']['learning_rate'] * lr_multiplier
    print(f"Reduced learning rate to: {fine_tune_lr:.6f} (multiplier: {lr_multiplier})")

    optimizer = optim.Adam(
        model.parameters(),
        lr=fine_tune_lr,
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

            # Apply label smoothing if configured
            if config['nn_params'].get('label_smoothing', 0.0) > 0:
                label_smooth = config['nn_params']['label_smoothing']
                batch_y_smooth = batch_y * (1 - label_smooth) + label_smooth / 2
            else:
                batch_y_smooth = batch_y

            optimizer.zero_grad()
            if isinstance(criterion, (nn.BCEWithLogitsLoss, FocalLoss)):
                outputs = model(batch_x, return_logits=True)
            else:
                outputs = model(batch_x)
            loss = criterion(outputs, batch_y_smooth)
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

                if isinstance(criterion, (nn.BCEWithLogitsLoss, FocalLoss)):
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


def evaluate_model(model, splits, config, model_type='xgboost', splits_to_eval=['val']):
    """
    Evaluate model and calculate comprehensive metrics

    Args:
        model: Trained model (XGBoost or NN)
        splits: Dictionary containing train/val/test splits
        config: Configuration dictionary
        model_type: 'xgboost' or 'nn'
        splits_to_eval: List of split names to evaluate (default: ['val'])
                        Use ['val'] during training, ['val', 'test'] for final evaluation
    """
    results = {}

    for split_name in splits_to_eval:
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
        if len(np.unique(y_true)) > 1:
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


def save_model_artifacts(model, feature_names, config, model_type, results, site_name):
    """
    Save fine-tuned model and metadata
    """
    print(f"Saving {model_type} model artifacts...")

    # Create output directory with site name
    script_dir = os.path.dirname(__file__)
    code_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(code_dir)
    model_dir = os.path.join(project_root, config['output_paths']['models_dir'], site_name, model_type)
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    if model_type == 'xgboost':
        model_path = os.path.join(model_dir, 'xgb_model.json')
        model.save_model(model_path)
    else:  # neural network
        model_path = os.path.join(model_dir, 'nn_model.pth')
        torch.save(model.state_dict(), model_path)

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

    print(f"✅ Model artifacts saved to: {model_dir}")
    return model_dir