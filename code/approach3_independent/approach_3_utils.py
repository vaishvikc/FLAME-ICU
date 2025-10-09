"""
Utility functions for FLAME-ICU Model Training and Inference
Author: FLAME-ICU Team
Description: Generic helper functions for training and inference across all approaches
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

    Supports two modes:
    - "explicit_list": Use exact feature list from config (for multi-site consistency)
    - "all_except_identifiers": Use all columns except those in exclude_columns
    """
    print("Preparing features...")

    feature_selection_mode = config['data_config'].get('feature_selection', 'all_except_identifiers')

    if feature_selection_mode == 'explicit_list':
        # Use explicit feature list from config (ensures multi-site consistency)
        feature_cols = config['data_config']['feature_columns']
        print(f"Using explicit feature list from config: {len(feature_cols)} features")

        # Check which features are available in the data
        available_features = [col for col in feature_cols if col in df.columns]
        missing_features = [col for col in feature_cols if col not in df.columns]

        if missing_features:
            print(f"⚠️  Warning: {len(missing_features)} features from config not found in data:")
            print(f"  {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")

        feature_cols = available_features
    else:
        # Use all columns except excluded ones
        exclude_cols = config['data_config']['exclude_columns']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        print(f"Using all columns except {len(exclude_cols)} excluded columns")

    print(f"Total features selected: {len(feature_cols)}")

    # Extract features
    features_df = df[feature_cols].copy()

    # Keep all features regardless of missing value percentage
    # Missing values will be handled model-specifically during training/inference

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
        X_filled = X.fillna(-1)

        # Ensure all columns are numeric (convert if possible, error if not)
        non_numeric_cols = X_filled.select_dtypes(include=['object']).columns.tolist()
        if non_numeric_cols:
            print(f"⚠️  Converting {len(non_numeric_cols)} object columns to numeric: {non_numeric_cols[:5]}...")
            for col in non_numeric_cols:
                try:
                    X_filled[col] = pd.to_numeric(X_filled[col], errors='raise')
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"Column '{col}' contains non-numeric data and cannot be converted. "
                        f"Sample values: {X_filled[col].dropna().unique()[:5].tolist()}\n"
                        f"Please check your feature list in the config and ensure the dataset was "
                        f"regenerated with boolean-to-int conversion from 02_feature_assmebly.py"
                    ) from e

        return X_filled
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

    # Shuffle training data for better learning
    print("Shuffling training data...")
    np.random.seed(config['random_seeds']['xgboost'])
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train.iloc[shuffle_idx].reset_index(drop=True)
    y_train = y_train.iloc[shuffle_idx].reset_index(drop=True)

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


def extract_xgboost_feature_importance(model, feature_names):
    """
    Extract feature importance from XGBoost model

    Returns dict with 3 importance types:
    - weight: number of times feature is used in splits
    - gain: average gain when feature is used
    - cover: average coverage when feature is used
    """
    print("Extracting XGBoost feature importance...")

    importance_types = ['weight', 'gain', 'cover']
    importance_dict = {}

    for imp_type in importance_types:
        # Get importance scores from model
        scores = model.get_score(importance_type=imp_type)

        # Map feature indices to feature names and create sorted list
        importance_list = []
        for fname, score in scores.items():
            # XGBoost may use actual feature names or generic 'f0', 'f1' format
            if fname in feature_names:
                # XGBoost is using actual feature names
                feature_name = fname
            else:
                # XGBoost is using generic 'f0', 'f1' format
                feature_idx = int(fname[1:])
                feature_name = feature_names[feature_idx]

            importance_list.append({
                'feature': feature_name,
                'importance': float(score)
            })

        # Sort by importance descending
        importance_list.sort(key=lambda x: x['importance'], reverse=True)
        importance_dict[imp_type] = importance_list

    print(f"✅ Extracted importance for {len(importance_dict['weight'])} features")
    return importance_dict


def calculate_permutation_importance(model, X, y, config, feature_names, n_repeats=5):
    """
    Calculate permutation importance for Neural Network model

    Measures AUC drop when each feature is randomly shuffled
    """
    print("Calculating permutation importance for Neural Network...")
    print(f"Using {n_repeats} repeats per feature...")

    # Get baseline performance
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    X_processed = apply_missing_value_handling(X, config, 'nn')
    X_tensor = torch.FloatTensor(X_processed.values).to(device)
    y_true = y.values

    with torch.no_grad():
        baseline_preds = model(X_tensor).cpu().numpy().flatten()
    baseline_auc = roc_auc_score(y_true, baseline_preds)

    print(f"Baseline AUC: {baseline_auc:.4f}")

    # Calculate importance for each feature
    importances = []

    for i, feature_name in enumerate(feature_names):
        feature_drops = []

        for _ in range(n_repeats):
            # Create a copy and shuffle the feature
            X_permuted = X_processed.copy()
            X_permuted.iloc[:, i] = np.random.permutation(X_permuted.iloc[:, i].values)

            # Get predictions with permuted feature
            X_perm_tensor = torch.FloatTensor(X_permuted.values).to(device)
            with torch.no_grad():
                perm_preds = model(X_perm_tensor).cpu().numpy().flatten()

            # Calculate AUC drop
            perm_auc = roc_auc_score(y_true, perm_preds)
            feature_drops.append(baseline_auc - perm_auc)

        # Average importance across repeats
        mean_importance = np.mean(feature_drops)
        std_importance = np.std(feature_drops)

        importances.append({
            'feature': feature_name,
            'importance': float(mean_importance),
            'importance_std': float(std_importance)
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(feature_names)} features...")

    # Sort by importance descending
    importances.sort(key=lambda x: x['importance'], reverse=True)

    print(f"✅ Calculated permutation importance for {len(importances)} features")
    return importances


def save_feature_importance(importance_data, model_type, config):
    """
    Save feature importance to JSON file
    """
    # Get output directory
    script_dir = os.path.dirname(os.path.dirname(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, config['output_paths']['results_dir'])
    os.makedirs(results_dir, exist_ok=True)

    # Save to JSON
    output_path = os.path.join(results_dir, f'{model_type}_feature_importance.json')
    with open(output_path, 'w') as f:
        json.dump(importance_data, f, indent=2)

    print(f"✅ Feature importance saved to: {output_path}")
    return output_path


def plot_feature_importance(importance_data, model_type, config, top_n=20):
    """
    Create horizontal bar plot of top N features

    For XGBoost: creates subplot for each importance type
    For NN: creates single plot
    """
    print(f"Creating feature importance plot (top {top_n})...")

    # Get output directory
    script_dir = os.path.dirname(os.path.dirname(__file__))
    project_root = os.path.dirname(script_dir)
    plots_dir = os.path.join(project_root, config['output_paths']['plots_dir'])
    os.makedirs(plots_dir, exist_ok=True)

    if model_type == 'xgboost':
        # XGBoost has 3 importance types
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))

        for idx, (imp_type, ax) in enumerate(zip(['weight', 'gain', 'cover'], axes)):
            # Get top N features for this importance type
            top_features = importance_data[imp_type][:top_n]

            # Reverse order for plotting (highest at top)
            features = [x['feature'] for x in reversed(top_features)]
            scores = [x['importance'] for x in reversed(top_features)]

            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            ax.barh(y_pos, scores, color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=9)
            ax.set_xlabel('Importance Score', fontsize=10)
            ax.set_title(f'XGBoost Feature Importance ({imp_type.capitalize()})', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

    else:  # Neural Network
        # NN has single importance list
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Get top N features
        top_features = importance_data[:top_n]

        # Reverse order for plotting (highest at top)
        features = [x['feature'] for x in reversed(top_features)]
        scores = [x['importance'] for x in reversed(top_features)]
        errors = [x['importance_std'] for x in reversed(top_features)]

        # Create horizontal bar plot with error bars
        y_pos = np.arange(len(features))
        ax.barh(y_pos, scores, xerr=errors, color='coral', capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Importance (AUC Drop)', fontsize=11)
        ax.set_title(f'Neural Network Feature Importance (Permutation)', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

    # Save plot
    output_path = os.path.join(plots_dir, f'{model_type}_importance_top{top_n}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Feature importance plot saved to: {output_path}")
    return output_path


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

    # Setup loss function
    if config['nn_params'].get('use_focal_loss', False):
        # Use Focal Loss for better class imbalance handling
        focal_alpha = config['nn_params'].get('focal_alpha', 0.75)
        focal_gamma = config['nn_params'].get('focal_gamma', 2.0)
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        print(f"Using Focal Loss with alpha: {focal_alpha}, gamma: {focal_gamma}")
    elif config['nn_params'].get('use_class_weights', True):
        # Calculate improved class weights using square root
        class_counts = np.bincount(splits['train']['target'])
        # Use sqrt of inverse frequency for more balanced weighting
        sqrt_inv_freq = np.sqrt(len(splits['train']['target']) / (2 * class_counts))
        pos_weight = torch.FloatTensor([sqrt_inv_freq[1] / sqrt_inv_freq[0]]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using improved class weights with pos_weight: {pos_weight.item():.2f}")
    else:
        criterion = nn.BCEWithLogitsLoss()

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

            # Apply label smoothing if configured
            if config['nn_params'].get('label_smoothing', 0.0) > 0:
                label_smooth = config['nn_params']['label_smoothing']
                # Smooth labels: 0 -> label_smooth/2, 1 -> 1 - label_smooth/2
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


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Formula: FL(p_t) = -α(1-p_t)^γ log(p_t)
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Calculate standard BCE loss (without reduction)
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate p_t (probability of correct class)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Calculate alpha_t (class-specific weight)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Apply focal loss formula
        F_loss = alpha_t * (1 - p_t) ** self.gamma * BCE_loss

        return F_loss.mean()


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