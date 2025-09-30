#!/usr/bin/env python3
"""
Neural Network Hyperparameter Optimization for FLAME-ICU

This script uses Optuna to find optimal Neural Network parameters that balance
discrimination (AUC) and calibration quality (ECE, Brier score).

The optimization explores different architectures, regularization strategies,
and loss functions to find the best configuration for clinical decision-making.
"""

import os
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.metrics import roc_auc_score

# Add parent directories to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
approach_dir = os.path.dirname(script_dir)
sys.path.insert(0, approach_dir)

from approach_1_utils import apply_missing_value_handling, ICUMortalityNN, FocalLoss
from optimization_utils import (
    load_optimization_data,
    calculate_composite_score,
    save_optimization_results,
    print_trial_summary
)

warnings.filterwarnings('ignore')


def create_model(trial, input_size):
    """
    Create a neural network model based on trial suggestions

    Args:
        trial: Optuna trial object
        input_size: Number of input features

    Returns:
        nn.Module: Neural network model
    """
    # Suggest architecture
    architecture = trial.suggest_categorical('architecture', ['small', 'medium', 'large', 'xlarge'])

    arch_configs = {
        'small': [32, 16],
        'medium': [64, 32],
        'large': [128, 64, 32],
        'xlarge': [256, 128, 64]
    }

    hidden_sizes = arch_configs[architecture]
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.6)

    return ICUMortalityNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        activation='relu',
        batch_norm=True
    )


def create_loss_function(trial):
    """
    Create loss function based on trial suggestions

    Args:
        trial: Optuna trial object

    Returns:
        Loss function and whether it needs class weights
    """
    loss_type = trial.suggest_categorical('loss_type', ['bce', 'focal', 'weighted_bce'])

    if loss_type == 'focal':
        alpha = trial.suggest_float('focal_alpha', 0.5, 0.95)
        gamma = trial.suggest_float('focal_gamma', 0.5, 3.0)
        return FocalLoss(alpha=alpha, gamma=gamma), False

    elif loss_type == 'weighted_bce':
        return None, True  # Will be created with pos_weight

    else:  # standard BCE
        return nn.BCEWithLogitsLoss(), False


def objective(trial, splits, feature_cols, config):
    """
    Objective function for Neural Network optimization

    Args:
        trial: Optuna trial object
        splits: Data splits dictionary
        feature_cols: List of feature column names
        config: Configuration dictionary

    Returns:
        float: Composite score (higher is better)
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
        weight_decay = trial.suggest_float('weight_decay', 0.0001, 0.1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2)

        # Prepare data
        X_train = apply_missing_value_handling(splits['train']['features'], config, 'nn')
        X_val = apply_missing_value_handling(splits['val']['features'], config, 'nn')

        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(splits['train']['target'].values).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.FloatTensor(splits['val']['target'].values).unsqueeze(1)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        input_size = X_train_tensor.shape[1]
        model = create_model(trial, input_size)
        model.to(device)

        # Create loss function
        criterion, use_pos_weight = create_loss_function(trial)

        if use_pos_weight:
            # Calculate class weights
            class_counts = np.bincount(splits['train']['target'])
            pos_weight = torch.FloatTensor([class_counts[0] / class_counts[1]]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if criterion is None:
            raise ValueError("Loss function not properly initialized")

        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Training loop
        max_epochs = 50  # Reduced for optimization speed
        patience = 10
        best_val_auc = 0
        patience_counter = 0

        for epoch in range(max_epochs):
            # Training phase
            model.train()
            train_loss = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # Apply label smoothing
                if label_smoothing > 0:
                    batch_y_smooth = batch_y * (1 - label_smoothing) + label_smoothing / 2
                else:
                    batch_y_smooth = batch_y

                optimizer.zero_grad()

                if isinstance(criterion, (nn.BCEWithLogitsLoss, FocalLoss)):
                    outputs = model(batch_x, return_logits=True)
                else:
                    outputs = model(batch_x)

                loss = criterion(outputs, batch_y_smooth)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation phase
            model.eval()
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    if isinstance(criterion, (nn.BCEWithLogitsLoss, FocalLoss)):
                        outputs_logits = model(batch_x, return_logits=True)
                        outputs = torch.sigmoid(outputs_logits)
                    else:
                        outputs = model(batch_x)

                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())

            # Calculate validation AUC
            val_auc = roc_auc_score(val_targets, val_preds)

            # Early stopping check
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            # Report intermediate result for pruning
            trial.report(val_auc, epoch)

            # Handle pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Final evaluation
        model.eval()
        final_preds = []
        final_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                if isinstance(criterion, (nn.BCEWithLogitsLoss, FocalLoss)):
                    outputs_logits = model(batch_x, return_logits=True)
                    outputs = torch.sigmoid(outputs_logits)
                else:
                    outputs = model(batch_x)

                final_preds.extend(outputs.cpu().numpy())
                final_targets.extend(batch_y.cpu().numpy())

        # Calculate composite score
        score, auc, ece, brier = calculate_composite_score(
            np.array(final_targets).flatten(),
            np.array(final_preds).flatten()
        )

        # Print trial summary
        params_summary = {
            'architecture': trial.params.get('architecture'),
            'learning_rate': trial.params.get('learning_rate'),
            'loss_type': trial.params.get('loss_type')
        }
        print_trial_summary(trial.number, params_summary, score, auc, ece, brier)

        return score

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return -1.0


def main():
    """Main optimization function"""
    print("=" * 80)
    print("FLAME-ICU Neural Network Hyperparameter Optimization")
    print("=" * 80)

    # Load data
    splits, feature_cols, config = load_optimization_data()

    # Create optimization study
    study = optuna.create_study(
        direction='maximize',
        study_name='nn_calibration_optimization',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    )

    print(f"\nStarting optimization with 100 trials...")
    print("Optimizing composite score: AUC - 0.5*ECE - 0.3*Brier")
    print("-" * 60)

    # Run optimization
    n_trials = 100  # Very small number for testing
    study.optimize(
        lambda trial: objective(trial, splits, feature_cols, config),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETED")
    print("=" * 80)

    # Print best results
    print(f"Best score: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results
    save_optimization_results(study, 'nn', feature_cols)

    print(f"\n✅ Neural Network optimization completed!")
    print(f"Use the saved parameters to update approach_1_config.json")


if __name__ == "__main__":
    main()