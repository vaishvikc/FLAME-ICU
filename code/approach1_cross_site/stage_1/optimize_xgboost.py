#!/usr/bin/env python3
"""
XGBoost Hyperparameter Optimization for FLAME-ICU

This script uses Optuna to find optimal XGBoost parameters that balance
discrimination (AUC) and calibration quality (ECE, Brier score).

The optimization focuses on finding parameters that produce well-calibrated
probability estimates suitable for clinical decision-making.
"""

import os
import sys
import warnings
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

# Add parent directories to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
approach_dir = os.path.dirname(script_dir)
sys.path.insert(0, approach_dir)

from approach_1_utils import apply_missing_value_handling
from optimization_utils import (
    load_optimization_data,
    calculate_composite_score,
    save_optimization_results,
    print_trial_summary
)

warnings.filterwarnings('ignore')


def objective(trial, splits, feature_cols, config):
    """
    Objective function for XGBoost optimization

    Args:
        trial: Optuna trial object
        splits: Data splits dictionary
        feature_cols: List of feature column names
        config: Configuration dictionary

    Returns:
        float: Composite score (higher is better)
    """
    # Suggest hyperparameters
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 100, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 100, log=True),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),

        # Fixed parameters
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 42,
        'verbosity': 0
    }

    # Calculate class weights
    class_counts = np.bincount(splits['train']['target'])
    scale_pos_weight = class_counts[0] / class_counts[1]
    params['scale_pos_weight'] = scale_pos_weight

    try:
        # Prepare data with missing value handling
        X_train = apply_missing_value_handling(splits['train']['features'], config, 'xgboost')
        y_train = splits['train']['target'].values
        X_val = apply_missing_value_handling(splits['val']['features'], config, 'xgboost')
        y_val = splits['val']['target'].values

        # Shuffle training data for better learning
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train.iloc[shuffle_idx].reset_index(drop=True)
        y_train = y_train[shuffle_idx]

        # Train model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train with early stopping
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=20,
            verbose_eval=False
        )

        # Get predictions on validation set
        y_pred_proba = model.predict(dval)

        # Calculate composite score
        score, auc, ece, brier = calculate_composite_score(y_val, y_pred_proba)

        # Print trial summary
        print_trial_summary(trial.number, params, score, auc, ece, brier)

        return score

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return -1.0  # Return very bad score for failed trials


def main():
    """Main optimization function"""
    print("=" * 80)
    print("FLAME-ICU XGBoost Hyperparameter Optimization")
    print("=" * 80)

    # Load data
    splits, feature_cols, config = load_optimization_data()

    # Create optimization study
    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost_calibration_optimization',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )

    print(f"\nStarting optimization with {config.get('n_trials', 200)} trials...")
    print("Optimizing composite score: AUC - 0.5*ECE - 0.3*Brier")
    print("-" * 60)

    # Run optimization
    n_trials = 200  # Small number for testing
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

    # Evaluate best model on validation set
    print("\nEvaluating best model on validation set...")
    best_params = study.best_params.copy()

    # Add fixed parameters
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 42,
        'verbosity': 0
    })

    # Calculate class weights
    class_counts = np.bincount(splits['train']['target'])
    best_params['scale_pos_weight'] = class_counts[0] / class_counts[1]

    # Train final model
    X_train = apply_missing_value_handling(splits['train']['features'], config, 'xgboost')
    y_train = splits['train']['target'].values
    X_val = apply_missing_value_handling(splits['val']['features'], config, 'xgboost')
    y_val = splits['val']['target'].values

    # Shuffle training data for better learning
    np.random.seed(42)
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train.iloc[shuffle_idx].reset_index(drop=True)
    y_train = y_train[shuffle_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # Final evaluation
    y_pred_proba = final_model.predict(dval)
    final_score, final_auc, final_ece, final_brier = calculate_composite_score(y_val, y_pred_proba)

    print(f"\nFinal validation results:")
    print(f"  AUC: {final_auc:.4f}")
    print(f"  ECE: {final_ece:.4f}")
    print(f"  Brier Score: {final_brier:.4f}")
    print(f"  Composite Score: {final_score:.4f}")

    # Save results
    save_optimization_results(study, 'xgboost', feature_cols)

    print(f"\n✅ XGBoost optimization completed!")
    print(f"Use the saved parameters to update approach_1_config.json")


if __name__ == "__main__":
    main()