#!/usr/bin/env python3
"""
Shared utilities for hyperparameter optimization of XGBoost and Neural Network models.

This module provides common functions for calculating calibration metrics,
loading data, and evaluating models during hyperparameter optimization.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

# Add parent directories to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
approach_dir = os.path.dirname(script_dir)  # approach1_cross_site/
code_dir = os.path.dirname(approach_dir)    # code/
sys.path.insert(0, approach_dir)

from approach_1_utils import load_config, load_and_preprocess_data


def calculate_ece(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE)

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        float: Expected Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def calculate_composite_score(y_true, y_prob, auc_weight=1.0, ece_weight=0.5, brier_weight=0.3):
    """
    Calculate composite score for optimization that balances discrimination and calibration

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        auc_weight: Weight for AUC (discrimination)
        ece_weight: Weight for ECE (calibration error)
        brier_weight: Weight for Brier score (overall accuracy)

    Returns:
        float: Composite score (higher is better)
    """
    auc = roc_auc_score(y_true, y_prob)
    ece = calculate_ece(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # Higher AUC is better, lower ECE and Brier are better
    score = auc_weight * auc - ece_weight * ece - brier_weight * brier

    return score, auc, ece, brier


def load_optimization_data():
    """
    Load and prepare data for optimization
    IMPORTANT: Only loads train and val data - test is excluded to prevent cheating

    Returns:
        dict: Dictionary containing ONLY train and val splits
    """
    print("Loading data for optimization...")

    # Load configuration
    config_path = os.path.join(approach_dir, 'approach_1_config.json')
    config = load_config(config_path)

    # Load and preprocess data (this already returns splits!)
    all_splits, feature_names = load_and_preprocess_data(config)

    # CRITICAL: Remove test split to prevent data leakage/cheating
    optimization_splits = {
        'train': all_splits['train'],
        'val': all_splits['val']
        # NO TEST DATA - this prevents cheating!
    }

    # Feature names are already the clean feature columns
    feature_cols = feature_names

    print(f"Data loaded successfully:")
    print(f"  Train: {len(optimization_splits['train']['features'])} samples")
    print(f"  Val: {len(optimization_splits['val']['features'])} samples")
    print(f"  Test: EXCLUDED (preventing data leakage)")
    print(f"  Features: {len(feature_cols)}")

    # Safety verification
    verify_no_test_contamination(optimization_splits)

    return optimization_splits, feature_cols, config


def save_optimization_results(study, model_type, feature_cols):
    """
    Save optimization results and best parameters

    Args:
        study: Optuna study object
        model_type: str, either 'xgboost' or 'nn'
        feature_cols: list of feature column names
    """
    results_dir = os.path.join(approach_dir, 'optimization_results')
    os.makedirs(results_dir, exist_ok=True)

    # Save best parameters
    best_params = study.best_params.copy()

    # Add metadata
    optimization_results = {
        'model_type': model_type,
        'optimization_timestamp': datetime.now().isoformat(),
        'n_trials': len(study.trials),
        'best_score': study.best_value,
        'best_params': best_params,
        'feature_count': len(feature_cols),
        'study_stats': {
            'n_complete_trials': len([t for t in study.trials if t.state.name == 'COMPLETE']),
            'n_failed_trials': len([t for t in study.trials if t.state.name == 'FAIL']),
            'n_pruned_trials': len([t for t in study.trials if t.state.name == 'PRUNED'])
        }
    }

    # Save to JSON
    results_path = os.path.join(results_dir, f'best_{model_type}_params.json')
    with open(results_path, 'w') as f:
        json.dump(optimization_results, f, indent=2)

    print(f"✅ Optimization results saved to: {results_path}")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best parameters: {best_params}")

    return results_path


def print_trial_summary(trial_number, params, score, auc, ece, brier):
    """
    Print summary of trial results

    Args:
        trial_number: int, trial number
        params: dict, trial parameters
        score: float, composite score
        auc: float, AUC score
        ece: float, ECE score
        brier: float, Brier score
    """
    print(f"Trial {trial_number:3d}: Score={score:.4f} | AUC={auc:.4f} | ECE={ece:.4f} | Brier={brier:.4f}")


def verify_no_test_contamination(splits):
    """
    Verify that test data is not accessible during optimization
    This is a critical safety check to prevent data leakage/cheating

    Args:
        splits: Dictionary of data splits

    Raises:
        AssertionError: If test data is found in splits
    """
    assert 'test' not in splits, "ERROR: Test data found in optimization splits! This is cheating!"
    assert len(splits) == 2, f"ERROR: Expected 2 splits (train, val), found {len(splits)}: {list(splits.keys())}"
    assert 'train' in splits, "ERROR: Train split missing from optimization data"
    assert 'val' in splits, "ERROR: Validation split missing from optimization data"

    print("✓ Safety check passed: No test data contamination")
    print("✓ Optimization is honest: using only train and validation data")
    return True


def create_optimization_plots(study, model_type):
    """
    Create optimization plots (placeholder for future implementation)

    Args:
        study: Optuna study object
        model_type: str, either 'xgboost' or 'nn'
    """
    # TODO: Implement plotting functionality
    # - Optimization history
    # - Parameter importance
    # - Parameter relationships
    pass