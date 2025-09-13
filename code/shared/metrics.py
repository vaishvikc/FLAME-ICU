#!/usr/bin/env python3
"""
Evaluation metrics utilities for FLAME-ICU federated learning.
Provides standardized metrics calculation across all approaches.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_standard_metrics(y_true: np.ndarray, 
                             y_pred_proba: np.ndarray, 
                             threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate standard classification metrics.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'brier_score': brier_score_loss(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba),
        'threshold': threshold
    }
    
    # Calculate specificity manually
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['sensitivity'] = metrics['recall']  # Recall is the same as sensitivity
    
    # Additional metrics
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    metrics['ppv'] = metrics['precision']  # Positive Predictive Value is precision
    
    return metrics

def calculate_calibration_metrics(y_true: np.ndarray, 
                                y_pred_proba: np.ndarray, 
                                n_bins: int = 10) -> Dict[str, Any]:
    """
    Calculate model calibration metrics.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration curve
    
    Returns:
        Dictionary with calibration metrics
    """
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
    )
    
    # Calculate Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return {
        'expected_calibration_error': ece,
        'fraction_of_positives': fraction_of_positives.tolist(),
        'mean_predicted_value': mean_predicted_value.tolist(),
        'n_bins': n_bins
    }

def calculate_threshold_metrics(y_true: np.ndarray, 
                              y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """
    Calculate metrics across different thresholds.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        Dictionary with threshold analysis
    """
    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
    
    # Calculate Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Find optimal threshold (Youden's index)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = roc_thresholds[optimal_idx]
    
    return {
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresholds.tolist()
        },
        'pr_curve': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist()
        },
        'optimal_threshold': optimal_threshold,
        'optimal_threshold_metrics': calculate_standard_metrics(y_true, y_pred_proba, optimal_threshold)
    }

def compare_models(models_results: Dict[str, Dict[str, float]], 
                  metric: str = 'auc_roc') -> Dict[str, Any]:
    """
    Compare multiple models based on a specific metric.
    
    Args:
        models_results: Dictionary with model names as keys and metrics as values
        metric: Metric to use for comparison
    
    Returns:
        Dictionary with comparison results
    """
    if not models_results:
        return {'error': 'No models to compare'}
    
    # Extract metric values
    metric_values = {}
    for model_name, metrics in models_results.items():
        if metric in metrics:
            metric_values[model_name] = metrics[metric]
    
    if not metric_values:
        return {'error': f'Metric {metric} not found in any model'}
    
    # Sort models by metric (descending)
    sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
    
    best_model = sorted_models[0][0]
    best_score = sorted_models[0][1]
    
    # Calculate relative performance
    relative_performance = {}
    for model_name, score in metric_values.items():
        relative_performance[model_name] = {
            'score': score,
            'relative_to_best': score - best_score,
            'relative_percentage': ((score - best_score) / best_score) * 100 if best_score > 0 else 0
        }
    
    return {
        'metric': metric,
        'best_model': best_model,
        'best_score': best_score,
        'ranking': [name for name, _ in sorted_models],
        'relative_performance': relative_performance
    }

def generate_performance_report(y_true: np.ndarray, 
                              y_pred_proba: np.ndarray,
                              model_name: str = "Model",
                              approach: str = "Unknown") -> Dict[str, Any]:
    """
    Generate comprehensive performance report.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        approach: Approach used (e.g., 'approach1_cross_site')
    
    Returns:
        Comprehensive performance report
    """
    report = {
        'model_name': model_name,
        'approach': approach,
        'dataset_info': {
            'total_samples': len(y_true),
            'positive_samples': int(np.sum(y_true)),
            'negative_samples': int(len(y_true) - np.sum(y_true)),
            'positive_rate': float(np.mean(y_true))
        }
    }
    
    # Standard metrics
    report['standard_metrics'] = calculate_standard_metrics(y_true, y_pred_proba)
    
    # Optimal threshold metrics
    report['optimal_threshold_metrics'] = calculate_standard_metrics(
        y_true, y_pred_proba, 
        threshold=calculate_threshold_metrics(y_true, y_pred_proba)['optimal_threshold']
    )
    
    # Calibration metrics
    report['calibration_metrics'] = calculate_calibration_metrics(y_true, y_pred_proba)
    
    # Threshold analysis
    report['threshold_analysis'] = calculate_threshold_metrics(y_true, y_pred_proba)
    
    return report

def save_metrics_to_file(metrics: Dict[str, Any], 
                        file_path: str,
                        format: str = 'json') -> None:
    """
    Save metrics to file.
    
    Args:
        metrics: Metrics dictionary
        file_path: Path to save file
        format: File format ('json' or 'csv')
    """
    import json
    import os
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if format == 'json':
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    elif format == 'csv':
        # Flatten metrics for CSV
        flat_metrics = {}
        
        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    # Skip lists in CSV output
                    continue
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flat_metrics = flatten_dict(metrics)
        df = pd.DataFrame([flat_metrics])
        df.to_csv(file_path, index=False)

if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 1000
    y_true = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    y_pred_proba = np.random.beta(2, 5, n_samples)  # Biased toward lower probabilities
    
    # Make predictions somewhat correlated with true labels
    mask = y_true == 1
    y_pred_proba[mask] = y_pred_proba[mask] + 0.3
    y_pred_proba = np.clip(y_pred_proba, 0, 1)
    
    # Generate comprehensive report
    report = generate_performance_report(
        y_true, y_pred_proba, 
        model_name="Example_XGBoost", 
        approach="approach1_cross_site"
    )
    
    print(f"AUC-ROC: {report['standard_metrics']['auc_roc']:.3f}")
    print(f"F1-Score: {report['standard_metrics']['f1_score']:.3f}")
    print(f"Precision: {report['standard_metrics']['precision']:.3f}")
    print(f"Recall: {report['standard_metrics']['recall']:.3f}")
    print(f"Expected Calibration Error: {report['calibration_metrics']['expected_calibration_error']:.3f}")