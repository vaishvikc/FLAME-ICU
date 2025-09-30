#!/usr/bin/env python3
"""
Approach 1 Stage 1: Inference Script for Cross-Site Validation

This script loads trained models from Stage 1 and runs inference on test data:
- Loads XGBoost and Neural Network models trained at RUSH
- Applies same preprocessing (QuantileTransformer) used during training
- Generates predictions on test set
- Calculates comprehensive evaluation metrics
- Saves results for Stage 2 sharing

This simulates how federated sites would use the trained models
for cross-site validation without any local training.
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from datetime import datetime
from pathlib import Path

# Add parent directories to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
approach_dir = os.path.dirname(script_dir)  # approach1_cross_site/
code_dir = os.path.dirname(approach_dir)    # code/
sys.path.insert(0, approach_dir)

from approach_1_utils import (
    load_config,
    evaluate_model,
    ICUMortalityNN,
    apply_missing_value_handling
)

warnings.filterwarnings('ignore')


def load_trained_models(config):
    """
    Load trained XGBoost and Neural Network models with their artifacts
    """
    print("Loading trained models...")

    # Get model directories
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    project_root = os.path.dirname(script_dir)
    models_base_dir = os.path.join(project_root, config['output_paths']['models_dir'])

    models = {}

    # Load XGBoost model
    print("Loading XGBoost model...")
    xgb_dir = os.path.join(models_base_dir, 'xgboost')

    # Load XGBoost model
    xgb_model_path = os.path.join(xgb_dir, 'xgb_model.json')
    if not os.path.exists(xgb_model_path):
        raise FileNotFoundError(f"XGBoost model not found: {xgb_model_path}")

    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_model_path)

    # Load feature names
    feature_names_path = os.path.join(xgb_dir, 'feature_names.pkl')
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)

    models['xgboost'] = {
        'model': xgb_model,
        'feature_names': feature_names
    }

    # Load Neural Network model
    print("Loading Neural Network model...")
    nn_dir = os.path.join(models_base_dir, 'nn')

    # Load model config to reconstruct architecture
    nn_config_path = os.path.join(nn_dir, 'model_config.json')
    with open(nn_config_path, 'r') as f:
        model_config = json.load(f)

    # Reconstruct NN model
    input_size = len(feature_names)
    nn_params = model_config['nn_params']
    nn_model = ICUMortalityNN(
        input_size=input_size,
        hidden_sizes=nn_params['hidden_sizes'],
        dropout_rate=nn_params['dropout_rate'],
        activation=nn_params['activation'],
        batch_norm=nn_params['batch_norm']
    )

    # Load trained weights
    nn_model_path = os.path.join(nn_dir, 'nn_model.pth')
    nn_model.load_state_dict(torch.load(nn_model_path, map_location='cpu'))
    nn_model.eval()

    models['nn'] = {
        'model': nn_model,
        'feature_names': feature_names
    }

    print("✅ All models loaded successfully")
    return models


def load_test_data(config, feature_names):
    """
    Load and preprocess test data for inference
    """
    print("Loading test data...")

    # Get absolute path to data
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, config['data_config']['consolidated_features_path'])

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_parquet(data_path)

    # Filter to test set only
    test_df = df[df['split_type'] == 'test'].copy()
    print(f"Test data shape: {test_df.shape}")

    # Prepare features (same as training)
    exclude_cols = config['data_config']['exclude_columns']
    available_feature_cols = [col for col in feature_names if col in test_df.columns]

    if len(available_feature_cols) != len(feature_names):
        missing_features = set(feature_names) - set(available_feature_cols)
        print(f"Warning: Missing features in test data: {missing_features}")

    # Extract features and target
    X_test = test_df[available_feature_cols]
    y_test = test_df[config['data_config']['target_column']]
    test_ids = test_df['hospitalization_id']

    # Note: Model-specific missing value handling will be applied per model during inference
    # - XGBoost: no imputation needed (handles NaN natively)
    # - Neural Network: will fill with -1 during inference

    print(f"Test features shape: {X_test.shape}")
    print(f"Test target distribution: {y_test.value_counts().to_dict()}")
    print(f"Missing values in test data: {X_test.isnull().sum().sum()}")

    return X_test, y_test, test_ids


def run_inference(models, X_test, y_test, test_ids, config):
    """
    Run inference with both models and generate predictions
    """
    print("Running inference...")

    results = {}

    for model_type, model_data in models.items():
        print(f"\nRunning {model_type} inference...")

        model = model_data['model']

        # Apply model-specific missing value handling
        X_test_processed = apply_missing_value_handling(X_test, config, model_type)

        # Generate predictions
        if model_type == 'xgboost':
            # Create DMatrix with feature names for XGBoost
            dtest = xgb.DMatrix(X_test_processed, feature_names=model_data['feature_names'])
            y_pred_proba = model.predict(dtest)
        else:  # neural network
            X_test_tensor = torch.FloatTensor(X_test_processed.values)
            with torch.no_grad():
                y_pred_proba = model(X_test_tensor).numpy().flatten()

        # Calculate metrics
        threshold = config['evaluation_config']['threshold']
        y_pred = (y_pred_proba > threshold).astype(int)

        # Store results
        results[model_type] = {
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'model_type': model_type
        }

        # Print basic metrics
        from sklearn.metrics import roc_auc_score, accuracy_score
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        print(f"  AUC: {auc:.4f}")
        print(f"  Accuracy: {acc:.4f}")

    # Create ensemble prediction (simple average)
    ensemble_proba = (results['xgboost']['y_pred_proba'] + results['nn']['y_pred_proba']) / 2
    ensemble_pred = (ensemble_proba > threshold).astype(int)

    results['ensemble'] = {
        'y_pred_proba': ensemble_proba,
        'y_pred': ensemble_pred,
        'model_type': 'ensemble'
    }

    # Print ensemble metrics
    ensemble_auc = roc_auc_score(y_test, ensemble_proba)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"\nEnsemble (Average):")
    print(f"  AUC: {ensemble_auc:.4f}")
    print(f"  Accuracy: {ensemble_acc:.4f}")

    return results


def calculate_detailed_metrics(y_test, results, config):
    """
    Calculate comprehensive evaluation metrics for all models
    """
    print("Calculating detailed metrics...")

    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
        average_precision_score, brier_score_loss, confusion_matrix, roc_curve,
        precision_recall_curve
    )
    from sklearn.calibration import calibration_curve

    detailed_results = {}

    for model_type, pred_data in results.items():
        y_pred_proba = pred_data['y_pred_proba']
        y_pred = pred_data['y_pred']

        # Basic metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba)
        }

        # Calibration error
        try:
            n_bins = config['evaluation_config']['calibration_bins']
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_pred_proba, n_bins=n_bins, strategy='uniform'
            )
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            metrics['calibration_error'] = calibration_error
        except:
            metrics['calibration_error'] = np.nan

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)

        detailed_results[model_type] = {
            'metrics': metrics,
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            },
            'pr_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        }

        # Print summary
        print(f"\n{model_type.upper()} Model Results:")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  PR AUC: {metrics['pr_auc']:.4f}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")

    return detailed_results


def save_inference_results(detailed_results, y_test, test_ids, config):
    """
    Save inference results and predictions for Stage 2 sharing
    """
    print("Saving inference results...")

    # Get output directory
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, config['output_paths']['results_dir'])
    os.makedirs(results_dir, exist_ok=True)

    # Save detailed metrics (aggregate statistics only)
    metrics_path = os.path.join(results_dir, 'inference_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    # Create summary report
    summary = {
        'approach': config['approach_name'],
        'stage': config['stage'],
        'inference_timestamp': datetime.now().isoformat(),
        'test_set_size': len(y_test),
        'mortality_rate': float(y_test.mean()),
        'model_performance': {
            'xgboost_auc': detailed_results['xgboost']['metrics']['roc_auc'],
            'nn_auc': detailed_results['nn']['metrics']['roc_auc'],
            'ensemble_auc': detailed_results['ensemble']['metrics']['roc_auc']
        },
        'files_created': {
            'metrics': 'inference_metrics.json',
            'summary': 'approach_1_stage_1_inference_summary.json'
        }
    }

    summary_path = os.path.join(results_dir, 'approach_1_stage_1_inference_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Results saved to: {results_dir}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Summary: {summary_path}")

    return results_dir


def main():
    """Main inference function for Approach 1 Stage 1"""
    print("=" * 80)
    print("FLAME-ICU Approach 1 Stage 1: Cross-Site Inference")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load configuration
    config_path = os.path.join(approach_dir, 'approach_1_config.json')
    config = load_config(config_path)

    print(f"Approach: {config['approach_name']}")
    print(f"Stage: {config['stage']}")
    print()

    try:
        # Step 1: Load trained models
        print("📥 STEP 1: Loading trained models")
        print("-" * 50)
        models = load_trained_models(config)
        feature_names = models['xgboost']['feature_names']
        print()

        # Step 2: Load test data
        print("📊 STEP 2: Loading test data")
        print("-" * 50)
        X_test, y_test, test_ids = load_test_data(config, feature_names)
        print()

        # Step 3: Run inference
        print("🔮 STEP 3: Running inference")
        print("-" * 50)
        results = run_inference(models, X_test, y_test, test_ids, config)
        print()

        # Step 4: Calculate detailed metrics
        print("📈 STEP 4: Calculating detailed metrics")
        print("-" * 50)
        detailed_results = calculate_detailed_metrics(y_test, results, config)
        print()

        # Step 5: Save results
        print("💾 STEP 5: Saving inference results")
        print("-" * 50)
        results_dir = save_inference_results(detailed_results, y_test, test_ids, config)
        print()

        print("=" * 80)
        print("🎉 INFERENCE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("Test Set Performance Summary:")
        print(f"XGBoost AUC: {detailed_results['xgboost']['metrics']['roc_auc']:.4f}")
        print(f"Neural Network AUC: {detailed_results['nn']['metrics']['roc_auc']:.4f}")
        print(f"Ensemble AUC: {detailed_results['ensemble']['metrics']['roc_auc']:.4f}")
        print()
        print("Results ready for Stage 2 cross-site analysis!")
        print(f"Output directory: {results_dir}")

    except Exception as e:
        print(f"❌ ERROR: Inference failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()