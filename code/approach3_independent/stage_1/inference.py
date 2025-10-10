#!/usr/bin/env python3
"""
Approach 3 Stage 1: Inference Script for Independent Site Models

This script loads site-specific trained models and runs inference on test data:
- Loads XGBoost and Neural Network models trained independently at the site
- Generates predictions on local test set (2023-2024)
- Calculates comprehensive evaluation metrics
- Saves results in site-specific folders for BOX upload

Note: This is run after train_models.py completes successfully.
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
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from pathlib import Path

# Set matplotlib backend for non-interactive plotting
matplotlib.use('Agg')

# Add parent directories to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
approach_dir = os.path.dirname(script_dir)  # approach3_independent/
code_dir = os.path.dirname(approach_dir)    # code/
sys.path.insert(0, approach_dir)

from approach_3_utils import (
    load_config,
    ICUMortalityNN,
    apply_missing_value_handling,
    calculate_bootstrap_confidence_intervals
)

warnings.filterwarnings('ignore')


def load_site_config():
    """
    Load site configuration from clif_config.json
    Returns site name from the config file
    """
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    approach_dir = os.path.dirname(script_dir)
    code_dir = os.path.dirname(approach_dir)
    project_root = os.path.dirname(code_dir)

    clif_config_path = os.path.join(project_root, 'clif_config.json')

    try:
        with open(clif_config_path, 'r') as f:
            clif_config = json.load(f)
        site_name = clif_config['site'].upper()
        return site_name
    except FileNotFoundError:
        raise FileNotFoundError(
            f"clif_config.json not found at {clif_config_path}. "
            "Please ensure the file exists in the project root."
        )
    except KeyError:
        raise KeyError(
            "'site' field not found in clif_config.json. "
            "Please ensure the config file contains a 'site' field."
        )


def update_config_paths(config, site_name):
    """Update config paths to include site name"""
    config_updated = config.copy()

    # Replace {site_name} placeholder in all output paths
    for key in config_updated['output_paths']:
        config_updated['output_paths'][key] = config_updated['output_paths'][key].replace(
            '{site_name}', site_name
        )

    return config_updated


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
    available_feature_cols = [col for col in feature_names if col in test_df.columns]

    if len(available_feature_cols) != len(feature_names):
        missing_features = set(feature_names) - set(available_feature_cols)
        print(f"Warning: Missing features in test data: {missing_features}")

    # Extract features and target
    X_test = test_df[available_feature_cols]
    y_test = test_df[config['data_config']['target_column']]
    test_ids = test_df['hospitalization_id']

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

        # Calibration data
        try:
            n_bins = config['evaluation_config']['calibration_bins']
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_pred_proba, n_bins=n_bins, strategy='uniform'
            )
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            metrics['calibration_error'] = calibration_error

            calibration_data = {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist(),
                'n_bins': n_bins,
                'calibration_error': calibration_error
            }
        except:
            metrics['calibration_error'] = np.nan
            calibration_data = {
                'fraction_of_positives': [],
                'mean_predicted_value': [],
                'n_bins': 0,
                'calibration_error': np.nan
            }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Calculate bootstrap confidence intervals
        ci_results = calculate_bootstrap_confidence_intervals(
            y_test.values, y_pred_proba, y_pred, config
        )
        metrics.update(ci_results)

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
            },
            'calibration_curve': calibration_data
        }

        # Print summary with confidence intervals if available
        print(f"\n{model_type.upper()} Model Results:")
        if 'roc_auc_ci_lower' in metrics:
            print(f"  ROC AUC: {metrics['roc_auc']:.4f} [{metrics['roc_auc_ci_lower']:.4f}-{metrics['roc_auc_ci_upper']:.4f}]")
            print(f"  Accuracy: {metrics['accuracy']:.4f} [{metrics['accuracy_ci_lower']:.4f}-{metrics['accuracy_ci_upper']:.4f}]")
            print(f"  Precision: {metrics['precision']:.4f} [{metrics['precision_ci_lower']:.4f}-{metrics['precision_ci_upper']:.4f}]")
            print(f"  Recall: {metrics['recall']:.4f} [{metrics['recall_ci_lower']:.4f}-{metrics['recall_ci_upper']:.4f}]")
            print(f"  F1 Score: {metrics['f1_score']:.4f} [{metrics['f1_score_ci_lower']:.4f}-{metrics['f1_score_ci_upper']:.4f}]")
            print(f"  PR AUC: {metrics['pr_auc']:.4f} [{metrics['pr_auc_ci_lower']:.4f}-{metrics['pr_auc_ci_upper']:.4f}]")
            print(f"  Brier Score: {metrics['brier_score']:.4f} [{metrics['brier_score_ci_lower']:.4f}-{metrics['brier_score_ci_upper']:.4f}]")
        else:
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  PR AUC: {metrics['pr_auc']:.4f}")
            print(f"  Brier Score: {metrics['brier_score']:.4f}")

    return detailed_results


def save_inference_results(detailed_results, y_test, results, config, site_name):
    """
    Save inference results and plots
    """
    print("Saving inference results...")

    # Get output directories
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, config['output_paths']['results_dir'])
    plots_dir = os.path.join(project_root, config['output_paths']['plots_dir'])
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Generate and save plots
    print("Generating plots...")
    plot_roc_curves(detailed_results, site_name, plots_dir)
    plot_calibration_curves(y_test, results, site_name, plots_dir)

    # Create results structure
    site_results = {
        'site': site_name,
        'approach': config['approach_name'],
        'stage': config['stage'],
        'timestamp': datetime.now().isoformat(),
        'data_summary': {
            'test_set_size': len(y_test),
            'mortality_rate': float(y_test.mean()),
            'positive_cases': int(y_test.sum()),
            'negative_cases': int(len(y_test) - y_test.sum())
        },
        'models': {}
    }

    # Add model-specific data
    for model_type in ['xgboost', 'nn']:
        site_results['models'][model_type] = {
            'metrics': detailed_results[model_type]['metrics'],
            'plot_data': {
                'roc_curve': detailed_results[model_type]['roc_curve'],
                'calibration_curve': detailed_results[model_type]['calibration_curve']
            }
        }

    # Save metrics file
    metrics_path = os.path.join(results_dir, f'inference_metrics_{site_name}.json')
    with open(metrics_path, 'w') as f:
        json.dump(site_results, f, indent=2)

    # Create summary
    summary = {
        'site': site_name,
        'approach': config['approach_name'],
        'stage': config['stage'],
        'inference_timestamp': datetime.now().isoformat(),
        'test_set_size': len(y_test),
        'mortality_rate': float(y_test.mean()),
        'model_performance': {
            'xgboost_auc': detailed_results['xgboost']['metrics']['roc_auc'],
            'nn_auc': detailed_results['nn']['metrics']['roc_auc']
        },
        'files_created': {
            'metrics': f'inference_metrics_{site_name}.json',
            'plots': {
                'roc_curves': f'plots/roc_curves_{site_name}.png',
                'calibration_curves': f'plots/calibration_curves_{site_name}.png'
            }
        }
    }

    summary_path = os.path.join(results_dir, f'inference_summary_{site_name}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Results saved to: {results_dir}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Summary: {summary_path}")
    print(f"  - Plots: {plots_dir}")

    return results_dir


def plot_roc_curves(detailed_results, site_name, plots_dir):
    """Generate and save ROC curves for all models"""
    plt.figure(figsize=(10, 8))

    colors = {'xgboost': '#1f77b4', 'nn': '#ff7f0e'}
    model_names = {'xgboost': 'XGBoost', 'nn': 'Neural Network'}

    for model_type in ['xgboost', 'nn']:
        roc_data = detailed_results[model_type]['roc_curve']
        auc = detailed_results[model_type]['metrics']['roc_auc']

        plt.plot(roc_data['fpr'], roc_data['tpr'],
                color=colors[model_type], linewidth=2,
                label=f'{model_names[model_type]} (AUC = {auc:.3f})')

    # Add diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {site_name.upper()} (Independent Training)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'roc_curves_{site_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ ROC curves saved: {plot_path}")


def plot_calibration_curves(y_test, results, site_name, plots_dir):
    """Generate calibration plots"""
    from sklearn.calibration import calibration_curve

    plt.figure(figsize=(10, 8))

    colors = {'xgboost': '#1f77b4', 'nn': '#ff7f0e'}
    model_names = {'xgboost': 'XGBoost', 'nn': 'Neural Network'}

    for model_type in ['xgboost', 'nn']:
        y_pred_proba = results[model_type]['y_pred_proba']

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=10, strategy='uniform'
        )

        plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                color=colors[model_type], linewidth=2, markersize=8,
                label=f'{model_names[model_type]}')

    # Add perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Perfect Calibration')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(f'Calibration Curves - {site_name.upper()} (Independent Training)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'calibration_curves_{site_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Calibration curves saved: {plot_path}")


def main():
    """Main inference function for Approach 3 Stage 1"""
    # Load site name from clif_config.json
    site_name = load_site_config()

    print("=" * 80)
    print("FLAME-ICU Approach 3 Stage 1: Independent Model Inference")
    print("=" * 80)
    print(f"Site: {site_name} (auto-detected from clif_config.json)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load configuration
    config_path = os.path.join(approach_dir, 'approach_3_config.json')
    config = load_config(config_path)

    # Update config with site-specific paths
    config = update_config_paths(config, site_name)

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
        results_dir = save_inference_results(detailed_results, y_test, results, config, site_name)
        print()

        print("=" * 80)
        print("🎉 INFERENCE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Site: {site_name}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("Test Set Performance Summary:")
        print(f"XGBoost AUC: {detailed_results['xgboost']['metrics']['roc_auc']:.4f}")
        print(f"Neural Network AUC: {detailed_results['nn']['metrics']['roc_auc']:.4f}")
        print()
        print(f"Upload results to BOX: PHASE1_RESULTS_UPLOAD_ME/approach_3_stage_1/{site_name}/")

    except Exception as e:
        print(f"❌ ERROR: Inference failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
