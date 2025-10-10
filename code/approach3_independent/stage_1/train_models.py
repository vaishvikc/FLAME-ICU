#!/usr/bin/env python3
"""
Approach 3 Stage 1: Independent Site Training

This script implements Approach 3 (Independent Site Training) for Stage 1:
- Each site trains XGBoost and Neural Network models from scratch on their local data
- Uses pre-optimized hyperparameters from RUSH (no local optimization needed)
- Trains on local training data with validation for early stopping
- Tests on local test data (2023-2024)
- Saves trained models in site-specific folders for BOX upload

Each site runs this independently on their own data.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directories to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
approach_dir = os.path.dirname(script_dir)  # approach3_independent/
code_dir = os.path.dirname(approach_dir)    # code/
sys.path.insert(0, approach_dir)

from approach_3_utils import (
    load_config,
    load_and_preprocess_data,
    train_xgboost_approach1,
    train_nn_approach1,
    evaluate_model,
    save_model_artifacts,
    extract_xgboost_feature_importance,
    calculate_permutation_importance,
    save_feature_importance,
    plot_feature_importance
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


def main():
    """Main training function for Approach 3 Stage 1"""
    # Load site name from clif_config.json
    site_name = load_site_config()

    print("=" * 80)
    print("FLAME-ICU Approach 3 Stage 1: Independent Site Training")
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
    print(f"Description: {config['description']}")
    print()

    # Set random seeds for reproducibility
    np.random.seed(config['random_seeds']['global'])

    try:
        # Step 1: Load and preprocess data
        print("📊 STEP 1: Loading and preprocessing data")
        print("-" * 50)
        print(f"Loading local data for {site_name}...")
        splits, feature_names = load_and_preprocess_data(config)
        print(f"✅ Data loaded successfully with {len(feature_names)} features")
        print()

        # Step 2: Train XGBoost model
        print("🌳 STEP 2: Training XGBoost model from scratch")
        print("-" * 50)
        print("Using pre-optimized hyperparameters (no local optimization)")
        xgb_model, xgb_evals = train_xgboost_approach1(splits, config, feature_names)
        print("✅ XGBoost training completed")
        print()

        # Step 2a: Extract XGBoost feature importance
        print("📊 STEP 2a: Extracting XGBoost feature importance")
        print("-" * 50)
        xgb_importance = extract_xgboost_feature_importance(xgb_model, feature_names)
        save_feature_importance(xgb_importance, 'xgboost', config)
        plot_feature_importance(xgb_importance, 'xgboost', config, top_n=20)
        print()

        # Step 3: Train Neural Network model
        print("🧠 STEP 3: Training Neural Network model from scratch")
        print("-" * 50)
        print("Using pre-optimized hyperparameters (no local optimization)")
        nn_model, nn_history = train_nn_approach1(splits, config, feature_names)
        print("✅ Neural Network training completed")
        print()

        # Step 3a: Calculate NN feature importance
        print("📊 STEP 3a: Calculating Neural Network feature importance")
        print("-" * 50)
        nn_importance = calculate_permutation_importance(
            nn_model, splits['val']['features'], splits['val']['target'],
            config, feature_names, n_repeats=5
        )
        save_feature_importance(nn_importance, 'nn', config)
        plot_feature_importance(nn_importance, 'nn', config, top_n=20)
        print()

        # Step 4: Evaluate models on validation set
        print("📈 STEP 4: Evaluating models on validation set")
        print("-" * 50)

        print("Evaluating XGBoost model...")
        xgb_results = evaluate_model(xgb_model, splits, config, model_type='xgboost', splits_to_eval=['val'])

        print("\nEvaluating Neural Network model...")
        nn_results = evaluate_model(nn_model, splits, config, model_type='nn', splits_to_eval=['val'])
        print()

        # Step 5: Save model artifacts
        print("💾 STEP 5: Saving model artifacts")
        print("-" * 50)

        # Save XGBoost model
        xgb_model_dir = save_model_artifacts(
            xgb_model, None, feature_names, config, 'xgboost', xgb_results
        )

        # Save Neural Network model
        nn_model_dir = save_model_artifacts(
            nn_model, None, feature_names, config, 'nn', nn_results
        )
        print()

        # Step 6: Create summary report
        print("📋 STEP 6: Creating summary report")
        print("-" * 50)
        create_summary_report(config, xgb_results, nn_results, feature_names,
                            xgb_importance, nn_importance, site_name)
        print()

        print("=" * 80)
        print("🎉 INDEPENDENT TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Site: {site_name}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("Model Performance Summary (Validation Set):")
        print(f"XGBoost Val AUC: {xgb_results['val']['metrics']['roc_auc']:.4f}")
        print(f"Neural Network Val AUC: {nn_results['val']['metrics']['roc_auc']:.4f}")
        print()
        print("Note: Test set performance will be evaluated separately during inference.")
        print()
        print("Trained models saved and ready for BOX upload!")
        print(f"XGBoost model: {xgb_model_dir}")
        print(f"Neural Network model: {nn_model_dir}")
        print()
        print(f"Upload the entire folder to BOX: PHASE1_RESULTS_UPLOAD_ME/approach_3_stage_1/{site_name}/")

    except Exception as e:
        print(f"❌ ERROR: Training failed with error: {str(e)}")
        raise


def create_summary_report(config, xgb_results, nn_results, feature_names,
                         xgb_importance, nn_importance, site_name):
    """Create a comprehensive summary report"""

    # Get output directory
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, config['output_paths']['results_dir'])
    os.makedirs(results_dir, exist_ok=True)

    # Create summary data
    summary = {
        'approach': config['approach_name'],
        'stage': config['stage'],
        'site_name': site_name,
        'training_timestamp': datetime.now().isoformat(),
        'training_type': 'independent_from_scratch',
        'hyperparameter_optimization': 'pre_optimized_rush_parameters',
        'data_info': {
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'scaling_method': 'RobustScaler (applied in preprocessing)',
            'missing_value_handling': config['preprocessing']['handle_missing']
        },
        'model_performance': {
            'xgboost': {
                'val_auc': xgb_results['val']['metrics']['roc_auc'],
                'val_accuracy': xgb_results['val']['metrics']['accuracy'],
                'val_precision': xgb_results['val']['metrics']['precision'],
                'val_recall': xgb_results['val']['metrics']['recall'],
                'val_f1': xgb_results['val']['metrics']['f1_score']
            },
            'neural_network': {
                'val_auc': nn_results['val']['metrics']['roc_auc'],
                'val_accuracy': nn_results['val']['metrics']['accuracy'],
                'val_precision': nn_results['val']['metrics']['precision'],
                'val_recall': nn_results['val']['metrics']['recall'],
                'val_f1': nn_results['val']['metrics']['f1_score']
            },
            'note': 'Test set performance will be evaluated separately during inference to prevent data leakage'
        },
        'feature_importance': {
            'xgboost_top_10_gain': xgb_importance['gain'][:10],
            'nn_top_10_permutation': nn_importance[:10]
        },
        'next_steps': {
            'stage_2_instructions': "Upload models to BOX for cross-site testing",
            'upload_folder': f"PHASE1_RESULTS_UPLOAD_ME/approach_3_stage_1/{site_name}/",
            'model_locations': {
                'xgboost': f"{config['output_paths']['models_dir']}xgboost/",
                'neural_network': f"{config['output_paths']['models_dir']}nn/"
            }
        }
    }

    # Save summary
    summary_path = os.path.join(results_dir, f'approach_3_stage_1_summary_{site_name}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
