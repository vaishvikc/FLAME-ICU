#!/usr/bin/env python3
"""
Approach 2 Stage 1: Transfer Learning with Main Model Initialization

This script implements Approach 2 (Transfer Learning) for Stage 1:
- Loads RUSH pre-trained models from Approach 1
- Fine-tunes models on local training data (with reduced learning rate)
- Uses validation set for early stopping
- Runs inference on local test data
- Saves fine-tuned models and results in the same format as Approach 1

Federated sites run this script to:
1. Load base models trained at RUSH
2. Fine-tune on their local training data
3. Evaluate on validation during training
4. Test on their local test set
5. Save results for Stage 2
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
approach_dir = os.path.dirname(script_dir)  # approach2_transfer_learning/
code_dir = os.path.dirname(approach_dir)    # code/
project_root = os.path.dirname(code_dir)    # project root
sys.path.insert(0, approach_dir)
sys.path.insert(0, os.path.join(code_dir, 'approach1_cross_site', 'stage_1'))

from approach_2_utils import (
    load_config,
    load_site_config,
    load_and_preprocess_data,
    load_base_models,
    fine_tune_xgboost,
    fine_tune_nn,
    evaluate_model,
    save_model_artifacts
)

# Import inference utilities from approach 1 to reuse the same output format
sys.path.insert(0, os.path.join(code_dir, 'approach1_cross_site', 'stage_1'))
from inference import (
    run_inference,
    calculate_detailed_metrics,
    save_inference_results
)

warnings.filterwarnings('ignore')


def main():
    """Main transfer learning function for Approach 2 Stage 1"""
    print("=" * 80)
    print("FLAME-ICU Approach 2 Stage 1: Transfer Learning with Model Fine-Tuning")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load site configuration
    site_name = load_site_config()
    print(f"Site: {site_name}")
    print()

    # Load configuration
    config_path = os.path.join(approach_dir, 'approach_2_config.json')
    config = load_config(config_path)

    print(f"Approach: {config['approach_name']}")
    print(f"Stage: {config['stage']}")
    print(f"Transfer Learning LR Multiplier: {config['transfer_learning']['learning_rate_multiplier']}")
    print()

    # Set random seeds for reproducibility
    np.random.seed(config['random_seeds']['global'])

    try:
        # Step 1: Load base models from Approach 1
        print("📥 STEP 1: Loading pre-trained base models")
        print("-" * 50)
        xgb_base, nn_base, base_feature_names = load_base_models(config)
        print()

        # Step 2: Load and preprocess local data
        print("📊 STEP 2: Loading and preprocessing local data")
        print("-" * 50)
        splits, feature_names = load_and_preprocess_data(config)
        print(f"✅ Data loaded successfully with {len(feature_names)} features")
        print()

        # Verify feature consistency
        if feature_names != base_feature_names:
            print("⚠️  WARNING: Feature names differ from base model")
            print("Ensure feature alignment for proper transfer learning")

        # Step 3: Fine-tune XGBoost model
        print("🌳 STEP 3: Fine-tuning XGBoost model")
        print("-" * 50)
        xgb_model, xgb_evals = fine_tune_xgboost(xgb_base, splits, config, feature_names)
        print("✅ XGBoost fine-tuning completed")
        print()

        # Step 4: Fine-tune Neural Network model
        print("🧠 STEP 4: Fine-tuning Neural Network model")
        print("-" * 50)
        nn_model, nn_history = fine_tune_nn(nn_base, splits, config, feature_names)
        print("✅ Neural Network fine-tuning completed")
        print()

        # Step 5: Evaluate fine-tuned models on validation set
        print("📈 STEP 5: Evaluating fine-tuned models on validation set")
        print("-" * 50)

        print("Evaluating XGBoost model...")
        xgb_results = evaluate_model(xgb_model, splits, config, model_type='xgboost', splits_to_eval=['val'])

        print("\nEvaluating Neural Network model...")
        nn_results = evaluate_model(nn_model, splits, config, model_type='nn', splits_to_eval=['val'])
        print()

        # Step 6: Save fine-tuned model artifacts
        print("💾 STEP 6: Saving fine-tuned model artifacts")
        print("-" * 50)

        xgb_model_dir = save_model_artifacts(
            xgb_model, feature_names, config, 'xgboost', xgb_results, site_name
        )

        nn_model_dir = save_model_artifacts(
            nn_model, feature_names, config, 'nn', nn_results, site_name
        )
        print()

        # Step 7: Run inference on test set
        print("🔬 STEP 7: Running inference on test set")
        print("-" * 50)

        # Package models for inference (match Approach 1 format)
        models = {
            'xgboost': {
                'model': xgb_model,
                'feature_names': feature_names
            },
            'nn': {
                'model': nn_model,
                'feature_names': feature_names
            }
        }

        # Get test data
        X_test = splits['test']['features']
        y_test = splits['test']['target']
        test_ids = splits['test']['hospitalization_id']

        # Run inference (reuses Approach 1's inference function)
        results = run_inference(models, X_test, y_test, test_ids, config)
        print()

        # Step 8: Calculate detailed metrics
        print("📊 STEP 8: Creating comprehensive evaluation results")
        print("-" * 50)

        detailed_results = calculate_detailed_metrics(y_test, results, config)
        print()

        # Step 9: Save inference results (includes plots)
        print("💾 STEP 9: Saving inference results and plots")
        print("-" * 50)

        save_inference_results(detailed_results, y_test, results, config)
        print()

        # Step 10: Create summary report
        print("📋 STEP 10: Creating summary report")
        print("-" * 50)
        create_summary_report(config, xgb_results, nn_results, detailed_results, site_name, feature_names)
        print()

        print("=" * 80)
        print("🎉 TRANSFER LEARNING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("Model Performance Summary:")
        print(f"XGBoost Test AUC: {detailed_results['xgboost']['metrics']['roc_auc']:.4f}")
        print(f"Neural Network Test AUC: {detailed_results['nn']['metrics']['roc_auc']:.4f}")
        print()
        print("Fine-tuned models and results saved!")
        print(f"Models saved to: {os.path.join(config['output_paths']['models_dir'], site_name)}")
        print(f"Results saved to: {config['output_paths']['results_dir']}")
        print()
        print("✅ Ready for Stage 2 cross-site testing")

    except Exception as e:
        print(f"❌ ERROR: Transfer learning failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def create_summary_report(config, xgb_results, nn_results, detailed_results, site_name, feature_names):
    """
    Create a comprehensive summary report

    Args:
        config: Configuration dictionary
        xgb_results: XGBoost validation results from training phase
        nn_results: Neural Network validation results from training phase
        detailed_results: Test results from final inference phase
        site_name: Name of the site
        feature_names: List of feature names
    """

    # Get output directory
    results_dir = os.path.join(project_root, config['output_paths']['results_dir'])
    os.makedirs(results_dir, exist_ok=True)

    # Create summary data
    summary = {
        'site': site_name,
        'approach': config['approach_name'],
        'stage': config['stage'],
        'timestamp': datetime.now().isoformat(),
        'transfer_learning': {
            'base_models_source': 'RUSH (Approach 1)',
            'learning_rate_multiplier': config['transfer_learning']['learning_rate_multiplier'],
            'fine_tuning_data': 'Local training split',
            'validation_data': 'Local validation split'
        },
        'data_info': {
            'n_features': len(feature_names),
            'scaling_method': 'RobustScaler (applied in preprocessing)',
            'missing_value_handling': config['preprocessing']['handle_missing']
        },
        'model_performance': {
            'xgboost': {
                'val_auc': xgb_results['val']['metrics']['roc_auc'],
                'val_accuracy': xgb_results['val']['metrics']['accuracy'],
                'val_precision': xgb_results['val']['metrics']['precision'],
                'val_recall': xgb_results['val']['metrics']['recall'],
                'val_f1': xgb_results['val']['metrics']['f1_score'],
                'test_auc': detailed_results['xgboost']['metrics']['roc_auc'],
                'test_accuracy': detailed_results['xgboost']['metrics']['accuracy'],
                'test_precision': detailed_results['xgboost']['metrics']['precision'],
                'test_recall': detailed_results['xgboost']['metrics']['recall'],
                'test_f1': detailed_results['xgboost']['metrics']['f1_score']
            },
            'neural_network': {
                'val_auc': nn_results['val']['metrics']['roc_auc'],
                'val_accuracy': nn_results['val']['metrics']['accuracy'],
                'val_precision': nn_results['val']['metrics']['precision'],
                'val_recall': nn_results['val']['metrics']['recall'],
                'val_f1': nn_results['val']['metrics']['f1_score'],
                'test_auc': detailed_results['nn']['metrics']['roc_auc'],
                'test_accuracy': detailed_results['nn']['metrics']['accuracy'],
                'test_precision': detailed_results['nn']['metrics']['precision'],
                'test_recall': detailed_results['nn']['metrics']['recall'],
                'test_f1': detailed_results['nn']['metrics']['f1_score']
            }
        },
        'next_steps': {
            'stage_2_instructions': "Fine-tuned models ready for cross-site testing",
            'model_locations': {
                'xgboost': f"{config['output_paths']['models_dir']}{site_name}/xgboost/",
                'neural_network': f"{config['output_paths']['models_dir']}{site_name}/nn/"
            }
        }
    }

    # Save summary
    summary_path = os.path.join(results_dir, f'approach_2_stage_1_summary_{site_name}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()