#!/usr/bin/env python3
"""
One-time script to run LSTM architecture exploration and hyperparameter optimization
This will find the best LSTM architecture and then optimize its hyperparameters
"""

import subprocess
import sys
import json
import os
import pandas as pd

def run_architecture_exploration():
    """Run architecture exploration and get best architecture"""
    print("=" * 50)
    print("STEP 1: Running LSTM Architecture Exploration")
    print("=" * 50)
    
    # Run architecture exploration
    result = subprocess.run(
        [sys.executable, "lstm_architecture_exploration.py"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    if result.returncode != 0:
        print("Error running architecture exploration:")
        print(result.stderr)
        return None
    
    print(result.stdout)
    
    # Load the best architecture results
    site_name = get_site_name()
    best_arch_file = f"../../protected_outputs/models/lstm/architecture_exploration/{site_name}_best_architecture.json"
    
    if os.path.exists(best_arch_file):
        with open(best_arch_file, 'r') as f:
            best_arch = json.load(f)
        return best_arch
    else:
        print(f"Best architecture file not found: {best_arch_file}")
        return None

def update_config_with_architecture(best_arch):
    """Update LSTM config with best architecture"""
    print("\n" + "=" * 50)
    print("STEP 2: Updating Config with Best Architecture")
    print("=" * 50)
    
    # Load current config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update model params based on best architecture
    arch_params = best_arch['params']
    
    # Map architecture parameters to config
    if 'hidden_size' in arch_params:
        config['model_params']['hidden_size1'] = arch_params['hidden_size']
    if 'hidden_sizes' in arch_params:
        # For stacked LSTM
        config['model_params']['hidden_size1'] = arch_params['hidden_sizes'][0]
        if len(arch_params['hidden_sizes']) > 1:
            config['model_params']['hidden_size2'] = arch_params['hidden_sizes'][1]
    
    config['model_params']['dropout_rate'] = arch_params.get('dropout_rate', 0.2)
    
    # Add architecture info
    config['architecture_info'] = {
        'best_architecture': best_arch['architecture'],
        'model_class': best_arch['model_class'],
        'cv_scores': best_arch['cv_scores'],
        'exploration_date': pd.Timestamp.now().isoformat()
    }
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config updated with best architecture: {best_arch['architecture']}")
    print(f"CV ROC-AUC: {best_arch['cv_scores']['roc_auc']:.4f}")
    
    return config

def run_hyperparameter_optimization():
    """Run hyperparameter optimization"""
    print("\n" + "=" * 50)
    print("STEP 3: Running Hyperparameter Optimization")
    print("=" * 50)
    
    # Run hyperparameter optimization
    result = subprocess.run(
        [sys.executable, "lstm_hyperparameter_optimization.py"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    if result.returncode != 0:
        print("Error running hyperparameter optimization:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True

def run_final_training():
    """Run final training with optimized parameters"""
    print("\n" + "=" * 50)
    print("STEP 4: Running Final Training")
    print("=" * 50)
    
    # Run training
    result = subprocess.run(
        [sys.executable, "training.py"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    if result.returncode != 0:
        print("Error running training:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True

def get_site_name():
    """Get site name from preprocessing config"""
    preprocessing_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'preprocessing', 'config_demo.json'
    )
    try:
        with open(preprocessing_config_path, 'r') as f:
            preprocessing_config = json.load(f)
        return preprocessing_config.get('site', 'unknown')
    except FileNotFoundError:
        return 'unknown'

def main():
    """Main function to run full LSTM optimization pipeline"""
    print("=" * 70)
    print("LSTM FULL OPTIMIZATION PIPELINE")
    print("=" * 70)
    
    # Step 1: Architecture exploration
    best_arch = run_architecture_exploration()
    if best_arch is None:
        print("Architecture exploration failed. Exiting.")
        return 1
    
    # Step 2: Update config with best architecture
    config = update_config_with_architecture(best_arch)
    
    # Step 3: Hyperparameter optimization
    if not run_hyperparameter_optimization():
        print("Hyperparameter optimization failed. Exiting.")
        return 1
    
    # Step 4: Final training
    user_input = input("\nDo you want to run final training with optimized parameters? (y/n): ")
    if user_input.lower() == 'y':
        if not run_final_training():
            print("Training failed.")
            return 1
    else:
        print("Skipping final training. You can run it later with: python training.py")
    
    print("\n" + "=" * 70)
    print("LSTM OPTIMIZATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    # Print final summary
    site_name = get_site_name()
    print(f"\nResults saved in:")
    print(f"  - Architecture exploration: ../../protected_outputs/models/lstm/architecture_exploration/")
    print(f"  - Optimization plots: ../../protected_outputs/models/lstm/optimization_plots/")
    print(f"  - Trained model: ../../protected_outputs/models/lstm/lstm_{site_name}_icu_mortality_model.pt")
    print(f"  - Metrics: ../../protected_outputs/models/lstm/{site_name}_metrics.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())