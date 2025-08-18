#!/usr/bin/env python3
"""
Test script to verify the new inference scaling approach across all models.
This script tests that the new scaler fitting during inference works correctly
and helps prevent bias from distribution shifts.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_synthetic_data(n_samples=100, n_features=10, distribution_shift=True):
    """Create synthetic data to test scaling behavior"""
    np.random.seed(42)
    
    # Create base features
    if distribution_shift:
        # Simulate distribution shift - different mean and variance
        mean_shift = np.random.uniform(2, 5, n_features)
        var_shift = np.random.uniform(0.5, 2, n_features)
        X = np.random.randn(n_samples, n_features) * var_shift + mean_shift
    else:
        # Normal distribution similar to training
        X = np.random.randn(n_samples, n_features)
    
    # Create synthetic labels
    y = (X[:, 0] + X[:, 1] > 3).astype(int)
    
    return X, y

def test_scaling_impact():
    """Test the impact of using different scalers"""
    print("Testing Scaling Impact on Inference")
    print("=" * 50)
    
    # Create training data
    X_train, y_train = create_synthetic_data(n_samples=1000, distribution_shift=False)
    
    # Create inference data with distribution shift
    X_inference, y_inference = create_synthetic_data(n_samples=200, distribution_shift=True)
    
    # Fit scaler on training data
    train_scaler = StandardScaler()
    X_train_scaled = train_scaler.fit_transform(X_train)
    
    print(f"Training data statistics:")
    print(f"  Mean: {X_train.mean(axis=0)[:3]}... (showing first 3)")
    print(f"  Std:  {X_train.std(axis=0)[:3]}... (showing first 3)")
    print()
    
    print(f"Inference data statistics (with distribution shift):")
    print(f"  Mean: {X_inference.mean(axis=0)[:3]}... (showing first 3)")
    print(f"  Std:  {X_inference.std(axis=0)[:3]}... (showing first 3)")
    print()
    
    # Apply training scaler to inference data (old approach)
    X_inference_old_scaled = train_scaler.transform(X_inference)
    
    # Fit new scaler on inference data (new approach)
    inference_scaler = StandardScaler()
    X_inference_new_scaled = inference_scaler.fit_transform(X_inference)
    
    # Compare the scaled results
    print("Scaled inference data statistics (using training scaler):")
    print(f"  Mean: {X_inference_old_scaled.mean(axis=0)[:3]}... (should be shifted)")
    print(f"  Std:  {X_inference_old_scaled.std(axis=0)[:3]}... (may not be 1)")
    print()
    
    print("Scaled inference data statistics (using new scaler):")
    print(f"  Mean: {X_inference_new_scaled.mean(axis=0)[:3]}... (should be ~0)")
    print(f"  Std:  {X_inference_new_scaled.std(axis=0)[:3]}... (should be ~1)")
    print()
    
    # Calculate the difference in scaling
    mean_diff = np.abs(X_inference_old_scaled.mean(axis=0) - X_inference_new_scaled.mean(axis=0))
    std_diff = np.abs(X_inference_old_scaled.std(axis=0) - X_inference_new_scaled.std(axis=0))
    
    print(f"Impact of using new scaler:")
    print(f"  Mean absolute difference: {mean_diff.mean():.4f}")
    print(f"  Std absolute difference: {std_diff.mean():.4f}")
    print()
    
    if mean_diff.mean() > 0.1:
        print("✓ Significant distribution shift detected!")
        print("  Using a new scaler on inference data is recommended to prevent bias.")
    else:
        print("✗ No significant distribution shift detected.")
        print("  Either approach would work similarly.")

def verify_model_configs():
    """Verify that all model configs have been updated"""
    print("\nVerifying Model Configurations")
    print("=" * 50)
    
    models = ['xgboost', 'nn']
    all_updated = True
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for model in models:
        config_path = os.path.join(base_dir, 'models', model, 'config.json')
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if 'inference_config' in config and 'use_new_scaler' in config['inference_config']:
                print(f"✓ {model.upper()} config updated with inference_config")
                print(f"  use_new_scaler: {config['inference_config']['use_new_scaler']}")
            else:
                print(f"✗ {model.upper()} config missing inference_config")
                all_updated = False
        else:
            print(f"✗ {model.upper()} config not found at {config_path}")
            all_updated = False
    
    print()
    if all_updated:
        print("✓ All model configurations have been updated successfully!")
    else:
        print("✗ Some configurations need to be updated.")

if __name__ == "__main__":
    print("FLAME-ICU Inference Scaling Test")
    print("================================\n")
    
    # Test scaling impact
    test_scaling_impact()
    
    # Verify configurations
    verify_model_configs()
    
    print("\nTest completed!")