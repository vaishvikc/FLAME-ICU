#!/usr/bin/env python3
"""Debug version of LSTM training to see what's happening"""

import sys
import os
import json

# Add debug prints
print("Starting LSTM training debug...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Check config
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
print(f"Config path: {config_path}")
print(f"Config exists: {os.path.exists(config_path)}")

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("Config loaded successfully")
    print(f"Train file: {config['data_split']['train_file']}")
    print(f"Test file: {config['data_split']['test_file']}")
    
    # Check if files exist
    train_file = config['data_split']['train_file']
    test_file = config['data_split']['test_file']
    
    if not os.path.isabs(train_file):
        train_file = os.path.join(os.path.dirname(config_path), train_file)
    if not os.path.isabs(test_file):
        test_file = os.path.join(os.path.dirname(config_path), test_file)
        
    print(f"Train file exists: {os.path.exists(train_file)}")
    print(f"Test file exists: {os.path.exists(test_file)}")
    
    # Check file sizes
    if os.path.exists(train_file):
        print(f"Train file size: {os.path.getsize(train_file) / 1024 / 1024:.2f} MB")
    if os.path.exists(test_file):
        print(f"Test file size: {os.path.getsize(test_file) / 1024 / 1024:.2f} MB")

# Try loading the data
print("\nAttempting to load training data...")
import pickle
import time

start_time = time.time()
try:
    with open(train_file, 'rb') as f:
        print("Opening train file...")
        train_data = pickle.load(f)
        print(f"Training data loaded in {time.time() - start_time:.2f} seconds")
        print(f"X_train shape: {train_data['X'].shape}")
        print(f"y_train shape: {train_data['y'].shape}")
        print(f"Number of features: {len(train_data['feature_cols'])}")
except Exception as e:
    print(f"Error loading training data: {e}")

# Now run the actual training with smaller epochs for testing
print("\nRunning abbreviated training...")
import training

# Temporarily modify config for faster testing
original_epochs = training.load_config()['training_config']['num_epochs']
config['training_config']['num_epochs'] = 5  # Just 5 epochs for testing

# Save temporary config
with open(config_path + '.bak', 'w') as f:
    json.dump(config, f)

try:
    # Run training
    training.train_lstm_model()
finally:
    # Restore original config
    original_config = training.load_config()
    original_config['training_config']['num_epochs'] = original_epochs
    with open(config_path, 'w') as f:
        json.dump(original_config, f, indent=2)