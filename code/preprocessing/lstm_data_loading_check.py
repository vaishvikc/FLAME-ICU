#!/usr/bin/env python3
"""Simple script to check LSTM data loading"""

import pickle
import time
import os
import sys

print("Checking LSTM data files...")

# File paths
train_file = "../../protected_outputs/intermediate/data/lstm_train_sequences.pkl"
test_file = "../../protected_outputs/intermediate/data/lstm_test_sequences.pkl"

print(f"Train file size: {os.path.getsize(train_file) / 1024 / 1024:.2f} MB")
print(f"Test file size: {os.path.getsize(test_file) / 1024 / 1024:.2f} MB")

# Try loading with different methods
print("\n1. Loading train data with pickle.load()...")
start = time.time()
try:
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    print(f"  Loaded in {time.time() - start:.2f} seconds")
    print(f"  X shape: {train_data['X'].shape}")
    print(f"  y shape: {train_data['y'].shape}")
    print(f"  Features: {len(train_data['feature_cols'])}")
    
    # Check memory usage
    import numpy as np
    X_memory = train_data['X'].nbytes / 1024 / 1024
    print(f"  X array memory: {X_memory:.2f} MB")
    
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Checking for NaN values...")
if 'train_data' in locals():
    nan_count = np.isnan(train_data['X']).sum()
    print(f"  NaN values in X: {nan_count}")
    print(f"  NaN percentage: {100 * nan_count / train_data['X'].size:.2f}%")

print("\nDone!")