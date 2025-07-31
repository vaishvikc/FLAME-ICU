#!/usr/bin/env python3
"""
LSTM Data Splitting Script
Splits sequential data into train/test sets for LSTM model training
Saves splits to intermediate folder for consistent usage across experiments
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

def load_config():
    """Load configuration from config file"""
    # Try models/lstm/config.json first
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'lstm', 'config.json')
    if not os.path.exists(config_path):
        # Fallback to preprocessing directory
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_preprocessing_config():
    """Load preprocessing configuration to get site name"""
    # Try top-level config_demo.json first (new location)
    preprocessing_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'config_demo.json'
    )
    
    if not os.path.exists(preprocessing_config_path):
        # Fallback to old location
        preprocessing_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'preprocessing', 'config_demo.json'
        )
    
    try:
        with open(preprocessing_config_path, 'r') as f:
            preprocessing_config = json.load(f)
        return preprocessing_config
    except FileNotFoundError:
        print(f"Warning: Preprocessing config not found at {preprocessing_config_path}")
        return {"site": "unknown"}

def prepare_sequences(df, feature_cols, sequence_length=24):
    """
    Transform the data into sequences for each hospitalization_id
    Apply padding for hospitalization_ids with fewer than sequence_length hours
    Handle missing values by filling with zeros
    """
    sequences = {}
    targets = {}
    
    for hosp_id, group in df.groupby('hospitalization_id'):
        # Sort by hour
        group = group.sort_values('nth_hour')
        
        # Get the target (same for all rows of the same hospitalization_id)
        target = group['disposition'].iloc[0]
        
        # Create sequence from features
        seq = group[feature_cols].values
        
        # Handle NaN values by replacing with zeros
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Pad if needed (if less than sequence_length hours)
        if len(seq) < sequence_length:
            # Create padding (zeros)
            padding = np.zeros((sequence_length - len(seq), len(feature_cols)))
            seq = np.vstack([seq, padding])
        elif len(seq) > sequence_length:
            # Truncate if more than sequence_length hours
            seq = seq[:sequence_length]
        
        # Store sequence and target
        sequences[hosp_id] = seq
        targets[hosp_id] = target
    
    return sequences, targets

def main():
    """Main function to create LSTM data splits"""
    print("=== LSTM Data Splitting ===")
    print("Creating consistent train/test splits for LSTM models\n")
    
    # Load configuration
    config = load_config()
    preprocessing_config = load_preprocessing_config()
    site_name = preprocessing_config.get('site', 'unknown')
    
    # Get paths and parameters from config
    preprocessing_path = config['data_config']['preprocessing_path']
    feature_file = config['data_config']['feature_file']
    sequence_length = config['data_config']['sequence_length']
    
    # Get split parameters
    split_config = config['data_split']
    train_ratio = split_config['train_ratio']
    test_ratio = split_config['test_ratio']
    random_state = split_config['random_state']
    stratify = split_config['stratify']
    
    # Convert relative path to absolute path based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(preprocessing_path):
        preprocessing_path = os.path.abspath(os.path.join(script_dir, preprocessing_path))
    
    # Load data
    data_path = os.path.join(preprocessing_path, feature_file)
    print(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Total data shape: {df.shape}")
    print(f"Unique hospitalization_ids: {df['hospitalization_id'].nunique():,}")
    
    # Extract relevant columns (all _max and _min columns)
    max_min_cols = [col for col in df.columns if '_max' in col or '_min' in col]
    required_cols = ['hospitalization_id', 'disposition', 'nth_hour'] + max_min_cols
    df_filtered = df[required_cols]
    
    print(f"\nFeatures per timestep: {len(max_min_cols)}")
    print(f"Sequence length: {sequence_length}")
    
    # Check class distribution
    print("\nClass distribution:")
    for label, count in df_filtered.groupby('hospitalization_id')['disposition'].first().value_counts().items():
        total = df_filtered['hospitalization_id'].nunique()
        print(f"  Class {label}: {count:,} ({100 * count / total:.2f}%)")
    
    # Prepare sequences
    print("\nPreparing sequences...")
    sequences, targets = prepare_sequences(df_filtered, max_min_cols, sequence_length)
    
    # Convert to numpy arrays
    hospitalization_ids = list(sequences.keys())
    X = np.array(list(sequences.values()))
    y = np.array(list(targets.values()))
    
    print(f"\nPrepared data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Perform train/test split
    print(f"\nSplitting data: {train_ratio*100:.0f}% train, {test_ratio*100:.0f}% test")
    print(f"Random state: {random_state}")
    print(f"Stratified: {stratify}")
    
    test_size = test_ratio / (train_ratio + test_ratio)
    
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, hospitalization_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )
    
    print(f"\nSplit results:")
    print(f"  Training set: {len(X_train):,} patients ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test):,} patients ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Training mortality rate: {y_train.mean():.3f}")
    print(f"  Test mortality rate: {y_test.mean():.3f}")
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(script_dir, "../../protected_outputs/intermediate/lstm"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    train_data = {
        'X': X_train,
        'y': y_train,
        'hospitalization_ids': ids_train,
        'feature_cols': max_min_cols
    }
    
    test_data = {
        'X': X_test,
        'y': y_test,
        'hospitalization_ids': ids_test,
        'feature_cols': max_min_cols
    }
    
    train_path = os.path.join(output_dir, "train_sequences.pkl")
    test_path = os.path.join(output_dir, "test_sequences.pkl")
    
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"\n✅ Data splits saved successfully:")
    print(f"  Training data: {train_path}")
    print(f"  Test data: {test_path}")
    
    # Save split metadata
    split_metadata = {
        'site': site_name,
        'total_patients': len(X),
        'train_patients': len(X_train),
        'test_patients': len(X_test),
        'train_mortality_rate': float(y_train.mean()),
        'test_mortality_rate': float(y_test.mean()),
        'sequence_length': X.shape[1],
        'num_features': X.shape[2],
        'feature_names': max_min_cols,
        'split_date': pd.Timestamp.now().isoformat(),
        'random_state': random_state,
        'train_ratio': train_ratio,
        'test_ratio': test_ratio,
        'stratified': stratify
    }
    
    metadata_path = os.path.join(output_dir, "split_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(split_metadata, f, indent=2)
    
    print(f"  Split metadata: {metadata_path}")
    
    # Update config with new paths
    config['data_split']['train_file'] = train_path
    config['data_split']['test_file'] = test_path
    
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Updated config.json with new data paths")
    print("\nData splitting complete!")

if __name__ == "__main__":
    main()