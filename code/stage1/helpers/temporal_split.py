#!/usr/bin/env python3
"""
Temporal data splitting utilities for FLAME-ICU federated learning.
Handles splitting data into training (2018-2022) and testing (2023-2024) periods.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any
import logging

def temporal_split(data: pd.DataFrame, 
                  datetime_col: str = 'datetime',
                  train_years: list = [2018, 2019, 2020, 2021, 2022],
                  test_years: list = [2023, 2024]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data based on temporal periods.
    
    Args:
        data: Input dataframe with datetime column
        datetime_col: Name of the datetime column
        train_years: List of years for training data
        test_years: List of years for testing data
    
    Returns:
        Tuple of (train_data, test_data)
    """
    if datetime_col not in data.columns:
        raise ValueError(f"Column '{datetime_col}' not found in data")
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(data[datetime_col]):
        data[datetime_col] = pd.to_datetime(data[datetime_col])
    
    # Extract year
    data['year'] = data[datetime_col].dt.year
    
    # Split data
    train_data = data[data['year'].isin(train_years)].copy()
    test_data = data[data['year'].isin(test_years)].copy()
    
    # Remove temporary year column
    train_data = train_data.drop('year', axis=1)
    test_data = test_data.drop('year', axis=1)
    
    logging.info(f"Training data: {len(train_data)} rows ({min(train_years)}-{max(train_years)})")
    logging.info(f"Testing data: {len(test_data)} rows ({min(test_years)}-{max(test_years)})")
    
    return train_data, test_data

def validate_temporal_consistency(train_data: pd.DataFrame, 
                                test_data: pd.DataFrame,
                                datetime_col: str = 'datetime') -> Dict[str, Any]:
    """
    Validate that temporal split was performed correctly.
    
    Args:
        train_data: Training dataset
        test_data: Testing dataset  
        datetime_col: Name of datetime column
    
    Returns:
        Dictionary with validation results
    """
    results = {}
    
    if datetime_col not in train_data.columns or datetime_col not in test_data.columns:
        results['status'] = 'error'
        results['message'] = f"Column '{datetime_col}' not found"
        return results
    
    # Get date ranges
    train_min = train_data[datetime_col].min()
    train_max = train_data[datetime_col].max()
    test_min = test_data[datetime_col].min()
    test_max = test_data[datetime_col].max()
    
    results['train_date_range'] = (train_min, train_max)
    results['test_date_range'] = (test_min, test_max)
    results['train_count'] = len(train_data)
    results['test_count'] = len(test_data)
    
    # Check for overlap (should be none for proper temporal split)
    overlap = train_max >= test_min
    results['temporal_overlap'] = overlap
    
    if not overlap:
        results['status'] = 'valid'
        results['message'] = 'Temporal split is valid - no overlap between train and test periods'
    else:
        results['status'] = 'warning'
        results['message'] = 'Temporal overlap detected between train and test periods'
    
    return results

def generate_temporal_statistics(data: pd.DataFrame, 
                                split_config: Dict[str, Any],
                                datetime_col: str = 'datetime') -> Dict[str, Any]:
    """
    Generate statistics about temporal data distribution.
    
    Args:
        data: Input dataframe
        split_config: Configuration with train_years and test_years
        datetime_col: Name of datetime column
    
    Returns:
        Dictionary with temporal statistics
    """
    stats = {}
    
    if datetime_col not in data.columns:
        stats['error'] = f"Column '{datetime_col}' not found"
        return stats
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(data[datetime_col]):
        data[datetime_col] = pd.to_datetime(data[datetime_col])
    
    # Overall statistics
    stats['total_records'] = len(data)
    stats['date_range'] = (data[datetime_col].min(), data[datetime_col].max())
    
    # Year-wise distribution
    data['year'] = data[datetime_col].dt.year
    year_counts = data['year'].value_counts().sort_index()
    stats['yearly_distribution'] = year_counts.to_dict()
    
    # Split statistics
    train_years = split_config.get('train_years', [2018, 2019, 2020, 2021, 2022])
    test_years = split_config.get('test_years', [2023, 2024])
    
    train_count = len(data[data['year'].isin(train_years)])
    test_count = len(data[data['year'].isin(test_years)])
    
    stats['train_count'] = train_count
    stats['test_count'] = test_count
    stats['train_percentage'] = (train_count / len(data)) * 100
    stats['test_percentage'] = (test_count / len(data)) * 100
    
    return stats

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2018-01-01', '2024-12-31', freq='D')
    sample_data = pd.DataFrame({
        'datetime': np.random.choice(dates, 1000),
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'mortality': np.random.choice([0, 1], 1000)
    })
    
    # Perform temporal split
    train_data, test_data = temporal_split(sample_data)
    
    # Validate split
    validation = validate_temporal_consistency(train_data, test_data)
    print(f"Validation: {validation}")
    
    # Generate statistics
    split_config = {'train_years': [2018, 2019, 2020, 2021, 2022], 'test_years': [2023, 2024]}
    stats = generate_temporal_statistics(sample_data, split_config)
    print(f"Statistics: {stats}")