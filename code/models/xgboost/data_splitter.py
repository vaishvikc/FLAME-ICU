import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

def load_config():
    """Load configuration from config file"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_preprocessing_config():
    """Load preprocessing configuration to get site name"""
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

def load_and_prepare_data():
    """Load and prepare data exactly like in training scripts"""
    print("Loading and preparing data...")
    
    # Load configuration
    config = load_config()
    preprocessing_config = load_preprocessing_config()
    site_name = preprocessing_config.get('site', 'unknown')
    
    # Get paths from config
    preprocessing_path = config['data_config']['preprocessing_path']
    feature_file = config['data_config']['feature_file']
    selected_features = config['data_config']['selected_features']
    
    # Load data
    data_path = os.path.join(preprocessing_path, feature_file)
    df = pd.read_parquet(data_path)
    print(f"Data shape: {df.shape}")
    print(f"Number of unique hospitalization_ids: {df['hospitalization_id'].nunique()}")

    # Check which features from our selection are available in the dataframe
    available_features = [col for col in selected_features if col in df.columns]
    missing_features = [col for col in selected_features if col not in df.columns]
    if missing_features:
        print(f"Warning: The following requested features are not in the dataset: {missing_features}")

    print(f"Available features: {len(available_features)}")

    # Identify categorical features (ending with _category)
    categorical_features = [col for col in available_features if col.endswith('_category')]
    numeric_features = [col for col in available_features if not col.endswith('_category')]
    
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Numeric features: {len(numeric_features)}")

    # Add hospitalization_id and disposition to the list of columns we need
    required_cols = ['hospitalization_id', 'disposition'] + available_features
    df_filtered = df[required_cols].copy()

    print(f"Filtered data shape: {df_filtered.shape}")
    print(f"Class distribution: \n{df_filtered['disposition'].value_counts()}")

    # Enhanced aggregation with categorical and numeric features
    print("Aggregating data by hospitalization_id...")
    
    # Start with basic columns
    agg_dfs = []
    
    # Handle numeric features with min, max, median, mean
    if numeric_features:
        print(f"Processing {len(numeric_features)} numeric features...")
        numeric_aggregation = {}
        for feature in numeric_features:
            # Special handling for age - just take first value
            if feature == 'age_at_admission':
                numeric_aggregation[feature] = 'first'
            else:
                numeric_aggregation[feature] = ['min', 'max', 'median', 'mean']
        numeric_aggregation['disposition'] = 'first'
        
        numeric_agg = df_filtered.groupby('hospitalization_id').agg(numeric_aggregation)
        # Flatten column names, but handle age_at_admission specially
        new_columns = []
        for col in numeric_agg.columns:
            if isinstance(col, tuple):
                if col[0] == 'age_at_admission':
                    new_columns.append('age_at_admission')
                else:
                    new_columns.append('_'.join(col).strip())
            else:
                new_columns.append(col)
        numeric_agg.columns = new_columns
        numeric_agg = numeric_agg.reset_index()
        agg_dfs.append(numeric_agg)
    
    # Handle categorical features with binary one-hot encoding
    if categorical_features:
        print(f"Processing {len(categorical_features)} categorical features...")
        for cat_feature in categorical_features:
            print(f"  Processing {cat_feature}...")
            
            # Create binary one-hot encoded columns (1 if exists, 0 if not)
            cat_data = df_filtered[['hospitalization_id', cat_feature]].copy()
            cat_data = cat_data.dropna(subset=[cat_feature])  # Remove rows where category is NaN
            
            if len(cat_data) > 0:
                # Get unique categories for this feature
                unique_categories = cat_data[cat_feature].unique()
                print(f"    Found {len(unique_categories)} categories")
                
                # Create binary indicator for each category
                # Use any() to get 1 if category exists for hospitalization, 0 otherwise
                cat_indicators = cat_data.drop_duplicates(subset=['hospitalization_id', cat_feature])
                cat_indicators['indicator'] = 1
                
                cat_pivot = cat_indicators.pivot_table(
                    index='hospitalization_id', 
                    columns=cat_feature, 
                    values='indicator', 
                    fill_value=np.nan,  # Let XGBoost handle missing values natively
                    aggfunc='max'  # In case of duplicates, max of 1s is still 1
                )
                
                # Rename columns to include feature name (no _count suffix)
                cat_pivot.columns = [f"{cat_feature}_{col}" for col in cat_pivot.columns]
                cat_pivot = cat_pivot.reset_index()
                
                agg_dfs.append(cat_pivot)
    
    # Combine all aggregated dataframes
    if len(agg_dfs) > 1:
        agg_df = agg_dfs[0]
        for df_to_merge in agg_dfs[1:]:
            agg_df = pd.merge(agg_df, df_to_merge, on='hospitalization_id', how='outer')
    else:
        agg_df = agg_dfs[0] if agg_dfs else pd.DataFrame()
    
    # Keep NaN values in categorical indicator columns for XGBoost native handling
    categorical_indicator_cols = [col for col in agg_df.columns if any(col.startswith(f"{cat}_") for cat in categorical_features)]
    print(f"Categorical indicator columns: {len(categorical_indicator_cols)} (keeping NaN values for native XGBoost handling)")

    print(f"Aggregated data shape: {agg_df.shape}")
    print(f"Number of unique hospitalization_ids after aggregation: {agg_df['hospitalization_id'].nunique()}")
    
    # Validate one row per hospitalization
    if len(agg_df) != agg_df['hospitalization_id'].nunique():
        print(f"WARNING: Found duplicate hospitalization_ids! {len(agg_df)} rows vs {agg_df['hospitalization_id'].nunique()} unique IDs")
        duplicates = agg_df[agg_df.duplicated(subset=['hospitalization_id'], keep=False)]
        print(f"Duplicate hospitalization_ids: {duplicates['hospitalization_id'].unique()[:5]}")
    else:
        print(f"✅ Confirmed: One row per hospitalization ({len(agg_df)} rows)")

    # Prepare features and target
    X = agg_df.drop(['hospitalization_id', 'disposition_first'], axis=1)
    y = agg_df['disposition_first']
    hospitalization_ids = agg_df['hospitalization_id']

    print(f"Final X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    
    return X, y, hospitalization_ids, site_name

def split_and_save_data():
    """Split data into train/test sets and save to protected folder"""
    print("=== Data Splitting for Consistent Train/Test Sets ===")
    
    # Load configuration
    config = load_config()
    split_config = config['data_split']
    
    # Load and prepare data
    X, y, hospitalization_ids, site_name = load_and_prepare_data()
    
    # Split data according to configuration
    train_ratio = split_config['train_ratio']
    test_ratio = split_config['test_ratio']
    random_state = split_config['random_state']
    stratify = split_config['stratify']
    
    print(f"\nSplitting data: {train_ratio*100:.0f}% train, {test_ratio*100:.0f}% test")
    print(f"Random state: {random_state}")
    print(f"Stratified: {stratify}")
    
    # Perform the split
    test_size = test_ratio / (train_ratio + test_ratio)  # Convert to sklearn format
    
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, hospitalization_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )
    
    print(f"\nSplit results:")
    print(f"Training set: {len(X_train):,} patients ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test):,} patients ({len(X_test)/len(X)*100:.1f}%)")
    print(f"Training mortality rate: {y_train.mean():.3f}")
    print(f"Test mortality rate: {y_test.mean():.3f}")
    
    # Create DataFrames for saving
    train_df = X_train.copy()
    train_df['disposition'] = y_train
    train_df['hospitalization_id'] = ids_train
    
    test_df = X_test.copy()
    test_df['disposition'] = y_test
    test_df['hospitalization_id'] = ids_test
    
    # Create output directory
    output_dir = "../../protected_outputs/intermediate/data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    train_path = os.path.join(output_dir, "train_df.parquet")
    test_path = os.path.join(output_dir, "test_df.parquet")
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"\n✅ Data splits saved successfully:")
    print(f"Training data: {train_path}")
    print(f"Test data: {test_path}")
    
    # Save split metadata
    split_metadata = {
        'site': site_name,
        'total_patients': len(X),
        'train_patients': len(X_train),
        'test_patients': len(X_test),
        'train_mortality_rate': float(y_train.mean()),
        'test_mortality_rate': float(y_test.mean()),
        'features': len(X.columns),
        'split_date': pd.Timestamp.now().isoformat(),
        'random_state': random_state,
        'stratified': stratify
    }
    
    metadata_path = os.path.join(output_dir, "split_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(split_metadata, f, indent=2)
    
    print(f"Split metadata: {metadata_path}")
    
    return train_df, test_df

def main():
    """Main function to create data splits"""
    split_and_save_data()

if __name__ == "__main__":
    main()