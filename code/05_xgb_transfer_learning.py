import pandas as pd
import numpy as np
import os
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.calibration import CalibrationDisplay

def load_model_and_metadata(model_dir):
    """Load saved XGBoost model, scaler, and feature columns"""
    model_path = os.path.join(model_dir, 'xgb_icu_mortality_model.json')
    scaler_path = os.path.join(model_dir, 'xgb_feature_scaler.pkl')
    feature_cols_path = os.path.join(model_dir, 'xgb_feature_columns.pkl')
    
    # Load feature columns
    with open(feature_cols_path, 'rb') as f:
        feature_cols = pickle.load(f)
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load XGBoost model
    model = xgb.Booster()
    model.load_model(model_path)
    
    return model, scaler, feature_cols

def prepare_aggregated_data(df, available_features):
    """
    Aggregate data by hospitalization_id, calculating min, max, and median for each feature
    Similar to the training process in 03_02_xgbmodel.py
    """
    # Create aggregation functions
    aggregation_funcs = {}
    for feature in available_features:
        aggregation_funcs[feature] = ['min', 'max', 'median']
    
    # Add disposition to the aggregation
    if 'disposition' in df.columns:
        aggregation_funcs['disposition'] = 'first'
    
    # Perform the aggregation
    agg_df = df.groupby('hospitalization_id').agg(aggregation_funcs)
    
    # Flatten the multi-level column index
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    
    # Reset index to make hospitalization_id a column again
    agg_df = agg_df.reset_index()
    
    return agg_df

def transfer_learning(data_path, model_dir, output_dir, num_boost_round=100):
    """
    Perform transfer learning on a new hospital system dataset using XGBoost
    
    Parameters:
    -----------
    data_path : str
        Path to the new hospital data file (parquet format)
    model_dir : str
        Directory containing the original saved model and metadata
    output_dir : str
        Directory to save the transfer-learned model and results
    num_boost_round : int
        Number of additional boosting rounds for transfer learning
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create graphs directory if it doesn't exist
    graphs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'final', 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Load pre-trained model and metadata
    original_model, scaler, expected_feature_cols = load_model_and_metadata(model_dir)
    
    # Load new hospital data
    print(f"Loading new hospital data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Data shape: {df.shape}")
    
    # Extract the base feature names from the expected columns
    base_features = set()
    for col in expected_feature_cols:
        if col.endswith('_min') or col.endswith('_max') or col.endswith('_median'):
            base_feature = col.rsplit('_', 1)[0]
            base_features.add(base_feature)
    
    base_features = list(base_features)
    
    # Check which features are available in the dataframe
    available_features = [col for col in base_features if col in df.columns]
    missing_features = [col for col in base_features if col not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing base features: {missing_features}")
    
    print(f"Available features: {available_features}")
    
    # Check for required columns
    required_cols = ['hospitalization_id', 'disposition'] + available_features
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        print("Available columns:", df.columns.tolist())
        return None
    
    # Prepare aggregated data
    print("Preparing aggregated data...")
    agg_df = prepare_aggregated_data(df[required_cols], available_features)
    
    print(f"Aggregated data shape: {agg_df.shape}")
    
    # Extract features and targets
    feature_columns = [col for col in agg_df.columns if col not in ['hospitalization_id', 'disposition_first']]
    X = agg_df[feature_columns]
    y = agg_df['disposition_first']
    
    print(f"Dataset has {len(X)} patients with {sum(y)} positive cases ({sum(y)/len(y)*100:.2f}%)")
    
    # Split data 50/50 for transfer learning and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    
    print(f"Transfer learning set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Handle missing values
    X_train_filled = np.nan_to_num(X_train.values, nan=0.0)
    X_test_filled = np.nan_to_num(X_test.values, nan=0.0)
    
    # Apply the same scaling as the original model
    X_train_scaled = scaler.transform(X_train_filled)
    X_test_scaled = scaler.transform(X_test_filled)
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=feature_columns)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=feature_columns)
    
    # Set up transfer learning parameters (more conservative)
    scale_pos_weight = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))
    
    transfer_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.01,  # Much lower learning rate for fine-tuning
        'max_depth': 2,  # Reduced depth to prevent overfitting
        'min_child_weight': 5,  # More conservative
        'subsample': 0.6,  # Sample fewer records
        'colsample_bytree': 0.6,  # Sample fewer features
        'scale_pos_weight': scale_pos_weight,
        'gamma': 2,  # Higher minimum loss reduction
        'reg_alpha': 0.5,  # Increased L1 regularization
        'reg_lambda': 2.0,  # Increased L2 regularization
        'seed': 42,
        'process_type': 'update',  # Important for transfer learning
        'refresh_leaf': True,  # Update tree leaf values
        'updater': 'refresh'  # Use refresh updater for transfer learning
    }
    
    # Set up evaluation list
    eval_list = [(dtrain, 'train'), (dtest, 'eval')]
    
    # Perform transfer learning by continuing training from the original model
    print("Starting transfer learning...")
    
    # Create a copy of the original model for transfer learning
    transfer_model = xgb.train(
        transfer_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=eval_list,
        xgb_model=original_model,  # Continue from original model
        early_stopping_rounds=20,
        verbose_eval=10
    )
    
    print(f"Transfer learning completed with {transfer_model.best_iteration} iterations")
    
    # Evaluate transfer-learned model
    print("\nEvaluating transfer-learned model...")
    y_pred_proba = transfer_model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Transfer-learned model - Accuracy: {accuracy:.4f}")
    print(f"Transfer-learned model - ROC AUC: {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Create calibration plot
    plt.figure(figsize=(10, 6))
    try:
        n_bins = min(5, len(np.unique(y_pred_proba)))
        disp = CalibrationDisplay.from_predictions(y_test, y_pred_proba, n_bins=n_bins, name='Transfer-Learned XGBoost')
        plt.title('Calibration Plot - Transfer Learning')
        plt.savefig(os.path.join(graphs_dir, 'xgb_transfer_learning_calibration.png'))
        plt.close()
        print("Calibration plot saved successfully.")
    except Exception as e:
        print(f"Warning: Could not create calibration plot: {e}")
    
    # Save the transfer-learned model
    transfer_model_path = os.path.join(output_dir, 'transfer_xgb_model.json')
    transfer_model.save_model(transfer_model_path)
    print(f"Transfer-learned model saved to {transfer_model_path}")
    
    # Copy the scaler and feature columns to the new directory
    scaler_path = os.path.join(output_dir, 'xgb_feature_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    feature_cols_path = os.path.join(output_dir, 'xgb_feature_columns.pkl')
    with open(feature_cols_path, 'wb') as f:
        pickle.dump(list(feature_columns), f)
    
    # Compare with original model performance on the same test set
    print("\nComparing with original model performance...")
    original_y_pred_proba = original_model.predict(dtest)
    original_y_pred = (original_y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics for original model
    original_accuracy = accuracy_score(y_test, original_y_pred)
    original_roc_auc = roc_auc_score(y_test, original_y_pred_proba)
    
    print(f"Original Model - Accuracy: {original_accuracy:.4f}")
    print(f"Original Model - ROC AUC: {original_roc_auc:.4f}")
    
    print("\nOriginal Model Classification Report:")
    print(classification_report(y_test, original_y_pred))
    
    print("Original Model Confusion Matrix:")
    print(confusion_matrix(y_test, original_y_pred))
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'ROC AUC'],
        'Original Model': [original_accuracy, original_roc_auc],
        'Transfer-Learned Model': [accuracy, roc_auc],
        'Improvement': [accuracy - original_accuracy, roc_auc - original_roc_auc]
    })
    
    print("\nModel Comparison:")
    print(comparison)
    
    # Save comparison to CSV
    comparison.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Plot feature importance comparison if possible
    try:
        plt.figure(figsize=(15, 8))
        
        # Original model importance
        plt.subplot(1, 2, 1)
        original_importance = original_model.get_score(importance_type='weight')
        if original_importance:
            names = list(original_importance.keys())
            values = list(original_importance.values())
            plt.barh(names[:10], values[:10])  # Top 10 features
            plt.title('Original Model Feature Importance')
            plt.xlabel('Weight')
        
        # Transfer-learned model importance
        plt.subplot(1, 2, 2)
        transfer_importance = transfer_model.get_score(importance_type='weight')
        if transfer_importance:
            names = list(transfer_importance.keys())
            values = list(transfer_importance.values())
            plt.barh(names[:10], values[:10])  # Top 10 features
            plt.title('Transfer-Learned Model Feature Importance')
            plt.xlabel('Weight')
        
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, 'xgb_feature_importance_comparison.png'))
        plt.close()
        print("Feature importance comparison plot saved successfully.")
    except Exception as e:
        print(f"Warning: Could not create feature importance plot: {e}")
    
    return {
        'model': transfer_model,
        'original_metrics': {
            'accuracy': original_accuracy,
            'roc_auc': original_roc_auc
        },
        'transfer_metrics': {
            'accuracy': accuracy,
            'roc_auc': roc_auc
        }
    }

def main():
    """Main function for transfer learning"""
    # Get paths
    code_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(code_dir)
    model_dir = os.path.join(project_dir, 'model')
    output_dir = os.path.join(project_dir, 'model', 'xgb_transfer_learning')
    
    # For demonstration, using the original data as "new hospital data"
    # In a real scenario, this would be data from a different hospital
    data_path = os.path.join(project_dir, 'output', 'intermitted', 'by_event_wide_df.parquet')
    
    # Check if original model exists
    if not os.path.exists(os.path.join(model_dir, 'xgb_icu_mortality_model.json')):
        print(f"Error: Original XGBoost model not found in {model_dir}")
        return
    
    # Perform transfer learning
    results = transfer_learning(data_path, model_dir, output_dir)
    
    if results is None:
        print("Transfer learning could not be completed.")
        return
    
    print("\nXGBoost transfer learning completed!")
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main()