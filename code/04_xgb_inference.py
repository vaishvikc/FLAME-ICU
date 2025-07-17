import pandas as pd
import numpy as np
import os
import pickle
import xgboost as xgb

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
    
    # Add disposition to the aggregation if it exists
    if 'disposition' in df.columns:
        aggregation_funcs['disposition'] = 'first'
    
    # Perform the aggregation
    agg_df = df.groupby('hospitalization_id').agg(aggregation_funcs)
    
    # Flatten the multi-level column index
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    
    # Reset index to make hospitalization_id a column again
    agg_df = agg_df.reset_index()
    
    return agg_df

def predict_mortality(data_path, model_dir):
    """
    Predict ICU mortality using the trained XGBoost model
    
    Parameters:
    -----------
    data_path : str
        Path to the data file (parquet format)
    model_dir : str
        Directory containing the saved model and metadata
    
    Returns:
    --------
    predictions : dict
        Dictionary mapping hospitalization_id to predicted mortality probability
    """
    # Load model and metadata
    model, scaler, expected_feature_cols = load_model_and_metadata(model_dir)
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Data shape: {df.shape}")
    
    # Extract the base feature names from the expected columns
    # Expected columns are like 'eosinophils_absolute_min', 'eosinophils_absolute_max', etc.
    base_features = set()
    for col in expected_feature_cols:
        if col.endswith('_min') or col.endswith('_max') or col.endswith('_median'):
            base_feature = col.rsplit('_', 1)[0]  # Remove the last part (_min, _max, _median)
            base_features.add(base_feature)
    
    base_features = list(base_features)
    
    # Check which features are available in the dataframe
    available_features = [col for col in base_features if col in df.columns]
    missing_features = [col for col in base_features if col not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing base features: {missing_features}")
    
    print(f"Available features: {available_features}")
    
    # Check for required columns
    required_cols = ['hospitalization_id'] + available_features
    if 'disposition' in df.columns:
        required_cols.append('disposition')
        
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
    hospitalization_ids = agg_df['hospitalization_id'].tolist()
    
    # Check if we have disposition data
    targets = None
    if 'disposition_first' in agg_df.columns:
        targets = agg_df['disposition_first'].tolist()
    
    # Handle missing values
    X_filled = np.nan_to_num(X.values, nan=0.0)
    
    # Scale features using the saved scaler
    print("Scaling features...")
    X_scaled = scaler.transform(X_filled)
    
    # Create DMatrix for XGBoost
    dmatrix = xgb.DMatrix(X_scaled, feature_names=feature_columns)
    
    # Make predictions
    print("Making predictions...")
    predictions = {}
    
    # Get probability predictions
    probs = model.predict(dmatrix)
    
    # Store predictions
    for i, hosp_id in enumerate(hospitalization_ids):
        predictions[hosp_id] = {
            'mortality_probability': float(probs[i]),
            'predicted_class': 1 if probs[i] > 0.5 else 0,
            'actual_class': targets[i] if targets else None
        }
    
    return predictions

def main():
    """Main function to run inference"""
    # Get paths
    code_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(code_dir)
    model_dir = os.path.join(project_dir, 'model')
    
    # For demonstration, using the original data file
    data_path = os.path.join(project_dir, 'output', 'intermitted', 'by_event_wide_df.parquet')
    
    # Check if model exists
    if not os.path.exists(os.path.join(model_dir, 'xgb_icu_mortality_model.json')):
        print(f"Error: XGBoost model file not found in {model_dir}")
        return
    
    # Make predictions
    predictions = predict_mortality(data_path, model_dir)
    
    if predictions is None:
        print("Error in making predictions. Please check the data format.")
        return
    
    # Print prediction results
    print("\nPrediction Results:")
    print(f"Total patients: {len(predictions)}")
    
    # Count predicted positives/negatives
    pred_positives = sum(1 for p in predictions.values() if p['predicted_class'] == 1)
    pred_negatives = sum(1 for p in predictions.values() if p['predicted_class'] == 0)
    
    print(f"Predicted positive (mortality): {pred_positives} ({pred_positives/len(predictions)*100:.2f}%)")
    print(f"Predicted negative (survival): {pred_negatives} ({pred_negatives/len(predictions)*100:.2f}%)")
    
    # Evaluate if actual labels are available
    if all(p['actual_class'] is not None for p in predictions.values()):
        from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
        
        y_true = [p['actual_class'] for p in predictions.values()]
        y_pred = [p['predicted_class'] for p in predictions.values()]
        y_prob = [p['mortality_probability'] for p in predictions.values()]
        
        print("\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"ROC AUC: {roc_auc_score(y_true, y_prob):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
    
    # Sample individual predictions (show first 5)
    print("\nSample Predictions:")
    for i, (hosp_id, pred) in enumerate(list(predictions.items())[:5]):
        print(f"Patient ID: {hosp_id}")
        print(f"  Mortality Probability: {pred['mortality_probability']:.4f}")
        print(f"  Predicted Outcome: {'Expired' if pred['predicted_class'] == 1 else 'Survived'}")
        if pred['actual_class'] is not None:
            print(f"  Actual Outcome: {'Expired' if pred['actual_class'] == 1 else 'Survived'}")
        print()
        
    print("XGBoost inference completed!")

if __name__ == "__main__":
    main()