import torch
import numpy as np
import pandas as pd
import pickle
import json
import os
from model import ICUMortalityNN


def load_config():
    """Load configuration from config file"""
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


def load_model_and_preprocessors(model_path=None, scaler_path=None, feature_cols_path=None):
    """
    Load trained model and preprocessing artifacts.
    
    Args:
        model_path: Path to saved model (optional, uses config if not provided)
        scaler_path: Path to saved scaler (optional)
        feature_cols_path: Path to saved feature columns (optional)
        
    Returns:
        model, scaler, feature_columns, device
    """
    config = load_config()
    output_config = config['output_config'].copy()
    
    # Load preprocessing config to get site name
    preprocessing_config = load_preprocessing_config()
    site_name = preprocessing_config.get('site', 'unknown')
    print(f"Loading model for site: {site_name}")
    
    # Update output paths with site name
    for key in output_config:
        if isinstance(output_config[key], str):
            # Replace {SITE_NAME} placeholder with actual site name
            output_config[key] = output_config[key].replace('/{SITE_NAME}/', f'/{site_name}/')
    
    # Use provided paths or fall back to config
    if model_path is None:
        model_path = output_config['model_path']
    if scaler_path is None:
        scaler_path = output_config['scaler_path']
    if feature_cols_path is None:
        feature_cols_path = output_config['feature_cols_path']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = ICUMortalityNN(
        input_size=checkpoint['input_size'],
        hidden_sizes=checkpoint['model_config']['hidden_sizes'],
        dropout_rate=checkpoint['model_config']['dropout_rate'],
        activation=checkpoint['model_config'].get('activation', 'relu'),
        batch_norm=checkpoint['model_config'].get('batch_norm', True)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load scaler
    scaler = None
    if os.path.exists(scaler_path):
        print(f"Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        print("Warning: No scaler found, assuming unscaled features")
    
    # Load feature columns
    print(f"Loading feature columns from: {feature_cols_path}")
    with open(feature_cols_path, 'rb') as f:
        feature_columns = pickle.load(f)
    
    return model, scaler, feature_columns, device


def prepare_features(df, feature_columns, scaler=None, use_new_scaler=True):
    """
    Prepare features for inference.
    
    Args:
        df: Input DataFrame
        feature_columns: List of expected feature columns
        scaler: Fitted StandardScaler (optional)
        use_new_scaler: If True, fit new scaler on inference data
        
    Returns:
        Processed feature tensor
    """
    # Ensure all required columns are present
    missing_cols = set(feature_columns) - set(df.columns)
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        for col in missing_cols:
            df[col] = np.nan
    
    # Select and order features
    X = df[feature_columns].copy()
    
    # Handle missing values - use median imputation
    X = X.fillna(X.median())
    
    # Apply scaling
    if use_new_scaler or scaler is None:
        print("Fitting new scaler on inference data...")
        from sklearn.preprocessing import StandardScaler
        inference_scaler = StandardScaler()
        X_scaled = inference_scaler.fit_transform(X)
        print(f"New scaler fitted with mean shape: {inference_scaler.mean_.shape}")
    else:
        print("Using saved scaler from training...")
        X_scaled = scaler.transform(X)
    
    # Convert to tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    return X_tensor


def predict(model, X_tensor, device, batch_size=256):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained neural network model
        X_tensor: Input tensor
        device: torch device
        batch_size: Batch size for inference
        
    Returns:
        predictions (probabilities), binary predictions
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Process in batches for memory efficiency
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            batch_pred = model(batch).cpu().numpy()
            predictions.extend(batch_pred)
    
    predictions = np.array(predictions).flatten()
    binary_predictions = (predictions > 0.5).astype(int)
    
    return predictions, binary_predictions


def predict_from_file(input_file, output_file=None, model_path=None, use_new_scaler=True):
    """
    Make predictions from a file containing patient data.
    
    Args:
        input_file: Path to input parquet/csv file
        output_file: Path to save predictions (optional)
        model_path: Path to model (optional, uses config if not provided)
        use_new_scaler: If True, fit new scaler on inference data (default: True)
        
    Returns:
        DataFrame with predictions
    """
    print(f"\n=== Neural Network Inference ===")
    print(f"Input file: {input_file}")
    
    # Load model and preprocessors
    model, scaler, feature_columns, device = load_model_and_preprocessors(model_path)
    
    # Load data
    if input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} records")
    
    # Prepare features
    X_tensor = prepare_features(df, feature_columns, scaler, use_new_scaler)
    
    # Make predictions
    print("Making predictions...")
    probabilities, binary_predictions = predict(model, X_tensor, device)
    
    # Add predictions to dataframe
    df['mortality_probability'] = probabilities
    df['mortality_prediction'] = binary_predictions
    
    # Calculate risk categories
    df['risk_category'] = pd.cut(
        df['mortality_probability'],
        bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Print summary statistics
    print(f"\nPrediction Summary:")
    print(f"Mean mortality probability: {probabilities.mean():.3f}")
    print(f"Predicted deaths: {binary_predictions.sum()} ({binary_predictions.mean()*100:.1f}%)")
    print(f"\nRisk distribution:")
    print(df['risk_category'].value_counts().sort_index())
    
    # Save predictions if output file specified
    if output_file:
        if output_file.endswith('.parquet'):
            df.to_parquet(output_file, index=False)
        else:
            df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to: {output_file}")
    
    return df


def predict_single_patient(patient_data, model_path=None, use_new_scaler=True):
    """
    Make prediction for a single patient.
    
    Args:
        patient_data: Dictionary or Series with patient features
        model_path: Path to model (optional)
        use_new_scaler: If True, fit new scaler on inference data (default: True)
        
    Returns:
        probability, prediction, risk_category
    """
    # Load model and preprocessors
    model, scaler, feature_columns, device = load_model_and_preprocessors(model_path)
    
    # Convert to DataFrame if needed
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    elif isinstance(patient_data, pd.Series):
        df = pd.DataFrame([patient_data])
    else:
        df = patient_data
    
    # Prepare features
    X_tensor = prepare_features(df, feature_columns, scaler, use_new_scaler)
    
    # Make prediction
    probability, binary_prediction = predict(model, X_tensor, device)
    
    probability = probability[0]
    binary_prediction = binary_prediction[0]
    
    # Determine risk category
    if probability < 0.1:
        risk_category = 'Very Low'
    elif probability < 0.3:
        risk_category = 'Low'
    elif probability < 0.5:
        risk_category = 'Medium'
    elif probability < 0.7:
        risk_category = 'High'
    else:
        risk_category = 'Very High'
    
    return {
        'mortality_probability': float(probability),
        'mortality_prediction': int(binary_prediction),
        'risk_category': risk_category
    }


def main():
    """Example usage of inference functions"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Neural Network ICU Mortality Prediction Inference')
    parser.add_argument('--input', type=str, required=True, help='Input file path (parquet or csv)')
    parser.add_argument('--output', type=str, help='Output file path (optional)')
    parser.add_argument('--model', type=str, help='Model path (optional, uses config default)')
    
    args = parser.parse_args()
    
    # Run inference
    predictions_df = predict_from_file(args.input, args.output, args.model)
    
    # Show sample predictions
    print("\nSample predictions (first 5 patients):")
    display_cols = ['hospitalization_id', 'mortality_probability', 'mortality_prediction', 'risk_category']
    available_cols = [col for col in display_cols if col in predictions_df.columns]
    print(predictions_df[available_cols].head())


if __name__ == "__main__":
    main()