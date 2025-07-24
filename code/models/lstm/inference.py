import pandas as pd
import numpy as np
import os
import json
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Define the same LSTM model architecture used in training
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, output_size=1, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size2, 16)
        self.fc2 = nn.Linear(16, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # First LSTM layer - output is (batch_size, seq_len, hidden_size1)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        # Second LSTM layer - we only need the output from the last time step
        # output is (batch_size, seq_len, hidden_size2)
        x, _ = self.lstm2(x)
        # Get the last time step and apply dropout
        x = self.dropout2(x[:, -1, :])  # Shape becomes (batch_size, hidden_size2)
        # Dense layers
        x = self.relu(self.fc1(x))
        # Raw logits for BCEWithLogitsLoss
        x = self.fc2(x)
        return x

def prepare_sequences(df, feature_cols, sequence_length=24):
    """
    Transform the data into sequences for each hospitalization_id
    Apply padding for hospitalization_ids with fewer than sequence_length hours
    Handle missing values by filling with zeros
    """
    # Dictionary to store sequences and targets
    sequences = {}
    targets = {}
    
    # Loop through each hospitalization_id
    for hosp_id, group in df.groupby('hospitalization_id'):
        # Sort by hour
        group = group.sort_values('nth_hour')
        
        # Create sequence from features
        seq = group[feature_cols].values
        
        # Handle NaN values by replacing with zeros
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get the target if available (for evaluation)
        if 'disposition' in df.columns:
            target = group['disposition'].iloc[0]
            targets[hosp_id] = target
        else:
            targets[hosp_id] = None
        
        # Pad if needed (if less than sequence_length hours)
        if len(seq) < sequence_length:
            # Create padding (zeros)
            padding = np.zeros((sequence_length - len(seq), len(feature_cols)))
            seq = np.vstack([seq, padding])
        # Truncate if more than sequence_length hours
        elif len(seq) > sequence_length:
            seq = seq[:sequence_length]
            
        # Store sequence
        sequences[hosp_id] = seq
    
    return sequences, targets

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

def load_model_and_metadata():
    """Load saved model, scaler, and feature columns"""
    config = load_config()
    output_config = config['output_config'].copy()
    model_params = config['model_params']
    data_config = config['data_config']
    
    # Load preprocessing config to get site name
    preprocessing_config = load_preprocessing_config()
    site_name = preprocessing_config.get('site', 'unknown')
    print(f"Loading model for site: {site_name}")
    
    # Update output paths with site name
    for key in output_config:
        if isinstance(output_config[key], str):
            # Replace the model name with site-specific name
            output_config[key] = output_config[key].replace(
                'lstm_icu_mortality_model', f'lstm_{site_name}_icu_mortality_model'
            ).replace(
                'feature_scaler', f'{site_name}_feature_scaler'
            ).replace(
                'feature_columns', f'{site_name}_feature_columns'
            )
    
    model_path = output_config['model_path']
    scaler_path = output_config['scaler_path']
    feature_cols_path = output_config['feature_cols_path']
    
    # Resolve paths if they are relative
    if not os.path.isabs(model_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, model_path)
        scaler_path = os.path.join(base_dir, output_config['scaler_path'])
        feature_cols_path = os.path.join(base_dir, output_config['feature_cols_path'])
    
    # Load feature columns
    with open(feature_cols_path, 'rb') as f:
        feature_cols = pickle.load(f)
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Initialize model with correct input size
    input_size = len(feature_cols)
    model = LSTMModel(
        input_size=input_size,
        hidden_size1=model_params['hidden_size1'],
        hidden_size2=model_params['hidden_size2'],
        dropout_rate=model_params['dropout_rate']
    )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    
    return model, scaler, feature_cols, data_config['sequence_length']

def predict_mortality(data_path=None):
    """
    Predict ICU mortality using the trained LSTM model
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the data file (parquet format). If None, uses the path from config.
    
    Returns:
    --------
    predictions : dict
        Dictionary mapping hospitalization_id to predicted mortality probability
    """
    # Load model and metadata
    model, scaler, feature_cols, sequence_length = load_model_and_metadata()
    
    # Get data path from config if not provided
    if data_path is None:
        config = load_config()
        data_config = config['data_config']
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(base_dir, data_config['preprocessing_path'], data_config['feature_file'])
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Data shape: {df.shape}")
    
    # Check for required columns
    required_cols = ['hospitalization_id', 'nth_hour'] + feature_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        print("Available columns:", df.columns.tolist())
        return None
    
    # Prepare sequences
    print("Preparing sequences...")
    sequences, targets = prepare_sequences(df, feature_cols, sequence_length)
    
    # Convert to numpy arrays
    X = np.array(list(sequences.values()))
    hospitalization_ids = list(sequences.keys())
    
    # Reshape to 2D for scaling
    X_reshape = X.reshape(-1, X.shape[-1])
    
    # Scale features
    X_scaled = scaler.transform(X_reshape)
    
    # Reshape back to 3D
    X_scaled = X_scaled.reshape(X.shape)
    
    # Convert to PyTorch tensor
    X_tensor = torch.FloatTensor(X_scaled)
    
    # Make predictions
    print("Making predictions...")
    predictions = {}
    
    with torch.no_grad():
        outputs = model(X_tensor)
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(outputs)
        # Handle any potential NaN values
        probs = torch.nan_to_num(probs, nan=0.5)
    
    # Convert to numpy and store in dictionary
    probs = probs.cpu().numpy()
    
    for i, hosp_id in enumerate(hospitalization_ids):
        predictions[hosp_id] = {
            'mortality_probability': float(probs[i][0]),
            'predicted_class': 1 if probs[i][0] > 0.5 else 0,
            'actual_class': targets[hosp_id]  # Will be None if disposition not in data
        }
    
    return predictions

def main():
    """Main function to run inference"""
    # Make predictions
    predictions = predict_mortality()
    
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
        
    print("Inference completed!")

if __name__ == "__main__":
    main()