import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Define the same LSTM model architecture used in training
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        
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

def prepare_sequences(df, feature_cols):
    """
    Transform the data into 24-hour sequences for each hospitalization_id
    Apply padding for hospitalization_ids with fewer than 24 hours
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
        
        # Pad if needed (if less than 24 hours)
        if len(seq) < 24:
            # Create padding (zeros)
            padding = np.zeros((24 - len(seq), len(feature_cols)))
            seq = np.vstack([seq, padding])
        # Truncate if more than 24 hours
        elif len(seq) > 24:
            seq = seq[:24]
            
        # Store sequence
        sequences[hosp_id] = seq
    
    return sequences, targets

def load_model_and_metadata(model_dir):
    """Load saved model, scaler, and feature columns"""
    model_path = os.path.join(model_dir, 'lstm_icu_mortality_model.pt')
    scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
    feature_cols_path = os.path.join(model_dir, 'feature_columns.pkl')
    
    # Load feature columns
    with open(feature_cols_path, 'rb') as f:
        feature_cols = pickle.load(f)
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Initialize model with correct input size
    input_size = len(feature_cols)
    model = LSTMModel(input_size=input_size)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    
    return model, scaler, feature_cols

def predict_mortality(data_path, model_dir):
    """
    Predict ICU mortality using the trained LSTM model
    
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
    model, scaler, feature_cols = load_model_and_metadata(model_dir)
    
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
    sequences, targets = prepare_sequences(df, feature_cols)
    
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
    # Get paths
    code_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(code_dir)
    model_dir = os.path.join(project_dir, 'model')
    
    # For demonstration, using the original data file
    data_path = os.path.join(project_dir, 'code', 'output', 'intermitted', 'by_hourly_wide_df.parquet')
    
    # Check if model exists
    if not os.path.exists(os.path.join(model_dir, 'lstm_icu_mortality_model.pt')):
        print(f"Error: Model file not found in {model_dir}")
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
        
    print("Inference completed!")

if __name__ == "__main__":
    main()
