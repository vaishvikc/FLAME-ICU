#!/usr/bin/env python3
"""
Model I/O utilities for FLAME-ICU federated learning.
Handles saving and loading models for BOX folder sharing.
"""

import os
import pickle
import json
import torch
import xgboost as xgb
from datetime import datetime
from typing import Dict, Any, Optional, Union
import logging

class ModelManager:
    """Manages model saving/loading for federated learning."""
    
    def __init__(self, box_folder: str = None, site_name: str = None):
        """
        Initialize ModelManager.
        
        Args:
            box_folder: Path to BOX folder for model sharing
            site_name: Name of the current site
        """
        self.box_folder = box_folder or "/path/to/box/folder"  # Update with actual BOX path
        self.site_name = site_name or "unknown_site"
        
        # Create directories if they don't exist
        os.makedirs(self.box_folder, exist_ok=True)
    
    def save_xgboost_model(self, model: xgb.Booster, 
                          scaler: Any, 
                          feature_columns: list,
                          approach: str,
                          metrics: Dict[str, float] = None) -> str:
        """
        Save XGBoost model with associated artifacts.
        
        Args:
            model: Trained XGBoost model
            scaler: Feature scaler
            feature_columns: List of feature column names
            approach: Approach name (e.g., 'approach2_transfer', 'approach3_independent')
            metrics: Model performance metrics
        
        Returns:
            Path to saved model directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.box_folder, f"{self.site_name}_{approach}_xgboost_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.json")
        model.save_model(model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature columns
        features_path = os.path.join(model_dir, "features.pkl")
        with open(features_path, 'wb') as f:
            pickle.dump(feature_columns, f)
        
        # Save metadata
        metadata = {
            'site_name': self.site_name,
            'approach': approach,
            'model_type': 'xgboost',
            'timestamp': timestamp,
            'feature_count': len(feature_columns),
            'metrics': metrics or {}
        }
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"XGBoost model saved to: {model_dir}")
        return model_dir
    
    def save_nn_model(self, model: torch.nn.Module,
                     scaler: Any,
                     feature_columns: list,
                     approach: str,
                     metrics: Dict[str, float] = None) -> str:
        """
        Save Neural Network model with associated artifacts.
        
        Args:
            model: Trained PyTorch model
            scaler: Feature scaler
            feature_columns: List of feature column names
            approach: Approach name
            metrics: Model performance metrics
        
        Returns:
            Path to saved model directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.box_folder, f"{self.site_name}_{approach}_nn_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        
        # Save model architecture info
        arch_info = {
            'model_class': model.__class__.__name__,
            'input_size': getattr(model, 'input_size', None),
            'hidden_layers': getattr(model, 'hidden_layers', None),
            'output_size': getattr(model, 'output_size', None)
        }
        
        arch_path = os.path.join(model_dir, "architecture.json")
        with open(arch_path, 'w') as f:
            json.dump(arch_info, f, indent=2)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature columns
        features_path = os.path.join(model_dir, "features.pkl")
        with open(features_path, 'wb') as f:
            pickle.dump(feature_columns, f)
        
        # Save metadata
        metadata = {
            'site_name': self.site_name,
            'approach': approach,
            'model_type': 'neural_network',
            'timestamp': timestamp,
            'feature_count': len(feature_columns),
            'metrics': metrics or {}
        }
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Neural Network model saved to: {model_dir}")
        return model_dir
    
    def load_xgboost_model(self, model_dir: str) -> Dict[str, Any]:
        """
        Load XGBoost model and associated artifacts.
        
        Args:
            model_dir: Path to model directory
        
        Returns:
            Dictionary with model, scaler, features, and metadata
        """
        # Load model
        model_path = os.path.join(model_dir, "model.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = xgb.Booster()
        model.load_model(model_path)
        
        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load feature columns
        features_path = os.path.join(model_dir, "features.pkl")
        with open(features_path, 'rb') as f:
            feature_columns = pickle.load(f)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'metadata': metadata
        }
    
    def load_nn_model(self, model_dir: str, model_class: torch.nn.Module = None) -> Dict[str, Any]:
        """
        Load Neural Network model and associated artifacts.
        
        Args:
            model_dir: Path to model directory
            model_class: PyTorch model class for reconstruction
        
        Returns:
            Dictionary with model, scaler, features, and metadata
        """
        # Load metadata and architecture
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        arch_path = os.path.join(model_dir, "architecture.json")
        with open(arch_path, 'r') as f:
            arch_info = json.load(f)
        
        # Load model state dict
        model_path = os.path.join(model_dir, "model.pth")
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Create model instance (requires model_class to be provided)
        if model_class is None:
            raise ValueError("model_class must be provided to load Neural Network model")
        
        model = model_class()
        model.load_state_dict(state_dict)
        
        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load feature columns
        features_path = os.path.join(model_dir, "features.pkl")
        with open(features_path, 'rb') as f:
            feature_columns = pickle.load(f)
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'metadata': metadata,
            'architecture': arch_info
        }
    
    def list_available_models(self, approach: str = None, model_type: str = None) -> list:
        """
        List available models in BOX folder.
        
        Args:
            approach: Filter by approach (optional)
            model_type: Filter by model type ('xgboost', 'neural_network') (optional)
        
        Returns:
            List of model directories matching criteria
        """
        if not os.path.exists(self.box_folder):
            return []
        
        models = []
        for item in os.listdir(self.box_folder):
            item_path = os.path.join(self.box_folder, item)
            if os.path.isdir(item_path):
                # Check if it's a model directory
                metadata_path = os.path.join(item_path, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Apply filters
                        if approach and metadata.get('approach') != approach:
                            continue
                        if model_type and metadata.get('model_type') != model_type:
                            continue
                        
                        models.append({
                            'path': item_path,
                            'metadata': metadata
                        })
                    except Exception as e:
                        logging.warning(f"Could not read metadata for {item}: {e}")
        
        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x['metadata'].get('timestamp', ''), reverse=True)
        return models

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize manager
    manager = ModelManager(box_folder="./test_box", site_name="test_site")
    
    # Example: Save a dummy XGBoost model (requires actual model for real usage)
    print("Model I/O utilities ready for use")
    print(f"BOX folder: {manager.box_folder}")
    print(f"Site name: {manager.site_name}")