#!/usr/bin/env python3
"""
Model loading utilities for Stage 2 testing.
Handles loading all models from BOX folder for cross-site testing and ensemble construction.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import sys

# Add stage1 helpers to path for model_io
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stage1', 'helpers'))
from model_io import ModelManager

class Stage2ModelLoader:
    """Loads and manages models for Stage 2 testing and ensemble construction."""
    
    def __init__(self, box_folder: str, current_site: str):
        """
        Initialize the model loader.
        
        Args:
            box_folder: Path to BOX folder containing shared models
            current_site: Name of current site (for exclusion in leave-one-out)
        """
        self.box_folder = box_folder
        self.current_site = current_site
        self.model_manager = ModelManager(box_folder, current_site)
        
    def discover_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Discover all models in BOX folder organized by approach and model type.
        
        Returns:
            Dictionary organized as {approach: {model_type: [model_info, ...]}}
        """
        all_models = self.model_manager.list_available_models()
        
        # Organize by approach and model type
        organized_models = {}
        
        for model_info in all_models:
            metadata = model_info['metadata']
            approach = metadata.get('approach', 'unknown')
            model_type = metadata.get('model_type', 'unknown')
            site_name = metadata.get('site_name', 'unknown')
            
            if approach not in organized_models:
                organized_models[approach] = {}
            
            if model_type not in organized_models[approach]:
                organized_models[approach][model_type] = []
            
            organized_models[approach][model_type].append({
                'path': model_info['path'],
                'site_name': site_name,
                'metadata': metadata
            })
        
        return organized_models
    
    def load_models_for_approach(self, 
                                approach: str, 
                                model_type: str = 'xgboost',
                                exclude_current_site: bool = True) -> List[Dict[str, Any]]:
        """
        Load all models for a specific approach and model type.
        
        Args:
            approach: Approach name (e.g., 'approach2_transfer_learning')
            model_type: Model type ('xgboost' or 'neural_network')
            exclude_current_site: Whether to exclude current site's model
        
        Returns:
            List of loaded models with metadata
        """
        discovered_models = self.discover_all_models()
        
        if approach not in discovered_models:
            logging.warning(f"No models found for approach: {approach}")
            return []
        
        if model_type not in discovered_models[approach]:
            logging.warning(f"No {model_type} models found for approach: {approach}")
            return []
        
        loaded_models = []
        
        for model_info in discovered_models[approach][model_type]:
            # Skip current site's model if requested
            if exclude_current_site and model_info['site_name'] == self.current_site:
                logging.info(f"Excluding current site's model: {model_info['site_name']}")
                continue
            
            try:
                if model_type == 'xgboost':
                    loaded_model = self.model_manager.load_xgboost_model(model_info['path'])
                else:  # neural_network
                    # Note: Requires model class to be provided separately
                    logging.warning("Neural network loading requires model class - skipping for now")
                    continue
                
                loaded_model['source_site'] = model_info['site_name']
                loaded_model['model_path'] = model_info['path']
                loaded_models.append(loaded_model)
                
                logging.info(f"Loaded {model_type} model from {model_info['site_name']}")
                
            except Exception as e:
                logging.error(f"Failed to load model from {model_info['path']}: {e}")
                continue
        
        logging.info(f"Successfully loaded {len(loaded_models)} {model_type} models for {approach}")
        return loaded_models
    
    def load_all_models_for_ensemble(self, 
                                   model_type: str = 'xgboost',
                                   approaches: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all models from multiple approaches for ensemble construction.
        Automatically excludes current site's models.
        
        Args:
            model_type: Model type to load
            approaches: List of approaches to include (default: all available)
        
        Returns:
            Dictionary with approach names as keys and loaded models as values
        """
        if approaches is None:
            # Default approaches from Stage 1
            approaches = [
                'approach2_transfer_learning',
                'approach3_independent',
                'approach4_round_robin'
            ]
        
        ensemble_models = {}
        
        for approach in approaches:
            models = self.load_models_for_approach(
                approach=approach,
                model_type=model_type,
                exclude_current_site=True  # Always exclude for ensemble
            )
            
            if models:
                ensemble_models[approach] = models
        
        total_models = sum(len(models) for models in ensemble_models.values())
        logging.info(f"Loaded {total_models} models total for ensemble from {len(ensemble_models)} approaches")
        
        return ensemble_models
    
    def get_model_performance_summary(self, loaded_models: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Get performance summary of loaded models.
        
        Args:
            loaded_models: List of loaded models
        
        Returns:
            DataFrame with model performance summary
        """
        import pandas as pd
        
        summary_data = []
        
        for model_info in loaded_models:
            metadata = model_info['metadata']
            metrics = metadata.get('metrics', {})
            
            summary_data.append({
                'source_site': model_info['source_site'],
                'approach': metadata.get('approach', 'unknown'),
                'model_type': metadata.get('model_type', 'unknown'),
                'timestamp': metadata.get('timestamp', 'unknown'),
                'auc_roc': metrics.get('auc_roc', None),
                'f1_score': metrics.get('f1_score', None),
                'precision': metrics.get('precision', None),
                'recall': metrics.get('recall', None)
            })
        
        return pd.DataFrame(summary_data)
    
    def validate_model_compatibility(self, loaded_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate that loaded models are compatible for ensemble.
        
        Args:
            loaded_models: List of loaded models
        
        Returns:
            Dictionary with validation results
        """
        if not loaded_models:
            return {'status': 'error', 'message': 'No models to validate'}
        
        # Check feature compatibility
        feature_sets = []
        for model_info in loaded_models:
            features = model_info.get('feature_columns', [])
            feature_sets.append(set(features))
        
        # Find common features
        common_features = feature_sets[0]
        for feature_set in feature_sets[1:]:
            common_features = common_features.intersection(feature_set)
        
        # Check if all models have the same features
        all_same = all(len(fs) == len(common_features) and fs == common_features for fs in feature_sets)
        
        validation_result = {
            'status': 'valid' if all_same else 'warning',
            'total_models': len(loaded_models),
            'common_features_count': len(common_features),
            'all_features_identical': all_same,
            'common_features': list(common_features),
        }
        
        if not all_same:
            validation_result['message'] = 'Models have different feature sets - ensemble may need feature alignment'
            # Report feature differences
            all_features = set()
            for fs in feature_sets:
                all_features = all_features.union(fs)
            
            validation_result['missing_features_per_model'] = []
            for i, model_info in enumerate(loaded_models):
                features = set(model_info.get('feature_columns', []))
                missing = all_features - features
                validation_result['missing_features_per_model'].append({
                    'site': model_info['source_site'],
                    'missing_features': list(missing)
                })
        
        return validation_result

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize loader
    loader = Stage2ModelLoader(
        box_folder="./test_box",
        current_site="test_site"
    )
    
    # Discover models
    all_models = loader.discover_all_models()
    print(f"Discovered models: {all_models}")
    
    # Load models for ensemble (example)
    ensemble_models = loader.load_all_models_for_ensemble(model_type='xgboost')
    print(f"Loaded ensemble models from {len(ensemble_models)} approaches")