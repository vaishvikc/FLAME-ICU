#!/usr/bin/env python3
"""
Ensemble utilities for Stage 2 testing.
Implements leave-one-out ensemble construction and testing logic.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
import sys

# Add stage1 helpers to path for metrics
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stage1', 'helpers'))
from metrics import calculate_standard_metrics, generate_performance_report

class EnsembleBuilder:
    """Builds and evaluates ensemble models using different strategies."""
    
    def __init__(self, current_site: str):
        """
        Initialize ensemble builder.
        
        Args:
            current_site: Name of current site (excluded from ensembles)
        """
        self.current_site = current_site
    
    def simple_average_ensemble(self, 
                               models: List[Dict[str, Any]], 
                               X_test: np.ndarray) -> np.ndarray:
        """
        Create simple average ensemble predictions.
        All models get equal weight.
        
        Args:
            models: List of loaded models
            X_test: Test features
        
        Returns:
            Ensemble predictions (probabilities)
        """
        if not models:
            raise ValueError("No models provided for ensemble")
        
        all_predictions = []
        
        for model_info in models:
            try:
                # Get model components
                model = model_info['model']
                scaler = model_info['scaler']
                feature_columns = model_info['feature_columns']
                
                # Prepare features (assuming X_test is DataFrame with proper columns)
                if hasattr(X_test, 'columns'):
                    # Align features
                    aligned_features = X_test[feature_columns]
                else:
                    # Assume X_test is already aligned
                    aligned_features = X_test
                
                # Scale features
                X_scaled = scaler.transform(aligned_features)
                
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    # For scikit-learn style models
                    predictions = model.predict_proba(X_scaled)[:, 1]
                elif hasattr(model, 'predict'):
                    # For XGBoost
                    import xgboost as xgb
                    if isinstance(model, xgb.Booster):
                        dtest = xgb.DMatrix(X_scaled)
                        predictions = model.predict(dtest)
                    else:
                        predictions = model.predict(X_scaled)
                else:
                    raise ValueError(f"Unknown model type: {type(model)}")
                
                all_predictions.append(predictions)
                logging.debug(f"Added predictions from {model_info.get('source_site', 'unknown')}")
                
            except Exception as e:
                logging.warning(f"Failed to get predictions from model {model_info.get('source_site', 'unknown')}: {e}")
                continue
        
        if not all_predictions:
            raise ValueError("No valid predictions obtained from any model")
        
        # Simple average
        ensemble_predictions = np.mean(all_predictions, axis=0)
        
        logging.info(f"Simple average ensemble created from {len(all_predictions)} models")
        return ensemble_predictions
    
    def accuracy_weighted_ensemble(self, 
                                  models: List[Dict[str, Any]], 
                                  X_test: np.ndarray,
                                  local_performance: Dict[str, float]) -> np.ndarray:
        """
        Create accuracy-weighted ensemble predictions.
        Models are weighted by their local AUC performance.
        
        Args:
            models: List of loaded models
            X_test: Test features
            local_performance: Dictionary mapping site names to AUC scores
        
        Returns:
            Ensemble predictions (probabilities)
        """
        if not models:
            raise ValueError("No models provided for ensemble")
        
        all_predictions = []
        weights = []
        
        for model_info in models:
            try:
                source_site = model_info.get('source_site', 'unknown')
                
                # Get model performance weight
                if source_site in local_performance:
                    weight = local_performance[source_site]
                else:
                    # Use default weight if performance not available
                    weight = 0.5  # Neutral weight
                    logging.warning(f"No performance data for {source_site}, using default weight {weight}")
                
                # Get model components
                model = model_info['model']
                scaler = model_info['scaler']
                feature_columns = model_info['feature_columns']
                
                # Prepare features
                if hasattr(X_test, 'columns'):
                    aligned_features = X_test[feature_columns]
                else:
                    aligned_features = X_test
                
                # Scale features
                X_scaled = scaler.transform(aligned_features)
                
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    predictions = model.predict_proba(X_scaled)[:, 1]
                elif hasattr(model, 'predict'):
                    import xgboost as xgb
                    if isinstance(model, xgb.Booster):
                        dtest = xgb.DMatrix(X_scaled)
                        predictions = model.predict(dtest)
                    else:
                        predictions = model.predict(X_scaled)
                else:
                    raise ValueError(f"Unknown model type: {type(model)}")
                
                all_predictions.append(predictions)
                weights.append(weight)
                
                logging.debug(f"Added predictions from {source_site} with weight {weight}")
                
            except Exception as e:
                logging.warning(f"Failed to get predictions from model {model_info.get('source_site', 'unknown')}: {e}")
                continue
        
        if not all_predictions:
            raise ValueError("No valid predictions obtained from any model")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted average
        ensemble_predictions = np.average(all_predictions, axis=0, weights=weights)
        
        logging.info(f"Accuracy-weighted ensemble created from {len(all_predictions)} models")
        logging.info(f"Weights used: {dict(zip([m.get('source_site', 'unknown') for m in models[:len(weights)]], weights))}")
        
        return ensemble_predictions
    
    def evaluate_ensemble(self, 
                         ensemble_predictions: np.ndarray,
                         y_true: np.ndarray,
                         ensemble_type: str,
                         individual_models_performance: List[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Evaluate ensemble performance and compare with individual models.
        
        Args:
            ensemble_predictions: Ensemble prediction probabilities
            y_true: True labels
            ensemble_type: Type of ensemble ('simple_average' or 'accuracy_weighted')
            individual_models_performance: List of individual model performance metrics
        
        Returns:
            Comprehensive evaluation results
        """
        # Calculate ensemble metrics
        ensemble_metrics = calculate_standard_metrics(y_true, ensemble_predictions)
        
        # Generate full performance report
        ensemble_report = generate_performance_report(
            y_true, ensemble_predictions,
            model_name=f"Ensemble_{ensemble_type}",
            approach="stage2_ensemble"
        )
        
        evaluation = {
            'ensemble_type': ensemble_type,
            'ensemble_metrics': ensemble_metrics,
            'ensemble_report': ensemble_report,
            'site_name': self.current_site
        }
        
        # Compare with individual models if provided
        if individual_models_performance:
            ensemble_auc = ensemble_metrics['auc_roc']
            
            # Find best individual model
            best_individual = max(individual_models_performance, key=lambda x: x.get('auc_roc', 0))
            best_individual_auc = best_individual.get('auc_roc', 0)
            
            # Calculate improvement
            improvement = ensemble_auc - best_individual_auc
            improvement_percentage = (improvement / best_individual_auc) * 100 if best_individual_auc > 0 else 0
            
            evaluation['comparison'] = {
                'best_individual_auc': best_individual_auc,
                'ensemble_auc': ensemble_auc,
                'improvement': improvement,
                'improvement_percentage': improvement_percentage,
                'ensemble_better': improvement > 0
            }
            
            # Statistical significance test (if needed)
            evaluation['individual_models_count'] = len(individual_models_performance)
        
        return evaluation
    
    def leave_one_out_testing(self, 
                            all_models: Dict[str, List[Dict[str, Any]]],
                            X_test: np.ndarray,
                            y_true: np.ndarray,
                            local_performance: Dict[str, Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Perform leave-one-out ensemble testing.
        Creates ensembles excluding current site's models and evaluates on local test data.
        
        Args:
            all_models: Dictionary of models organized by approach
            X_test: Local test features
            y_true: Local test labels
            local_performance: Local performance metrics for weighting
        
        Returns:
            Complete leave-one-out testing results
        """
        results = {
            'site_name': self.current_site,
            'total_models_available': sum(len(models) for models in all_models.values()),
            'approaches_included': list(all_models.keys()),
            'ensembles': {}
        }
        
        # Flatten all models (already excluding current site)
        all_models_flat = []
        for approach, models in all_models.items():
            all_models_flat.extend(models)
        
        if not all_models_flat:
            results['error'] = 'No models available for ensemble (all excluded due to leave-one-out)'
            return results
        
        logging.info(f"Creating ensembles from {len(all_models_flat)} models (current site excluded)")
        
        # Test individual models first for comparison
        individual_performance = []
        for model_info in all_models_flat:
            try:
                # Get single model predictions
                single_model_predictions = self.simple_average_ensemble([model_info], X_test)
                single_metrics = calculate_standard_metrics(y_true, single_model_predictions)
                single_metrics['source_site'] = model_info.get('source_site', 'unknown')
                individual_performance.append(single_metrics)
            except Exception as e:
                logging.warning(f"Failed to evaluate individual model from {model_info.get('source_site', 'unknown')}: {e}")
        
        results['individual_models_performance'] = individual_performance
        
        # Simple Average Ensemble
        try:
            simple_predictions = self.simple_average_ensemble(all_models_flat, X_test)
            simple_evaluation = self.evaluate_ensemble(
                simple_predictions, y_true, 'simple_average', individual_performance
            )
            results['ensembles']['simple_average'] = simple_evaluation
        except Exception as e:
            results['ensembles']['simple_average'] = {'error': str(e)}
            logging.error(f"Failed to create simple average ensemble: {e}")
        
        # Accuracy Weighted Ensemble
        if local_performance:
            try:
                # Flatten local performance (assuming it's organized by site)
                flat_performance = {}
                for site_perf in local_performance.values():
                    if isinstance(site_perf, dict):
                        flat_performance.update(site_perf)
                    else:
                        # If it's already flat
                        flat_performance = local_performance
                        break
                
                weighted_predictions = self.accuracy_weighted_ensemble(
                    all_models_flat, X_test, flat_performance
                )
                weighted_evaluation = self.evaluate_ensemble(
                    weighted_predictions, y_true, 'accuracy_weighted', individual_performance
                )
                results['ensembles']['accuracy_weighted'] = weighted_evaluation
            except Exception as e:
                results['ensembles']['accuracy_weighted'] = {'error': str(e)}
                logging.error(f"Failed to create accuracy weighted ensemble: {e}")
        else:
            results['ensembles']['accuracy_weighted'] = {
                'error': 'No local performance data provided for weighting'
            }
        
        return results

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize ensemble builder
    ensemble_builder = EnsembleBuilder(current_site="test_site")
    
    # Example with synthetic data
    np.random.seed(42)
    X_test = np.random.randn(100, 10)
    y_true = np.random.choice([0, 1], 100)
    
    print("Ensemble utilities ready for use")
    print(f"Current site: {ensemble_builder.current_site}")