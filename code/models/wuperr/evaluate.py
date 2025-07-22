#!/usr/bin/env python3
"""
Evaluation script for WUPERR federated learning results.
Analyzes training progress, site contributions, and model performance.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import WUPERR components
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .git_model_manager import GitModelManager
from .sequential_wuperr import SequentialWUPERR

# Import LSTM model
from .sequential_train import LSTMModel, prepare_sequences, create_data_loaders

class WUPERRAnalyzer:
    """Analyze WUPERR federated learning results."""
    
    def __init__(self, output_dir: str = "results"):
        """Initialize analyzer."""
        self.output_dir = output_dir
        self.graphs_dir = os.path.join("..", "output", "final", "graphs")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)
        
        # Initialize Git manager
        self.git_manager = GitModelManager()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_federated_data(self):
        """Load federated learning data and results."""
        logger.info("Loading federated learning data...")
        
        # Load site contributions
        contributions = self.git_manager.load_site_contributions()
        if not contributions:
            logger.error("No site contributions found")
            return None
        
        # Load site metadata
        site_metadata_path = os.path.join("data", "sites", "site_metadata.json")
        site_metadata = {}
        if os.path.exists(site_metadata_path):
            with open(site_metadata_path, 'r') as f:
                site_metadata = json.load(f)
        
        # Load training results
        training_results_path = os.path.join("data", "sites", "training_results.json")
        training_results = {}
        if os.path.exists(training_results_path):
            with open(training_results_path, 'r') as f:
                training_results = json.load(f)
        
        return {
            'contributions': contributions,
            'site_metadata': site_metadata,
            'training_results': training_results
        }
    
    def analyze_training_progression(self, data: dict):
        """Analyze training progression across sites and rounds."""
        logger.info("Analyzing training progression...")
        
        contributions = data['contributions']
        
        # Extract metrics across all sites and rounds
        progression_data = []
        
        for site_key, site_data in contributions.items():
            site_id = int(site_key.split('_')[1])
            
            for round_key, round_data in site_data.items():
                round_num = int(round_key.split('_')[1])
                metrics = round_data['metrics']
                
                progression_data.append({
                    'site_id': site_id,
                    'round_num': round_num,
                    'accuracy': metrics.get('accuracy', 0),
                    'auc': metrics.get('auc', 0),
                    'loss': metrics.get('val_loss', 0),
                    'train_loss': metrics.get('train_loss', 0),
                    'timestamp': round_data['timestamp']
                })
        
        df = pd.DataFrame(progression_data)
        
        # Create progression plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # AUC progression
        axes[0, 0].set_title('AUC Progression by Site')
        for site_id in df['site_id'].unique():
            site_data = df[df['site_id'] == site_id]
            axes[0, 0].plot(site_data['round_num'], site_data['auc'], 
                           marker='o', label=f'Site {site_id}')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('AUC')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy progression
        axes[0, 1].set_title('Accuracy Progression by Site')
        for site_id in df['site_id'].unique():
            site_data = df[df['site_id'] == site_id]
            axes[0, 1].plot(site_data['round_num'], site_data['accuracy'], 
                           marker='o', label=f'Site {site_id}')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss progression
        axes[1, 0].set_title('Validation Loss Progression by Site')
        for site_id in df['site_id'].unique():
            site_data = df[df['site_id'] == site_id]
            axes[1, 0].plot(site_data['round_num'], site_data['loss'], 
                           marker='o', label=f'Site {site_id}')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Validation Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Overall improvement heatmap
        pivot_auc = df.pivot(index='site_id', columns='round_num', values='auc')
        sns.heatmap(pivot_auc, annot=True, fmt='.3f', cmap='viridis', 
                   ax=axes[1, 1], cbar_kws={'label': 'AUC'})
        axes[1, 1].set_title('AUC Heatmap (Site x Round)')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Site ID')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, 'wuperr_training_progression.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return df
    
    def analyze_site_contributions(self, data: dict):
        """Analyze individual site contributions."""
        logger.info("Analyzing site contributions...")
        
        contributions = data['contributions']
        site_metadata = data['site_metadata']
        
        # Calculate site statistics
        site_stats = []
        
        for site_key, site_data in contributions.items():
            site_id = int(site_key.split('_')[1])
            
            # Get site metadata
            site_info = site_metadata.get(site_key, {})
            
            # Calculate improvement metrics
            aucs = []
            for round_data in site_data.values():
                aucs.append(round_data['metrics'].get('auc', 0))
            
            site_stats.append({
                'site_id': site_id,
                'hospital_type': site_info.get('hospital_type', 'unknown'),
                'num_samples': site_info.get('num_samples', 0),
                'mortality_rate': site_info.get('mortality_rate', 0),
                'rounds_completed': len(site_data),
                'best_auc': max(aucs) if aucs else 0,
                'final_auc': aucs[-1] if aucs else 0,
                'auc_improvement': aucs[-1] - aucs[0] if len(aucs) > 1 else 0,
                'avg_auc': np.mean(aucs) if aucs else 0
            })
        
        df_stats = pd.DataFrame(site_stats)
        
        # Create contribution analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Site performance vs data size
        axes[0, 0].scatter(df_stats['num_samples'], df_stats['best_auc'], 
                          c=df_stats['site_id'], cmap='tab10', s=100)
        axes[0, 0].set_xlabel('Number of Training Samples')
        axes[0, 0].set_ylabel('Best AUC')
        axes[0, 0].set_title('Site Performance vs Data Size')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add site ID labels
        for i, row in df_stats.iterrows():
            axes[0, 0].annotate(f'Site {row["site_id"]}', 
                              (row['num_samples'], row['best_auc']),
                              xytext=(5, 5), textcoords='offset points')
        
        # AUC improvement by site
        axes[0, 1].bar(df_stats['site_id'], df_stats['auc_improvement'], 
                      color=plt.cm.viridis(df_stats['site_id'] / df_stats['site_id'].max()))
        axes[0, 1].set_xlabel('Site ID')
        axes[0, 1].set_ylabel('AUC Improvement')
        axes[0, 1].set_title('AUC Improvement by Site')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Hospital type performance
        type_performance = df_stats.groupby('hospital_type')['best_auc'].mean().sort_values(ascending=False)
        axes[1, 0].bar(range(len(type_performance)), type_performance.values)
        axes[1, 0].set_xticks(range(len(type_performance)))
        axes[1, 0].set_xticklabels(type_performance.index, rotation=45)
        axes[1, 0].set_ylabel('Average Best AUC')
        axes[1, 0].set_title('Performance by Hospital Type')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Mortality rate vs performance
        axes[1, 1].scatter(df_stats['mortality_rate'], df_stats['best_auc'], 
                          c=df_stats['site_id'], cmap='tab10', s=100)
        axes[1, 1].set_xlabel('Site Mortality Rate')
        axes[1, 1].set_ylabel('Best AUC')
        axes[1, 1].set_title('Mortality Rate vs Performance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, 'wuperr_site_contributions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistics
        df_stats.to_csv(os.path.join(self.output_dir, 'site_statistics.csv'), index=False)
        
        return df_stats
    
    def evaluate_final_model(self, global_test_data_path: str = None):
        """Evaluate the final federated model."""
        logger.info("Evaluating final federated model...")
        
        # Load final model
        model = self.git_manager.load_model(LSTMModel)
        if model is None:
            logger.error("Could not load final model")
            return None
        
        # Load metadata
        metadata = self.git_manager.load_metadata()
        if metadata is None:
            logger.error("Could not load model metadata")
            return None
        
        # If global test data is provided, use it; otherwise use original data
        if global_test_data_path and os.path.exists(global_test_data_path):
            logger.info(f"Using global test data: {global_test_data_path}")
            test_df = pd.read_parquet(global_test_data_path)
        else:
            logger.info("Using original dataset for evaluation")
            test_df = pd.read_parquet('../../../output/intermitted/by_hourly_wide_df.parquet')
        
        # Prepare test data
        feature_cols = metadata['feature_columns']
        sequences, targets = prepare_sequences(test_df, feature_cols)
        
        X_test = np.array(list(sequences.values()))
        y_test = np.array(list(targets.values()))
        
        # Create test data loader
        _, test_loader, _ = create_data_loaders(X_test, y_test, test_size=0.0)
        
        # Evaluate model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        y_pred_proba = []
        y_true = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                probs = torch.sigmoid(output)
                y_pred_proba.extend(probs.cpu().numpy().flatten())
                y_true.extend(target.cpu().numpy().flatten())
        
        y_pred_proba = np.array(y_pred_proba)
        y_true = np.array(y_true)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'num_samples': len(y_true),
            'num_positive': sum(y_true),
            'num_negative': len(y_true) - sum(y_true)
        }
        
        # Create evaluation plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        axes[0, 0].plot(fpr, tpr, label=f'ROC (AUC = {metrics["auc"]:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve - Final Federated Model')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        axes[0, 1].plot(recall, precision)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Prediction distribution
        axes[1, 1].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Negative', density=True)
        axes[1, 1].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Positive', density=True)
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Prediction Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, 'wuperr_final_model_evaluation.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics
        with open(os.path.join(self.output_dir, 'final_model_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Final model evaluation completed:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  AUC: {metrics['auc']:.4f}")
        logger.info(f"  Test samples: {metrics['num_samples']}")
        
        return metrics
    
    def generate_report(self, data: dict, progression_df: pd.DataFrame, 
                       site_stats: pd.DataFrame, final_metrics: dict):
        """Generate comprehensive evaluation report."""
        logger.info("Generating comprehensive evaluation report...")
        
        report = {
            'experiment_info': {
                'total_sites': len(data['contributions']),
                'total_rounds': progression_df['round_num'].max() if not progression_df.empty else 0,
                'evaluation_timestamp': datetime.now().isoformat()
            },
            'federated_performance': {
                'best_overall_auc': progression_df['auc'].max() if not progression_df.empty else 0,
                'best_site': int(progression_df.loc[progression_df['auc'].idxmax(), 'site_id']) if not progression_df.empty else None,
                'average_final_auc': progression_df.groupby('site_id')['auc'].last().mean() if not progression_df.empty else 0,
                'improvement_rate': (progression_df.groupby('site_id')['auc'].last() - 
                                   progression_df.groupby('site_id')['auc'].first()).mean() if not progression_df.empty else 0
            },
            'site_analysis': {
                'most_improved_site': int(site_stats.loc[site_stats['auc_improvement'].idxmax(), 'site_id']) if not site_stats.empty else None,
                'best_performing_hospital_type': site_stats.loc[site_stats['best_auc'].idxmax(), 'hospital_type'] if not site_stats.empty else None,
                'data_size_correlation': site_stats[['num_samples', 'best_auc']].corr().iloc[0, 1] if not site_stats.empty else 0
            },
            'final_model_performance': final_metrics if final_metrics else {},
            'recommendations': []
        }
        
        # Add recommendations based on analysis
        if report['federated_performance']['improvement_rate'] > 0.05:
            report['recommendations'].append("Strong federated learning benefit observed. Consider more rounds.")
        
        if report['site_analysis']['data_size_correlation'] > 0.7:
            report['recommendations'].append("Larger sites show better performance. Consider data augmentation for smaller sites.")
        
        if final_metrics and final_metrics.get('auc', 0) > 0.8:
            report['recommendations'].append("Model shows strong performance. Ready for clinical validation.")
        
        # Save report
        with open(os.path.join(self.output_dir, 'wuperr_evaluation_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Evaluation report saved successfully")
        return report

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='WUPERR Federated Learning Evaluation')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Global test data path for final evaluation')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = WUPERRAnalyzer(output_dir=args.output_dir)
    
    # Load data
    data = analyzer.load_federated_data()
    if data is None:
        logger.error("Failed to load federated learning data")
        return
    
    # Analyze training progression
    progression_df = analyzer.analyze_training_progression(data)
    
    # Analyze site contributions
    site_stats = analyzer.analyze_site_contributions(data)
    
    # Evaluate final model
    final_metrics = analyzer.evaluate_final_model(args.test_data)
    
    # Generate comprehensive report
    report = analyzer.generate_report(data, progression_df, site_stats, final_metrics)
    
    # Print summary
    logger.info("\n=== WUPERR Federated Learning Evaluation Summary ===")
    logger.info(f"Total sites: {report['experiment_info']['total_sites']}")
    logger.info(f"Total rounds: {report['experiment_info']['total_rounds']}")
    logger.info(f"Best overall AUC: {report['federated_performance']['best_overall_auc']:.4f}")
    logger.info(f"Average improvement: {report['federated_performance']['improvement_rate']:.4f}")
    
    if final_metrics:
        logger.info(f"Final model AUC: {final_metrics['auc']:.4f}")
        logger.info(f"Final model accuracy: {final_metrics['accuracy']:.4f}")
    
    logger.info("\nRecommendations:")
    for rec in report['recommendations']:
        logger.info(f"- {rec}")
    
    logger.info(f"\nDetailed results saved to: {args.output_dir}/")
    logger.info(f"Visualizations saved to: {analyzer.graphs_dir}/")

if __name__ == "__main__":
    main()