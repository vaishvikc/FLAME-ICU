import os
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import argparse
import subprocess
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultisiteSimulator:
    """
    Simulates federated learning across multiple sites by:
    1. Splitting data into heterogeneous site-specific datasets
    2. Running sequential WUPERR training across sites
    3. Monitoring and evaluating the federated training process
    """
    
    def __init__(self, data_path: str, num_sites: int = 8, output_dir: str = "data/sites"):
        """
        Initialize multisite simulator.
        
        Args:
            data_path: Path to the main dataset
            num_sites: Number of sites to simulate
            output_dir: Directory to save site-specific data
        """
        self.data_path = data_path
        self.num_sites = num_sites
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Site configuration - simulating different hospital sizes and characteristics
        self.site_config = {
            1: {'proportion': 0.25, 'type': 'large_academic', 'bias': 'high_acuity'},
            2: {'proportion': 0.20, 'type': 'large_community', 'bias': 'balanced'},
            3: {'proportion': 0.15, 'type': 'medium_academic', 'bias': 'research_focused'},
            4: {'proportion': 0.15, 'type': 'medium_community', 'bias': 'balanced'},
            5: {'proportion': 0.10, 'type': 'small_community', 'bias': 'low_acuity'},
            6: {'proportion': 0.08, 'type': 'small_rural', 'bias': 'limited_resources'},
            7: {'proportion': 0.05, 'type': 'specialty_hospital', 'bias': 'specific_conditions'},
            8: {'proportion': 0.02, 'type': 'critical_access', 'bias': 'basic_care'}
        }
    
    def load_main_dataset(self) -> pd.DataFrame:
        """Load the main dataset."""
        logger.info(f"Loading main dataset from {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        df = pd.read_parquet(self.data_path)
        logger.info(f"Main dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        
        return df
    
    def create_site_specific_bias(self, df: pd.DataFrame, site_id: int) -> pd.DataFrame:
        """
        Create site-specific bias to simulate real-world hospital differences.
        
        Args:
            df: Dataset to modify
            site_id: Site identifier
            
        Returns:
            Modified dataset with site-specific characteristics
        """
        site_info = self.site_config[site_id]
        bias_type = site_info['bias']
        
        logger.info(f"Applying {bias_type} bias to site {site_id}")
        
        # Apply different biases based on site characteristics
        if bias_type == 'high_acuity':
            # Higher mortality rate, more severe cases
            mortality_mask = df['disposition'] == 1
            df_biased = df.copy()
            # Increase representation of mortality cases
            mortality_samples = df[mortality_mask]
            if len(mortality_samples) > 0:
                additional_samples = mortality_samples.sample(
                    n=min(len(mortality_samples), int(len(df) * 0.1)), 
                    replace=True, 
                    random_state=42 + site_id
                )
                df_biased = pd.concat([df_biased, additional_samples], ignore_index=True)
        
        elif bias_type == 'low_acuity':
            # Lower mortality rate, less severe cases
            survival_mask = df['disposition'] == 0
            df_biased = df[survival_mask].copy()
            # Add some mortality cases to maintain balance
            mortality_samples = df[~survival_mask].sample(
                n=min(len(df[~survival_mask]), int(len(df_biased) * 0.05)), 
                replace=True, 
                random_state=42 + site_id
            )
            df_biased = pd.concat([df_biased, mortality_samples], ignore_index=True)
        
        elif bias_type == 'limited_resources':
            # Fewer lab values, more missing data
            df_biased = df.copy()
            # Randomly set some lab values to NaN
            lab_cols = [col for col in df.columns if any(lab in col.lower() for lab in ['lab', 'glucose', 'sodium', 'potassium'])]
            for col in lab_cols:
                mask = np.random.random(len(df_biased)) < 0.3  # 30% missing
                df_biased.loc[mask, col] = np.nan
        
        elif bias_type == 'specific_conditions':
            # Focus on specific types of patients (e.g., cardiac patients)
            df_biased = df.copy()
            # Simulate focus on cardiac patients by boosting cardiac-related features
            cardiac_features = [col for col in df.columns if any(term in col.lower() for term in ['heart', 'cardiac', 'troponin'])]
            for col in cardiac_features:
                if col in df_biased.columns:
                    df_biased[col] = df_biased[col] * 1.2  # Boost cardiac features
        
        else:  # balanced
            df_biased = df.copy()
        
        return df_biased
    
    def create_site_splits(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Create heterogeneous splits for each site.
        
        Args:
            df: Main dataset
            
        Returns:
            Dictionary mapping site_id to site-specific dataset
        """
        logger.info("Creating site-specific data splits")
        
        # Shuffle data
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        site_datasets = {}
        start_idx = 0
        
        for site_id in range(1, self.num_sites + 1):
            proportion = self.site_config[site_id]['proportion']
            site_size = int(len(df_shuffled) * proportion)
            
            # Extract site data
            end_idx = start_idx + site_size
            site_data = df_shuffled.iloc[start_idx:end_idx].copy()
            
            # Apply site-specific bias
            site_data = self.create_site_specific_bias(site_data, site_id)
            
            # Store site dataset
            site_datasets[site_id] = site_data
            
            logger.info(f"Site {site_id} ({self.site_config[site_id]['type']}): "
                       f"{len(site_data)} samples, "
                       f"mortality rate: {site_data['disposition'].mean():.3f}")
            
            start_idx = end_idx
        
        return site_datasets
    
    def save_site_datasets(self, site_datasets: Dict[int, pd.DataFrame]):
        """Save site-specific datasets to disk."""
        logger.info("Saving site datasets to disk")
        
        for site_id, site_data in site_datasets.items():
            output_path = os.path.join(self.output_dir, f"site_{site_id}_data.parquet")
            site_data.to_parquet(output_path, index=False)
            logger.info(f"Saved site {site_id} data to {output_path}")
    
    def create_site_metadata(self, site_datasets: Dict[int, pd.DataFrame]):
        """Create metadata for each site."""
        metadata = {}
        
        for site_id, site_data in site_datasets.items():
            site_info = self.site_config[site_id]
            
            metadata[f"site_{site_id}"] = {
                'site_id': site_id,
                'hospital_type': site_info['type'],
                'bias_type': site_info['bias'],
                'num_samples': len(site_data),
                'mortality_rate': float(site_data['disposition'].mean()),
                'feature_availability': {
                    'total_features': len(site_data.columns),
                    'missing_data_rate': float(site_data.isnull().sum().sum() / (len(site_data) * len(site_data.columns)))
                },
                'temporal_range': {
                    'start_index': int(site_data.index.min()),
                    'end_index': int(site_data.index.max())
                }
            }
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, "site_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Site metadata saved to {metadata_path}")
    
    def run_sequential_training(self, num_rounds: int = 3, config_path: str = "wuperr/config_wuperr.json"):
        """
        Run sequential WUPERR training across all sites.
        
        Args:
            num_rounds: Number of training rounds
            config_path: Path to WUPERR configuration
        """
        logger.info(f"Starting sequential WUPERR training for {num_rounds} rounds")
        
        training_results = []
        
        for round_num in range(1, num_rounds + 1):
            logger.info(f"Starting Round {round_num}")
            
            for site_id in range(1, self.num_sites + 1):
                logger.info(f"Training Site {site_id} in Round {round_num}")
                
                # Run training script
                cmd = [
                    sys.executable, "sequential_train.py",
                    "--site_id", str(site_id),
                    "--round_num", str(round_num),
                    "--config", config_path
                ]
                
                try:
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True,
                        timeout=1800  # 30 minutes timeout
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"Site {site_id} Round {round_num} completed successfully")
                        training_results.append({
                            'site_id': site_id,
                            'round_num': round_num,
                            'status': 'success',
                            'stdout': result.stdout,
                            'stderr': result.stderr
                        })
                    else:
                        logger.error(f"Site {site_id} Round {round_num} failed")
                        training_results.append({
                            'site_id': site_id,
                            'round_num': round_num,
                            'status': 'failed',
                            'stdout': result.stdout,
                            'stderr': result.stderr
                        })
                
                except subprocess.TimeoutExpired:
                    logger.error(f"Site {site_id} Round {round_num} timed out")
                    training_results.append({
                        'site_id': site_id,
                        'round_num': round_num,
                        'status': 'timeout',
                        'stdout': '',
                        'stderr': 'Training timed out'
                    })
                
                except Exception as e:
                    logger.error(f"Error running Site {site_id} Round {round_num}: {e}")
                    training_results.append({
                        'site_id': site_id,
                        'round_num': round_num,
                        'status': 'error',
                        'stdout': '',
                        'stderr': str(e)
                    })
        
        # Save training results
        results_path = os.path.join(self.output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info(f"Training results saved to {results_path}")
        return training_results
    
    def analyze_results(self):
        """Analyze and summarize federated training results."""
        from .git_model_manager import GitModelManager
        
        logger.info("Analyzing federated training results")
        
        git_manager = GitModelManager()
        contributions = git_manager.load_site_contributions()
        
        if not contributions:
            logger.warning("No training contributions found")
            return
        
        # Analyze performance across sites and rounds
        analysis = {
            'summary': {
                'total_sites': len(contributions),
                'total_rounds': 0,
                'best_overall_auc': 0.0,
                'best_site': None
            },
            'site_performance': {},
            'round_progression': {}
        }
        
        # Process each site's contributions
        for site_key, site_data in contributions.items():
            site_id = int(site_key.split('_')[1])
            site_performance = {
                'rounds_completed': len(site_data),
                'best_auc': 0.0,
                'best_round': None,
                'improvement_trend': []
            }
            
            # Analyze each round for this site
            for round_key, round_data in site_data.items():
                round_num = int(round_key.split('_')[1])
                metrics = round_data['metrics']
                
                auc = metrics.get('auc', 0.0)
                
                if auc > site_performance['best_auc']:
                    site_performance['best_auc'] = auc
                    site_performance['best_round'] = round_num
                
                if auc > analysis['summary']['best_overall_auc']:
                    analysis['summary']['best_overall_auc'] = auc
                    analysis['summary']['best_site'] = site_id
                
                site_performance['improvement_trend'].append({
                    'round': round_num,
                    'auc': auc,
                    'accuracy': metrics.get('accuracy', 0.0),
                    'loss': metrics.get('val_loss', 0.0)
                })
                
                analysis['summary']['total_rounds'] = max(
                    analysis['summary']['total_rounds'], round_num
                )
            
            analysis['site_performance'][site_id] = site_performance
        
        # Save analysis
        analysis_path = os.path.join(self.output_dir, "federated_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Federated learning analysis saved to {analysis_path}")
        logger.info(f"Best overall AUC: {analysis['summary']['best_overall_auc']:.4f} "
                   f"(Site {analysis['summary']['best_site']})")
        
        return analysis

def main():
    """Main function to run multisite simulation."""
    parser = argparse.ArgumentParser(description='Multisite WUPERR Simulation')
    parser.add_argument('--data_path', type=str, 
                       default='../../../output/intermitted/by_hourly_wide_df.parquet',
                       help='Path to main dataset')
    parser.add_argument('--num_sites', type=int, default=8,
                       help='Number of sites to simulate')
    parser.add_argument('--num_rounds', type=int, default=3,
                       help='Number of training rounds')
    parser.add_argument('--output_dir', type=str, default='data/sites',
                       help='Output directory for site data')
    parser.add_argument('--config', type=str, default='config_wuperr.json',
                       help='WUPERR configuration file')
    parser.add_argument('--setup_only', action='store_true',
                       help='Only setup site data, do not run training')
    
    args = parser.parse_args()
    
    # Create simulator
    simulator = MultisiteSimulator(
        data_path=args.data_path,
        num_sites=args.num_sites,
        output_dir=args.output_dir
    )
    
    # Load and split data
    df = simulator.load_main_dataset()
    site_datasets = simulator.create_site_splits(df)
    
    # Save site datasets and metadata
    simulator.save_site_datasets(site_datasets)
    simulator.create_site_metadata(site_datasets)
    
    if not args.setup_only:
        # Run sequential training
        training_results = simulator.run_sequential_training(
            num_rounds=args.num_rounds,
            config_path=args.config
        )
        
        # Analyze results
        analysis = simulator.analyze_results()
        
        # Print summary
        logger.info("=== WUPERR Federated Learning Summary ===")
        logger.info(f"Total sites: {args.num_sites}")
        logger.info(f"Total rounds: {args.num_rounds}")
        
        successful_trainings = sum(1 for r in training_results if r['status'] == 'success')
        total_trainings = len(training_results)
        
        logger.info(f"Successful trainings: {successful_trainings}/{total_trainings}")
        
        if analysis:
            logger.info(f"Best overall AUC: {analysis['summary']['best_overall_auc']:.4f}")
            logger.info(f"Best performing site: {analysis['summary']['best_site']}")
    
    logger.info("Multisite simulation completed!")

if __name__ == "__main__":
    main()