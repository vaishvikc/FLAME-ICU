import os
import json
import subprocess
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import torch

class GitModelManager:
    """
    Manages model versioning and synchronization through Git for federated learning.
    Handles automated pull, commit, and push operations for model updates.
    """
    
    def __init__(self, repo_path: str = None, model_dir: str = "model"):
        """
        Initialize Git model manager.
        
        Args:
            repo_path: Path to git repository root
            model_dir: Directory within repo for model storage
        """
        self.repo_path = repo_path or os.getcwd()
        self.model_dir = os.path.join(self.repo_path, model_dir)
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Model file paths
        self.model_path = os.path.join(self.model_dir, "lstm_wuperr_model.pt")
        self.metadata_path = os.path.join(self.model_dir, "model_metadata.json")
        self.importance_path = os.path.join(self.model_dir, "weight_importance.pkl")
        self.contributions_path = os.path.join(self.model_dir, "site_contributions.json")
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize git repository if needed
        self._ensure_git_repo()
        
    def _ensure_git_repo(self):
        """Ensure we're in a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                self.logger.warning("Not in a git repository. Please initialize git first.")
        except Exception as e:
            self.logger.error(f"Error checking git repository: {e}")
    
    def _run_git_command(self, command: list, check_output: bool = False) -> Tuple[bool, str]:
        """
        Run a git command safely.
        
        Args:
            command: Git command as list
            check_output: Whether to return output
            
        Returns:
            Tuple of (success, output/error)
        """
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip()
                
        except Exception as e:
            return False, str(e)
    
    def pull_latest_model(self) -> bool:
        """
        Pull latest model from remote repository.
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Pulling latest model from remote repository...")
        
        # Pull latest changes
        success, output = self._run_git_command(["git", "pull", "origin", "main"])
        
        if success:
            self.logger.info("Successfully pulled latest changes")
            return True
        else:
            self.logger.error(f"Failed to pull latest changes: {output}")
            return False
    
    def commit_and_push_model(self, site_id: int, round_num: int, metrics: Dict) -> bool:
        """
        Commit and push updated model to repository.
        
        Args:
            site_id: ID of the site that trained the model
            round_num: Training round number
            metrics: Training metrics to include in commit message
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Committing and pushing model for Site {site_id}, Round {round_num}")
        
        # Add model files to staging
        files_to_add = [
            self.model_path,
            self.metadata_path,
            self.importance_path,
            self.contributions_path
        ]
        
        for file_path in files_to_add:
            if os.path.exists(file_path):
                rel_path = os.path.relpath(file_path, self.repo_path)
                success, output = self._run_git_command(["git", "add", rel_path])
                if not success:
                    self.logger.error(f"Failed to add {rel_path}: {output}")
                    return False
        
        # Create commit message
        commit_message = self._create_commit_message(site_id, round_num, metrics)
        
        # Commit changes
        success, output = self._run_git_command(["git", "commit", "-m", commit_message])
        if not success:
            if "nothing to commit" in output:
                self.logger.info("No changes to commit")
                return True
            else:
                self.logger.error(f"Failed to commit changes: {output}")
                return False
        
        # Push to remote
        success, output = self._run_git_command(["git", "push", "origin", "main"])
        if success:
            self.logger.info("Successfully pushed model to remote repository")
            return True
        else:
            self.logger.error(f"Failed to push to remote: {output}")
            return False
    
    def _create_commit_message(self, site_id: int, round_num: int, metrics: Dict) -> str:
        """Create formatted commit message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract key metrics
        accuracy = metrics.get('accuracy', 0.0)
        loss = metrics.get('loss', 0.0)
        auc = metrics.get('auc', 0.0)
        
        commit_message = f"WUPERR Site {site_id} Round {round_num}: "
        commit_message += f"Acc={accuracy:.3f}, Loss={loss:.3f}"
        if auc > 0:
            commit_message += f", AUC={auc:.3f}"
        commit_message += f" [{timestamp}]"
        
        return commit_message
    
    def load_model(self, model_class) -> Optional[torch.nn.Module]:
        """
        Load latest model from repository.
        
        Args:
            model_class: Class to instantiate model
            
        Returns:
            Loaded model or None if not found
        """
        if not os.path.exists(self.model_path):
            self.logger.warning(f"Model file not found at {self.model_path}")
            return None
        
        try:
            # Load metadata to get model configuration
            metadata = self.load_metadata()
            if metadata is None:
                self.logger.error("Could not load model metadata")
                return None
            
            # Initialize model with correct input size
            input_size = metadata.get('input_size', 68)  # Default from current implementation
            model = model_class(input_size=input_size)
            
            # Load model weights
            model.load_state_dict(torch.load(self.model_path))
            
            self.logger.info(f"Successfully loaded model from {self.model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None
    
    def save_model(self, model: torch.nn.Module, metadata: Dict):
        """
        Save model and metadata to repository.
        
        Args:
            model: PyTorch model to save
            metadata: Model metadata
        """
        try:
            # Save model weights
            torch.save(model.state_dict(), self.model_path)
            
            # Save metadata
            self.save_metadata(metadata)
            
            self.logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_metadata(self) -> Optional[Dict]:
        """Load model metadata."""
        if not os.path.exists(self.metadata_path):
            return None
        
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
            return None
    
    def save_metadata(self, metadata: Dict):
        """Save model metadata."""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def load_site_contributions(self) -> Dict:
        """Load site contributions history."""
        if not os.path.exists(self.contributions_path):
            return {}
        
        try:
            with open(self.contributions_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading site contributions: {e}")
            return {}
    
    def save_site_contributions(self, contributions: Dict):
        """Save site contributions history."""
        try:
            with open(self.contributions_path, 'w') as f:
                json.dump(contributions, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving site contributions: {e}")
    
    def update_site_contribution(self, site_id: int, round_num: int, metrics: Dict):
        """Update contributions for a specific site."""
        contributions = self.load_site_contributions()
        
        site_key = f"site_{site_id}"
        if site_key not in contributions:
            contributions[site_key] = {}
        
        contributions[site_key][f"round_{round_num}"] = {
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.save_site_contributions(contributions)
    
    def get_current_round(self) -> int:
        """Get the current training round number."""
        contributions = self.load_site_contributions()
        if not contributions:
            return 1
        
        max_round = 0
        for site_data in contributions.values():
            for round_key in site_data.keys():
                if round_key.startswith('round_'):
                    round_num = int(round_key.split('_')[1])
                    max_round = max(max_round, round_num)
        
        return max_round
    
    def get_last_training_site(self) -> Optional[int]:
        """Get the ID of the last site that trained the model."""
        contributions = self.load_site_contributions()
        if not contributions:
            return None
        
        latest_timestamp = None
        latest_site = None
        
        for site_key, site_data in contributions.items():
            for round_data in site_data.values():
                timestamp = datetime.fromisoformat(round_data['timestamp'])
                if latest_timestamp is None or timestamp > latest_timestamp:
                    latest_timestamp = timestamp
                    latest_site = int(site_key.split('_')[1])
        
        return latest_site
    
    def setup_git_lfs(self):
        """Set up Git LFS for large model files."""
        self.logger.info("Setting up Git LFS for model files...")
        
        # Track model files with LFS
        patterns = ["*.pt", "*.pkl", "*.pth", "*.h5"]
        
        for pattern in patterns:
            success, output = self._run_git_command(["git", "lfs", "track", pattern])
            if success:
                self.logger.info(f"Added {pattern} to Git LFS tracking")
            else:
                self.logger.warning(f"Failed to add {pattern} to Git LFS: {output}")
        
        # Add .gitattributes file
        success, output = self._run_git_command(["git", "add", ".gitattributes"])
        if success:
            self.logger.info("Added .gitattributes to git")
        else:
            self.logger.warning(f"Failed to add .gitattributes: {output}")
    
    def check_git_status(self) -> Dict:
        """Check git repository status."""
        success, output = self._run_git_command(["git", "status", "--porcelain"])
        
        if success:
            return {
                'clean': len(output) == 0,
                'status': output
            }
        else:
            return {
                'clean': False,
                'status': f"Error: {output}"
            }