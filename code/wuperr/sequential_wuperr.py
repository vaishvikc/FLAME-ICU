import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, Optional, Tuple
import logging

class SequentialWUPERR:
    """
    Sequential WUPERR (Weighted Update with Partial Error Reduction and Regularization)
    for federated learning in a round-robin fashion.
    
    This implementation prevents catastrophic forgetting while allowing beneficial updates
    from each site in sequence.
    """
    
    def __init__(self, 
                 regularization_lambda: float = 0.01,
                 update_threshold: float = 0.001,
                 fisher_samples: int = 200,
                 ewc_lambda: float = 0.4):
        """
        Initialize WUPERR algorithm parameters.
        
        Args:
            regularization_lambda: Weight for preventing drift from important weights
            update_threshold: Minimum gradient magnitude for parameter updates
            fisher_samples: Number of samples for Fisher Information Matrix calculation
            ewc_lambda: Elastic Weight Consolidation regularization strength
        """
        self.lambda_reg = regularization_lambda
        self.threshold = update_threshold
        self.fisher_samples = fisher_samples
        self.ewc_lambda = ewc_lambda
        
        # Store important weights and Fisher Information
        self.important_weights = {}
        self.fisher_information = {}
        self.previous_params = {}
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def calculate_fisher_information(self, model: nn.Module, data_loader, device: str = 'cpu'):
        """
        Calculate Fisher Information Matrix to identify important weights.
        
        Args:
            model: PyTorch model
            data_loader: Data loader for Fisher calculation
            device: Device to run calculations on
        """
        model.eval()
        fisher_info = {}
        
        # Initialize Fisher information dict
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)
        
        # Calculate Fisher information
        num_samples = min(self.fisher_samples, len(data_loader.dataset))
        sample_count = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            if sample_count >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss (use the model's prediction as "target" for Fisher)
            probs = torch.sigmoid(output)
            loss = -torch.log(probs + 1e-8).mean()
            
            # Backward pass
            loss.backward()
            
            # Accumulate squared gradients (Fisher Information)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data.pow(2)
            
            sample_count += data.size(0)
        
        # Normalize by number of samples
        for name in fisher_info:
            fisher_info[name] /= num_samples
        
        self.fisher_information = fisher_info
        self.logger.info(f"Calculated Fisher Information for {len(fisher_info)} parameter groups")
        
    def store_previous_params(self, model: nn.Module):
        """Store current model parameters as previous parameters for EWC."""
        self.previous_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.previous_params[name] = param.data.clone()
    
    def calculate_weight_importance(self, model: nn.Module, data_loader, device: str = 'cpu'):
        """
        Calculate weight importance combining Fisher Information and gradient magnitude.
        
        Args:
            model: Current model
            data_loader: Data loader for importance calculation
            device: Device for calculations
        """
        # Calculate Fisher Information
        self.calculate_fisher_information(model, data_loader, device)
        
        # Store current parameters
        self.store_previous_params(model)
        
        # Calculate combined importance score
        self.important_weights = {}
        for name, fisher in self.fisher_information.items():
            # Normalize Fisher information
            fisher_norm = fisher / (fisher.max() + 1e-8)
            self.important_weights[name] = fisher_norm
            
        self.logger.info("Weight importance calculated and stored")
    
    def wuperr_loss(self, model: nn.Module, standard_loss: torch.Tensor) -> torch.Tensor:
        """
        Calculate WUPERR loss with regularization terms.
        
        Args:
            model: Current model
            standard_loss: Standard training loss (BCE, CrossEntropy, etc.)
            
        Returns:
            Total loss with WUPERR regularization
        """
        total_loss = standard_loss
        
        # EWC regularization term
        ewc_loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher_information and name in self.previous_params:
                fisher = self.fisher_information[name]
                prev_param = self.previous_params[name]
                ewc_loss += (fisher * (param - prev_param).pow(2)).sum()
        
        total_loss += self.ewc_lambda * ewc_loss
        
        # Additional L2 regularization for stability
        l2_loss = 0.0
        for param in model.parameters():
            if param.requires_grad:
                l2_loss += param.pow(2).sum()
        
        total_loss += self.lambda_reg * l2_loss
        
        return total_loss
    
    def apply_partial_updates(self, model: nn.Module, optimizer: optim.Optimizer):
        """
        Apply partial updates by zeroing gradients below threshold.
        
        Args:
            model: Model being trained
            optimizer: Optimizer being used
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Calculate gradient magnitude
                    grad_magnitude = param.grad.abs().mean().item()
                    
                    # Zero out gradients below threshold
                    if grad_magnitude < self.threshold:
                        param.grad.zero_()
                        
                    # Scale gradients by importance if available
                    elif name in self.important_weights:
                        importance = self.important_weights[name]
                        # Reduce updates to highly important weights
                        param.grad *= (1.0 - importance * 0.5)
    
    def train_step(self, model: nn.Module, data_loader, optimizer: optim.Optimizer, 
                   criterion, device: str = 'cpu') -> Dict[str, float]:
        """
        Perform one training step with WUPERR regularization.
        
        Args:
            model: Model to train
            data_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss criterion
            device: Device for training
            
        Returns:
            Dictionary with training metrics
        """
        model.train()
        total_loss = 0.0
        total_standard_loss = 0.0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate standard loss
            standard_loss = criterion(output, target)
            
            # Calculate WUPERR loss
            total_loss_batch = self.wuperr_loss(model, standard_loss)
            
            # Backward pass
            total_loss_batch.backward()
            
            # Apply partial updates
            self.apply_partial_updates(model, optimizer)
            
            # Optimizer step
            optimizer.step()
            
            # Accumulate metrics
            total_loss += total_loss_batch.item() * data.size(0)
            total_standard_loss += standard_loss.item() * data.size(0)
            total_samples += data.size(0)
        
        return {
            'total_loss': total_loss / total_samples,
            'standard_loss': total_standard_loss / total_samples,
            'samples': total_samples
        }
    
    def evaluate_model(self, model: nn.Module, data_loader, criterion, device: str = 'cpu') -> Dict[str, float]:
        """
        Evaluate model without WUPERR modifications.
        
        Args:
            model: Model to evaluate
            data_loader: Evaluation data loader
            criterion: Loss criterion
            device: Device for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                
                # Calculate accuracy
                pred = torch.sigmoid(output) > 0.5
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': correct / total_samples,
            'samples': total_samples
        }
    
    def save_importance_weights(self, filepath: str):
        """Save importance weights to file."""
        torch.save({
            'important_weights': self.important_weights,
            'fisher_information': self.fisher_information,
            'previous_params': self.previous_params
        }, filepath)
        
    def load_importance_weights(self, filepath: str):
        """Load importance weights from file."""
        checkpoint = torch.load(filepath)
        self.important_weights = checkpoint['important_weights']
        self.fisher_information = checkpoint['fisher_information']
        self.previous_params = checkpoint['previous_params']
        self.logger.info(f"Loaded importance weights from {filepath}")