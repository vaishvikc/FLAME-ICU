import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ICUMortalityNN(nn.Module):
    """
    Multi-layer perceptron for ICU mortality prediction.
    Supports configurable hidden layers, dropout, and batch normalization.
    """
    
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.3, activation='relu', batch_norm=True):
        super(ICUMortalityNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build the network layers
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if batch_norm else None
        self.dropout_layers = nn.ModuleList()
        
        # Input layer to first hidden layer
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            if batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_size))
            
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, 1)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid activation
        """
        # Pass through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply batch normalization if enabled
            if self.batch_norm:
                x = self.bn_layers[i](x)
            
            # Apply activation
            x = self.activation(x)
            
            # Apply dropout
            x = self.dropout_layers[i](x)
        
        # Output layer with sigmoid activation for binary classification
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        
        # Clamp to avoid numerical issues with BCE loss
        x = torch.clamp(x, min=1e-7, max=1-1e-7)
        
        return x
    
    def get_feature_importance(self, X, method='gradient'):
        """
        Calculate feature importance using gradient-based methods.
        
        Args:
            X: Input tensor or numpy array
            method: 'gradient' for simple gradients or 'integrated_gradients'
            
        Returns:
            Feature importance scores
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        if method == 'gradient':
            # Simple gradient-based importance
            X.requires_grad = True
            outputs = self.forward(X)
            
            # Calculate gradients for positive class predictions
            gradients = torch.autograd.grad(
                outputs.sum(), X, retain_graph=True
            )[0]
            
            # Average absolute gradients across samples
            importance = torch.abs(gradients).mean(dim=0).detach().numpy()
            
        elif method == 'integrated_gradients':
            # Integrated gradients for more robust importance scores
            baseline = torch.zeros_like(X)
            steps = 50
            
            integrated_grads = torch.zeros_like(X)
            
            for i in range(steps):
                alpha = i / steps
                interpolated = baseline + alpha * (X - baseline)
                interpolated.requires_grad = True
                
                outputs = self.forward(interpolated)
                gradients = torch.autograd.grad(
                    outputs.sum(), interpolated, retain_graph=True
                )[0]
                
                integrated_grads += gradients / steps
            
            # Average across samples and take absolute values
            importance = torch.abs(integrated_grads * (X - baseline)).mean(dim=0).detach().numpy()
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return importance


class EarlyStopping:
    """Early stopping handler to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max' and score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == 'min' and score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop