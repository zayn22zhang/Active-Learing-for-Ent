"""
Neural network for predicting χ(ρ) from features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChiPredictor(nn.Module):
    """
    Predicts visibility χ from density matrix features.
    Output is in [0, 1] via sigmoid.
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass, returns χ ∈ [0, 1]"""
        return self.net(x).squeeze(-1)
    
    def predict_with_uncertainty(self, x, n_mc_samples=20):
        """
        Monte Carlo Dropout for uncertainty estimation.
        Returns (mean_chi, std_chi)
        """
        self.train()  # Enable dropout for MC sampling
        with torch.no_grad():
            predictions = torch.stack([self(x) for _ in range(n_mc_samples)])
        self.eval()
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        return mean, std


class SimpleChiPredictor(nn.Module):
    """Smaller network for quick experiments."""
    
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)