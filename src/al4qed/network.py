"""Small-data MLP and safe MC dropout. Predictions are not certificates."""
from contextlib import contextmanager
import torch
from torch import nn


@contextmanager
def dropout_only(model):
    """Enable only dropout; preserve BatchNorm buffers and every module's mode."""
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.train()
        yield
    finally:
        for module, training in modes.items():
            module.training = training


class ChiPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout_rate=0.2):
        super().__init__()
        hidden_dims = [128,64] if hidden_dims is None else hidden_dims
        layers = []
        for size in hidden_dims:
            layers.extend([nn.Linear(input_dim, size), nn.LayerNorm(size), nn.ReLU(), nn.Dropout(dropout_rate)])
            input_dim = size
        layers.extend([nn.Linear(input_dim, 1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict_with_uncertainty(self, x, n_mc_samples=20):
        if n_mc_samples < 2:
            raise ValueError('At least two MC samples required')
        with dropout_only(self), torch.no_grad():
            samples = torch.stack([self(x) for _ in range(n_mc_samples)])
        return samples.mean(0), samples.std(0, unbiased=False)


class SimpleChiPredictor(ChiPredictor):
    def __init__(self, input_dim):
        super().__init__(input_dim, hidden_dims=[64,32])
