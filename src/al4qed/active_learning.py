"""
Active Learning with Neural Network surrogate.
Uses MC Dropout for uncertainty estimation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

from oracle import AdaptivePolytopeOracle
from dataset import extract_features_advanced


class ActiveLearner:
    """
    Active Learning with Neural Network surrogate.
    
    Pipeline:
        1. Initial labeled set → train NN to predict χ
        2. For each unlabeled state, compute uncertainty via MC Dropout
        3. Select highest uncertainty states → query Oracle
        4. Add to labeled set → retrain NN
    """
    
    def __init__(self, oracle, model, input_dim, device='cpu',
                 lr=1e-3, batch_size=32, n_epochs=100, val_split=0.2):
        self.oracle = oracle
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.val_split = val_split
        
        self.scaler = StandardScaler()
        self.X_labeled = []      # Features of labeled states
        self.y_labeled = []      # χ values from Oracle
        self.pool = []           # Unlabeled states (features, original rho, dims)
        
        self.training_history = []  # List of (train_losses, val_losses)
    
    def add_labeled(self, X, y):
        """Add labeled data (from Oracle)."""
        if isinstance(X, np.ndarray):
            X = X.tolist()
        if isinstance(y, np.ndarray):
            y = y.tolist()
        self.X_labeled.extend(X)
        self.y_labeled.extend(y)
    
    def add_to_pool(self, states, dims_list):
        """Add unlabeled states to pool."""
        for rho, dA, dB in zip(states, dims_list):
            features = extract_features_advanced(rho)
            self.pool.append({
                'features': features,
                'rho': rho,
                'dA': dA,
                'dB': dB
            })
    
    def _prepare_data(self):
        """Prepare labeled data for training with validation split."""
        if len(self.X_labeled) < 10:
            return None, None, None, None
        
        X = np.array(self.X_labeled, dtype=np.float32)
        y = np.array(self.y_labeled, dtype=np.float32)
        
        # Normalize features
        X = self.scaler.fit_transform(X)
        
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)
        
        # Split into train/val
        val_size = int(len(X_tensor) * self.val_split)
        train_size = len(X_tensor) - val_size
        
        if val_size > 0:
            train_dataset, val_dataset = random_split(
                TensorDataset(X_tensor, y_tensor), 
                [train_size, val_size]
            )
            return train_dataset, val_dataset, train_size, val_size
        else:
            return TensorDataset(X_tensor, y_tensor), None, train_size, 0
    
    def train_surrogate(self, verbose=False):
        """Train neural network on labeled data with validation."""
        train_dataset, val_dataset, train_size, val_size = self._prepare_data()
        if train_dataset is None:
            if verbose:
                print("Not enough labeled data (need ≥10 samples)")
            return
        
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, train_size), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=min(self.batch_size, val_size)) if val_dataset else None
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        self.model.train()
        train_losses = []
        val_losses = []
        
        for epoch in range(self.n_epochs):
            # Training
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            if val_loader:
                self.model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        pred = self.model(batch_X)
                        loss = criterion(pred, batch_y)
                        total_val_loss += loss.item()
                avg_val_loss = total_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                self.model.train()
                
                if verbose and (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{self.n_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{self.n_epochs}, Train Loss: {avg_train_loss:.6f}")
        
        self.training_history.append((train_losses, val_losses))
        self.model.eval()
    
    def predict_chi(self, features):
        """Predict χ for a single state (features)."""
        if not hasattr(self.scaler, 'mean_') and self.X_labeled:
            # Fit scaler on labeled data
            self.scaler.fit(np.array(self.X_labeled))
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        X_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            chi_pred = self.model(X_tensor).cpu().numpy()[0]
        return chi_pred
    
    def predict_uncertainty(self, features, n_mc_samples=20):
        """
        Predict χ with uncertainty using MC Dropout.
        Returns (mean_chi, std_chi)
        """
        if not hasattr(self.scaler, 'mean_') and self.X_labeled:
            self.scaler.fit(np.array(self.X_labeled))
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        X_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
        
        self.model.train()  # Enable dropout
        with torch.no_grad():
            predictions = []
            for _ in range(n_mc_samples):
                pred = self.model(X_tensor).cpu().numpy()[0]
                predictions.append(pred)
        self.model.eval()
        
        mean = np.mean(predictions)
        std = np.std(predictions)
        return mean, std
    
    def query_uncertain(self, n_queries=5, n_mc_samples=20):
        """
        Select and query states with highest uncertainty.
        FIXED: delete indices in reverse order to avoid shifting.
        Returns list of oracle results.
        """
        if not self.pool:
            return []
        
        # Compute uncertainty for all pool states
        uncertainties = []
        for item in self.pool:
            try:
                _, std = self.predict_uncertainty(item['features'], n_mc_samples)
                uncertainties.append(std)
            except:
                uncertainties.append(1.0)  # Default high uncertainty if fails
        
        # Select top uncertainties
        indices = np.argsort(uncertainties)[::-1][:n_queries]
        
        results = []
        # FIXED: delete from largest index to smallest
        for idx in sorted(indices, reverse=True):
            item = self.pool[idx]
            oracle_result = self.oracle.query(item['rho'], item['dA'], item['dB'])
            results.append(oracle_result)
            
            # Move from pool to labeled set
            self.X_labeled.append(item['features'])
            self.y_labeled.append(oracle_result['chi'])
            del self.pool[idx]
        
        return results
    
    def active_learning_cycle(self, n_cycles=10, queries_per_cycle=5):
        """
        Run full active learning cycle.
        """
        for cycle in range(n_cycles):
            print(f"\n{'='*50}")
            print(f"Active Learning Cycle {cycle+1}/{n_cycles}")
            print(f"Labeled samples: {len(self.X_labeled)}")
            print(f"Pool samples: {len(self.pool)}")
            
            # Train surrogate model
            print("Training surrogate...")
            self.train_surrogate(verbose=(cycle % 3 == 0))
            
            # Query uncertain states
            print(f"Querying {queries_per_cycle} most uncertain states...")
            results = self.query_uncertain(n_queries=queries_per_cycle)
            
            for res in results:
                print(f"  χ = {res['chi']:.5f}, label = {res['label'].name}")
            
            if len(self.pool) == 0:
                print("Pool exhausted.")
                break
        
        return self.X_labeled, self.y_labeled


if __name__ == "__main__":
    from network import SimpleChiPredictor
    from states import horodecki_3x3
    
    print("=" * 60)
    print("Testing Active Learner")
    print("=" * 60)
    
    # Create simple wrapper for testing
    from adaptive import adaptive_polytope_bipartite
    from oracle import OracleLabel
    
    class SimpleOracle:
        def query(self, rho, dA, dB, seed=None):
            chi, _, _ = adaptive_polytope_bipartite(rho, dA, dB, N=50, max_iter=8, seed=seed)
            label = OracleLabel.ENTANGLED if chi < 0.99 else OracleLabel.UNKNOWN
            return {'chi': chi, 'label': label, 'history': []}
    
    oracle = SimpleOracle()
    
    # Create model
    input_dim = 2*9*9 + 9 + 1 + 3 + 3  # 162 + 9 + 1 + 3 + 3 = 178
    model = SimpleChiPredictor(input_dim)
    
    learner = ActiveLearner(oracle, model, input_dim, n_epochs=10)
    
    # Add dummy labeled data
    X_dummy = np.random.randn(20, input_dim).astype(np.float32)
    y_dummy = np.random.rand(20).astype(np.float32)
    learner.add_labeled(X_dummy, y_dummy)
    
    # Add to pool
    rho = horodecki_3x3(0.3)
    learner.add_to_pool([rho], [(3, 3)])
    
    # Test prediction
    features = extract_features_advanced(rho)
    chi_pred = learner.predict_chi(features)
    print(f"\nTest prediction: χ ≈ {chi_pred:.4f}")
    
    print("\n✅ Active Learner ready")