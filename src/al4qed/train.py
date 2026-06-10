#!/usr/bin/env python3
"""
Complete training pipeline with configurable ML models.
Supports: Neural Network, SVM, Random Forest, XGBoost, Linear Regression
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

from oracle import AdaptivePolytopeOracle
from dataset import generate_dataset, extract_features_advanced, random_density_matrix
from states import horodecki_3x3


# ============================================================
#  Model factories
# ============================================================

def get_model(model_name, **kwargs):
    """
    Factory function to create ML models.
    
    Supported models:
        - 'nn': Neural Network (PyTorch)
        - 'svm': Support Vector Regressor
        - 'rf': Random Forest
        - 'xgb': XGBoost
        - 'linear': Linear Regression
    """
    if model_name == 'nn':
        import torch.nn as nn
        import torch
        
        class ChiPredictor(nn.Module):
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
                return self.net(x).squeeze(-1)
        
        model = ChiPredictor(
            input_dim=kwargs.get('input_dim', 178),
            hidden_dims=kwargs.get('hidden_dims', [256, 128, 64]),
            dropout_rate=kwargs.get('dropout_rate', 0.3)
        )
        return model
    
    elif model_name == 'svm':
        from sklearn.svm import SVR
        return SVR(
            kernel=kwargs.get('kernel', 'rbf'),
            C=kwargs.get('C', 1.0),
            epsilon=kwargs.get('epsilon', 0.01),
            cache_size=kwargs.get('cache_size', 1000)
        )
    
    elif model_name == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            n_jobs=-1,
            random_state=kwargs.get('random_state', 42)
        )
    
    elif model_name == 'xgb':
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=kwargs.get('random_state', 42)
            )
        except ImportError:
            print("XGBoost not installed. Install with: pip install xgboost")
            raise
    
    elif model_name == 'linear':
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: nn, svm, rf, xgb, linear")


def train_model(model, model_name, X_train, y_train, X_val, y_val, **kwargs):
    """
    Train model with appropriate training loop.
    For NN: uses PyTorch training loop
    For sklearn models: uses .fit()
    """
    if model_name == 'nn':
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train.astype(np.float32))
        y_train_t = torch.tensor(y_train.astype(np.float32))
        X_val_t = torch.tensor(X_val.astype(np.float32))
        y_val_t = torch.tensor(y_val.astype(np.float32))
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=kwargs.get('batch_size', 64), shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=kwargs.get('lr', 1e-3))
        criterion = nn.MSELoss()
        
        n_epochs = kwargs.get('n_epochs', 100)
        patience = kwargs.get('patience', 20)
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(n_epochs):
            # Training
            model.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            val_losses.append(val_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    model.load_state_dict(best_model_state)
                    break
        
        return model, {'train_losses': train_losses, 'val_losses': val_losses}
    
    else:
        # Sklearn models
        model.fit(X_train, y_train)
        return model, {}


def evaluate_model(model, model_name, X_test, y_test, scaler=None):
    """Evaluate model and return predictions and metrics."""
    if model_name == 'nn':
        import torch
        model.eval()
        if scaler is not None:
            X_test = scaler.transform(X_test)
        X_tensor = torch.tensor(X_test.astype(np.float32))
        with torch.no_grad():
            y_pred = model(X_tensor).cpu().numpy()
    else:
        if scaler is not None:
            X_test = scaler.transform(X_test)
        y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return y_pred, mse, mae, r2


# ============================================================
#  Plotting functions
# ============================================================

def plot_predictions(y_true, y_pred, model_name, save_path=None):
    """Scatter plot of Oracle χ vs predicted χ."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10, c='blue')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
    plt.xlabel('Oracle χ (true)', fontsize=12)
    plt.ylabel('Predicted χ', fontsize=12)
    plt.title(f'{model_name.upper()}: Prediction of Visibility', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


def plot_horodecki_curve(model, model_name, scaler, a_vals=None, save_path=None):
    """Compare Oracle vs model on Horodecki family."""
    if a_vals is None:
        a_vals = np.linspace(0.2, 1.0, 9)
    
    oracle = AdaptivePolytopeOracle(N=100, max_iter=10)
    true_chi = []
    pred_chi = []
    
    print(f"\nHorodecki family evaluation ({model_name}):")
    for a in a_vals:
        rho = horodecki_3x3(a)
        features = extract_features_advanced(rho).reshape(1, -1)
        
        # Get Oracle truth
        chi_true = oracle.query(rho, 3, 3)['chi']
        true_chi.append(chi_true)
        
        # Get model prediction
        if model_name == 'nn':
            import torch
            if scaler is not None:
                features = scaler.transform(features)
            X_tensor = torch.tensor(features.astype(np.float32))
            model.eval()
            with torch.no_grad():
                chi_pred = model(X_tensor).cpu().numpy()[0]
        else:
            if scaler is not None:
                features = scaler.transform(features)
            chi_pred = model.predict(features)[0]
        pred_chi.append(chi_pred)
        
        print(f"  a={a:.3f}: true={chi_true:.4f}, pred={chi_pred:.4f}, err={abs(chi_true-chi_pred):.4f}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(a_vals, true_chi, 'b-', linewidth=2, label='Oracle (true)')
    plt.plot(a_vals, pred_chi, 'r--', linewidth=2, label=f'{model_name.upper()} predicted')
    plt.xlabel('Parameter a', fontsize=12)
    plt.ylabel('Visibility χ(a)', fontsize=12)
    plt.title(f'{model_name.upper()}: Horodecki Family Prediction', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()
    
    return true_chi, pred_chi


# ============================================================
#  Main training function
# ============================================================

def run_training(
    model_name='rf',           # 'nn', 'svm', 'rf', 'xgb', 'linear'
    n_samples=2000,            # Number of training samples
    n_test=200,                # Number of test samples
    test_size=0.2,             # Train/val split ratio
    dims=(3, 3),               # System dimensions
    oracle_N=100,              # Oracle polytope vertices
    oracle_max_iter=10,        # Oracle max iterations
    save_model=True,           # Save trained model
    model_params=None,         # Additional model parameters
    nn_params=None,            # NN-specific parameters
    verbose=True
):
    """
    Main training pipeline.
    
    Parameters
    ----------
    model_name : str
        'nn', 'svm', 'rf', 'xgb', 'linear'
    n_samples : int
        Number of training samples
    n_test : int
        Number of test samples
    test_size : float
        Train/validation split ratio
    dims : tuple
        (dA, dB) dimensions
    oracle_N : int
        Number of polytope vertices for Oracle
    oracle_max_iter : int
        Max iterations for Oracle
    save_model : bool
        Whether to save the trained model
    model_params : dict
        Parameters for the model (SVM, RF, XGB, Linear)
    nn_params : dict
        Parameters for Neural Network (if model_name='nn')
    """
    
    print("=" * 70)
    print(f"AL4QED: Training {model_name.upper()} Model")
    print("=" * 70)
    
    # ============================================================
    # 1. Initialize Oracle and generate dataset
    # ============================================================
    print("\n[1] Initializing Oracle...")
    oracle = AdaptivePolytopeOracle(N=oracle_N, max_iter=oracle_max_iter, ent_threshold=0.99)
    
    print(f"\n[2] Generating {n_samples} training samples...")
    print("    (This may take 10-30 minutes)")
    X_train_full, y_train_full = generate_dataset(
        oracle,
        random_density_matrix,
        n_samples=n_samples,
        dims=dims,
        feature_fn=extract_features_advanced,
        save_path=f"data/{model_name}_train_dataset",
        seed=42
    )
    
    print(f"\n[3] Generating {n_test} test samples...")
    X_test, y_test = generate_dataset(
        oracle,
        random_density_matrix,
        n_samples=n_test,
        dims=dims,
        feature_fn=extract_features_advanced,
        save_path=f"data/{model_name}_test_dataset",
        seed=123
    )
    
    # ============================================================
    # 2. Split training data into train/validation
    # ============================================================
    print(f"\n[4] Splitting data (train/val = {1-test_size:.0%}/{test_size:.0%})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=test_size, random_state=42
    )
    print(f"    Train: {X_train.shape[0]} samples")
    print(f"    Val: {X_val.shape[0]} samples")
    print(f"    Test: {X_test.shape[0]} samples")
    
    # ============================================================
    # 3. Normalize features
    # ============================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # ============================================================
    # 4. Create and train model
    # ============================================================
    print(f"\n[5] Creating {model_name.upper()} model...")
    
    # Prepare model parameters
    if model_params is None:
        model_params = {}
    
    if model_name == 'nn':
        if nn_params is None:
            nn_params = {}
        model_params = {
            'input_dim': X_train.shape[1],
            'hidden_dims': nn_params.get('hidden_dims', [256, 128, 64]),
            'dropout_rate': nn_params.get('dropout_rate', 0.3)
        }
    
    model = get_model(model_name, **model_params)
    print(f"    Model created: {model}")
    
    print(f"\n[6] Training {model_name.upper()} model...")
    
    # Training parameters
    train_kwargs = {}
    if model_name == 'nn':
        train_kwargs = {
            'batch_size': nn_params.get('batch_size', 64) if nn_params else 64,
            'lr': nn_params.get('lr', 1e-3) if nn_params else 1e-3,
            'n_epochs': nn_params.get('n_epochs', 100) if nn_params else 100,
            'patience': nn_params.get('patience', 20) if nn_params else 20
        }
    
    model, training_log = train_model(
        model, model_name,
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        **train_kwargs
    )
    
    # ============================================================
    # 5. Evaluate on test set
    # ============================================================
    print(f"\n[7] Evaluating on test set...")
    y_pred, mse, mae, r2 = evaluate_model(model, model_name, X_test_scaled, y_test)
    
    print(f"\n{'='*50}")
    print("TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"  MSE  = {mse:.6f}")
    print(f"  MAE  = {mae:.6f}")
    print(f"  R²   = {r2:.4f}")
    print(f"{'='*50}")
    
    # ============================================================
    # 6. Visualization
    # ============================================================
    print(f"\n[8] Generating visualizations...")
    
    # Scatter plot
    plot_predictions(
        y_test, y_pred, model_name,
        save_path=f"figures/{model_name}_prediction.png"
    )
    
    # Horodecki curve
    plot_horodecki_curve(
        model, model_name, scaler,
        save_path=f"figures/{model_name}_horodecki.png"
    )
    
    # ============================================================
    # 7. Save model
    # ============================================================
    if save_model:
        os.makedirs("models", exist_ok=True)
        
        if model_name == 'nn':
            import torch
            torch.save(model.state_dict(), f"models/{model_name}_model.pt")
            print(f"\n✅ Model saved to models/{model_name}_model.pt")
        else:
            joblib.dump(model, f"models/{model_name}_model.joblib")
            print(f"\n✅ Model saved to models/{model_name}_model.joblib")
        
        # Save scaler
        joblib.dump(scaler, f"models/{model_name}_scaler.joblib")
        print(f"✅ Scaler saved to models/{model_name}_scaler.joblib")
    
    return model, scaler, {'mse': mse, 'mae': mae, 'r2': r2}


# ============================================================
#  Main entry point
# ============================================================

if __name__ == "__main__":
    # Example 1: Random Forest (default)
    print("\n" + "="*70)
    print("EXAMPLE 1: Random Forest")
    print("="*70)
    rf_model, rf_scaler, rf_metrics = run_training(
        model_name='rf',
        n_samples=500,      # Start with 500 for quick test
        n_test=50,
        model_params={
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5
        }
    )
    
    # Example 2: Neural Network (uncomment to run)
    # print("\n" + "="*70)
    # print("EXAMPLE 2: Neural Network")
    # print("="*70)
    # nn_model, nn_scaler, nn_metrics = run_training(
    #     model_name='nn',
    #     n_samples=1000,
    #     n_test=100,
    #     nn_params={
    #         'hidden_dims': [128, 64, 32],
    #         'dropout_rate': 0.3,
    #         'batch_size': 64,
    #         'lr': 1e-3,
    #         'n_epochs': 200,
    #         'patience': 30
    #     }
    # )
    
    # Example 3: SVM (uncomment to run)
    # print("\n" + "="*70)
    # print("EXAMPLE 3: SVM")
    # print("="*70)
    # svm_model, svm_scaler, svm_metrics = run_training(
    #     model_name='svm',
    #     n_samples=500,
    #     n_test=50,
    #     model_params={
    #         'kernel': 'rbf',
    #         'C': 1.0,
    #         'epsilon': 0.01
    #     }
    # )
    
    # Example 4: XGBoost (uncomment to run - requires xgboost installed)
    # print("\n" + "="*70)
    # print("EXAMPLE 4: XGBoost")
    # print("="*70)
    # xgb_model, xgb_scaler, xgb_metrics = run_training(
    #     model_name='xgb',
    #     n_samples=500,
    #     n_test=50,
    #     model_params={
    #         'n_estimators': 100,
    #         'max_depth': 6,
    #         'learning_rate': 0.1
    #     }
    # )