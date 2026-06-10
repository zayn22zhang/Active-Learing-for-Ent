"""
Dataset generator using Adaptive Polytope Oracle.
Generates (features, chi) pairs for training neural networks.
"""

import numpy as np
from tqdm import tqdm
from oracle import AdaptivePolytopeOracle
from states import horodecki_3x3, werner_state, isotropic_state


def random_density_matrix(d, rng=None):
    """Generate random density matrix via Ginibre ensemble."""
    if rng is None:
        rng = np.random.default_rng()
    G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = G @ G.conj().T
    return rho / np.trace(rho)


def extract_features_advanced(rho):
    """
    Extract rich features from density matrix for ML input.
    
    Features:
    - Flattened real and imag parts (2*d^2 dimensions)
    - Eigenvalues of rho (d dimensions)
    - Purity = Tr(ρ²) (1 dimension)
    - Local eigenvalues (reduced density matrices) (dA + dB dimensions)
    
    Returns feature vector of dimension: 2*d^2 + d + 1 + dA + dB
    """
    d = rho.shape[0]
    dA = int(np.sqrt(d))  # Assumes dA = dB for simplicity
    dB = dA
    
    # 1. Flattened real and imag
    flat = rho.flatten()
    features = list(np.real(flat)) + list(np.imag(flat))
    
    # 2. Eigenvalues of rho
    eigs = np.linalg.eigvalsh(rho)
    features.extend(eigs)
    
    # 3. Purity = Tr(ρ²)
    purity = np.real(np.trace(rho @ rho))
    features.append(purity)
    
    # 4. Reduced density matrices eigenvalues
    # Reshape to get partial trace
    rho_reshaped = rho.reshape(dA, dB, dA, dB)
    
    # Reduced on A: trace over B
    rho_A = np.einsum('ijik', rho_reshaped).reshape(dA, dA)
    rho_A = rho_A / np.trace(rho_A)
    eigs_A = np.linalg.eigvalsh(rho_A)
    features.extend(eigs_A)
    
    # Reduced on B: trace over A
    rho_B = np.einsum('ijki', rho_reshaped).reshape(dB, dB)
    rho_B = rho_B / np.trace(rho_B)
    eigs_B = np.linalg.eigvalsh(rho_B)
    features.extend(eigs_B)
    
    return np.array(features, dtype=np.float32)


def extract_features_simple(rho):
    """Simple feature extraction: flattened real + imag only."""
    flat = rho.flatten()
    features = np.concatenate([np.real(flat), np.imag(flat)])
    return features.astype(np.float32)


def generate_dataset(oracle, state_generator, n_samples, dims=(3, 3), 
                     feature_fn=extract_features_advanced,
                     save_path=None, seed=42):
    """
    Generate dataset of (features, chi) pairs.
    
    Parameters
    ----------
    oracle : AdaptivePolytopeOracle
    state_generator : callable or list of callables
        Function that returns a density matrix, or list of family names
    n_samples : int
        Number of samples to generate
    dims : tuple
        (dA, dB) dimensions
    feature_fn : callable
        Function to extract features from density matrix
    save_path : str or None
        Path to save numpy arrays
    
    Returns
    -------
    X : np.ndarray, shape (n_samples, feature_dim)
    y : np.ndarray, shape (n_samples,)
    """
    rng = np.random.default_rng(seed)
    d_total = dims[0] * dims[1]
    
    # First generate a sample to get feature dimension
    test_rho = random_density_matrix(d_total, rng)
    feature_dim = len(feature_fn(test_rho))
    
    X = np.zeros((n_samples, feature_dim), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    
    print(f"Generating {n_samples} samples with Oracle...")
    print(f"  Feature dimension: {feature_dim}")
    
    for i in tqdm(range(n_samples)):
        # Generate random state
        if callable(state_generator):
            # FIXED: pass d_total, not dims[0]
            rho = state_generator(d_total, rng=rng)
        else:
            # state_generator is a list of family names
            family = rng.choice(state_generator)
            if family == 'horodecki':
                a = rng.uniform(0.2, 1.0)
                rho = horodecki_3x3(a)
            elif family == 'werner':
                p = rng.uniform(0, 1)
                rho = werner_state(p, d=2)  # 2x2 only
            elif family == 'isotropic':
                p = rng.uniform(0, 1)
                rho = isotropic_state(p, d=2)
            else:
                rho = random_density_matrix(d_total, rng)
        
        # Query oracle to get χ
        result = oracle.query(rho, dims[0], dims[1])
        chi = result['chi']
        
        # Extract features
        X[i] = feature_fn(rho)
        y[i] = chi
    
    if save_path:
        np.save(f"{save_path}_X.npy", X)
        np.save(f"{save_path}_y.npy", y)
        print(f"Saved to {save_path}_X.npy and {save_path}_y.npy")
    
    return X, y


if __name__ == "__main__":
    from oracle import AdaptivePolytopeOracle
    
    oracle = AdaptivePolytopeOracle(N=100, max_iter=10)
    X, y = generate_dataset(oracle, random_density_matrix, n_samples=100, dims=(3, 3))
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"y range: [{y.min():.3f}, {y.max():.3f}]")