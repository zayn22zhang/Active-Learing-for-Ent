"""Physical density matrices and dimension-aware, deterministic features."""
from pathlib import Path
import numpy as np
try:
    from .states import horodecki_3x3, werner_state, isotropic_state, ghz_w_3x3
except ImportError:
    from states import horodecki_3x3, werner_state, isotropic_state, ghz_w_3x3

FEATURE_VERSION = 'density-local-spectra-v2'


def validate_density(rho, dims=None, tol=1e-8):
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError('rho must be a square matrix')
    if dims is not None and (len(dims) != 2 or any(int(d) != d or d < 2 for d in dims)
                             or np.prod(dims) != rho.shape[0]):
        raise ValueError('dims must be two local dimensions >=2 matching rho')
    if not np.isfinite(rho).all() or not np.allclose(rho, rho.conj().T, atol=tol, rtol=0):
        raise ValueError('rho must be finite and Hermitian')
    if abs(np.trace(rho) - 1) > tol or np.linalg.eigvalsh(rho).min() < -tol:
        raise ValueError('rho must be positive semidefinite with trace one')
    return rho


def resolve_dims(rho, dims=None):
    if dims is None:
        side = int(np.sqrt(len(rho)))
        if side * side != len(rho):
            raise ValueError('Non-square bipartitions require explicit dims=(dA,dB)')
        dims = (side, side)
    if np.prod(dims) != len(rho) or len(dims) != 2:
        raise ValueError('dims do not match rho')
    return tuple(dims)


def reduced_states(rho, dims=None):
    dA, dB = resolve_dims(rho, dims)
    tensor = np.asarray(rho).reshape(dA, dB, dA, dB)
    return np.trace(tensor, axis1=1, axis2=3), np.trace(tensor, axis1=0, axis2=2)


def extract_features_simple(rho):
    flat = np.asarray(rho).reshape(-1)
    return np.concatenate((flat.real, flat.imag)).astype(np.float32)


def extract_features_advanced(rho, dims=None):
    """2D²+D+1+dA+dB features; preserves the original feature length for 3x3."""
    rho = validate_density(rho, dims)
    a, b = reduced_states(rho, dims)
    return np.concatenate((extract_features_simple(rho), np.linalg.eigvalsh(rho),
                           [np.trace(rho @ rho).real], np.linalg.eigvalsh(a),
                           np.linalg.eigvalsh(b))).astype(np.float32)


def random_density_matrix(d, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    g = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = g @ g.conj().T
    return rho / np.trace(rho)


def sample_state(dims, rng, distribution='mixed'):
    """Mixed distribution broadens visibility coverage; no hidden label queries."""
    dA, dB = dims
    if distribution == 'ghz_w_3x3':
        if tuple(dims) != (3, 3):
            raise ValueError('ghz_w_3x3 requires dims=(3,3)')
        pg, pw, _ = rng.dirichlet(np.ones(3))
        return ghz_w_3x3(pg, pw)
    if distribution == 'ginibre':
        return random_density_matrix(dA*dB, rng)
    if distribution == 'horodecki':
        if dims != (3, 3):
            raise ValueError('Horodecki family requires dims=(3,3)')
        return horodecki_3x3(rng.uniform())
    if distribution != 'mixed':
        raise ValueError(f'Unknown distribution: {distribution}')
    branch = rng.integers(3)
    if branch == 0:
        return random_density_matrix(dA*dB, rng)
    if branch == 1:
        # Convex combinations of product states are physically separable.
        weights = rng.dirichlet(np.ones(4))
        return sum(w * np.kron(random_density_matrix(dA, rng), random_density_matrix(dB, rng))
                   for w in weights)
    v = rng.normal(size=dA*dB) + 1j*rng.normal(size=dA*dB)
    v /= np.linalg.norm(v)
    p = rng.uniform()
    return p*np.outer(v, v.conj()) + (1-p)*np.eye(dA*dB)/(dA*dB)


def generate_dataset(oracle, state_generator, n_samples, dims=(3,3),
                     feature_fn=extract_features_advanced, save_path=None, seed=42):
    dims = tuple(dims)
    if n_samples < 1:
        raise ValueError('n_samples must be positive')
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_samples):
        if callable(state_generator):
            rho = state_generator(np.prod(dims), rng=rng)
        else:
            family = rng.choice(state_generator)
            if family == 'horodecki':
                rho = sample_state(tuple(dims), rng, 'horodecki')
            elif family in ('werner', 'isotropic'):
                if dims[0] != dims[1] or (family == 'werner' and dims != (2,2)):
                    raise ValueError(f'{family} incompatible with dims={dims}')
                fn = werner_state if family == 'werner' else isotropic_state
                rho = fn(rng.uniform(), d=dims[0])
            elif family == 'random':
                rho = random_density_matrix(np.prod(dims), rng)
            else:
                raise ValueError(f'Unknown family: {family}')
        validate_density(rho, dims)
        result = oracle.query(rho, *dims)
        chi = result['chi']
        if not np.isfinite(chi):
            raise RuntimeError('Oracle returned a nonfinite training target')
        X.append(feature_fn(rho, dims=dims) if feature_fn is extract_features_advanced else feature_fn(rho))
        y.append(chi)
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(f'{save_path}_X.npy', X)
        np.save(f'{save_path}_y.npy', y)
    return X, y
