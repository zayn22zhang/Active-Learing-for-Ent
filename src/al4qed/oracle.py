"""Visibility target providers, separated from entanglement certificates.

An inner-polytope value below one is NOT evidence of entanglement.
Finite-budget polytope targets are numerical estimates of lower bounds,
not ground-truth visibility and not ML-generated certificates.
"""
from enum import Enum
from pathlib import Path
import hashlib
import importlib
import json
import time
import warnings
import numpy as np
try:
    from .dataset import validate_density
except ImportError:
    from dataset import validate_density


class OracleLabel(Enum):
    ENTANGLED = 0
    UNKNOWN = 1
    SEPARABLE = 2


def partial_transpose(rho, dims):
    a, b = dims
    return rho.reshape(a,b,a,b).transpose(0,3,2,1).reshape(a*b,a*b)


def ppt_visibility(rho, dims):
    """Upper bound in general; exact capped SEP visibility in 2x2/2x3."""
    lowest = float(np.linalg.eigvalsh(partial_transpose(rho, dims)).min())
    return (1.0 if lowest >= 0 else 1.0/(1.0-np.prod(dims)*lowest)), lowest


class AdaptivePolytopeOracle:
    def __init__(self, N=200, max_iter=15, tol=1e-4, ent_threshold=None,
                 backend='local', external=None, backend_version=None,
                 cache_dir=None, seed=42, certificate_tol=1e-8):
        if N < 2 or max_iter < 1 or tol <= 0 or certificate_tol <= 0:
            raise ValueError('Invalid oracle settings')
        if backend not in ('local','ppt','external'):
            raise ValueError('backend must be local, ppt, or external')
        if ent_threshold is not None:
            warnings.warn('ent_threshold is ignored: a lower bound cannot certify entanglement',
                          DeprecationWarning, stacklevel=2)
        self.N, self.max_iter, self.tol = N, max_iter, tol
        self.backend, self.seed = backend, seed
        self.certificate_tol = certificate_tol
        self.external = external
        self.backend_version = backend_version or ('local-validated-v5' if backend == 'local' else 'ppt-exact-v1')
        if backend == 'external':
            if not external or not backend_version:
                raise ValueError('external backend requires module:function and backend_version (commit/settings)')
            module, function = external.split(':', 1)
            self.provider = getattr(importlib.import_module(module), function)
            if module.endswith('ohst_bridge'):
                bridge = importlib.import_module(module)
                import os
                expected = bridge.BRIDGE_VERSION+'-'+os.environ.get('OHST_SOLVER','SCS').upper()
                if backend_version != expected:
                    raise ValueError(f'Use --backend-version {expected} for the supplied Ohst bridge')
        self.solver_stack = None
        if backend == 'external' and module.endswith('ohst_bridge'):
            self.solver_stack = bridge.provenance()
        if backend == 'local':
            import cvxpy as cp
            self.solver_stack = dict(cvxpy=cp.__version__, installed=sorted(cp.installed_solvers()))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.requests = self.backend_calls = self.cache_hits = 0
        self.elapsed_seconds = 0.0

    def config(self):
        return dict(backend=self.backend, external=self.external, version=self.backend_version,
                    N=self.N, max_iter=self.max_iter, tol=self.tol,
                    certificate_tol=self.certificate_tol, seed=self.seed, solver_stack=self.solver_stack, schema=2)

    def query(self, rho, dA, dB, seed=None):
        dims = (dA, dB)
        rho = validate_density(rho, dims)
        config_bytes = json.dumps(self.config(), sort_keys=True).encode()
        digest = hashlib.sha256(rho.astype('<c16').tobytes() + str(dims).encode() + config_bytes).hexdigest()
        # State-specific seeds make labels independent of query ordering and strategy.
        state_seed = int(digest[:8], 16) if seed is None else int(seed)
        key = hashlib.sha256((digest+str(state_seed)).encode()).hexdigest()
        path = self.cache_dir / f'{key}.json' if self.cache_dir else None
        self.requests += 1
        if path and path.exists():
            result = json.loads(path.read_text())
            result['label'] = OracleLabel[result['label']]
            result['cache_hit'] = True
            self.cache_hits += 1
            return result
        start = time.perf_counter()
        self.backend_calls += 1
        upper, min_pt = ppt_visibility(rho, dims)
        label, evidence = OracleLabel.UNKNOWN, 'no classification certificate'
        if min_pt < -self.certificate_tol:
            label, evidence = OracleLabel.ENTANGLED, 'negative partial transpose (numerical tolerance)'
        elif sorted(dims) in ([2,2], [2,3]) and min_pt >= 0:
            label, evidence = OracleLabel.SEPARABLE, 'PPT sufficient in 2x2/2x3 (numerical)'
        if self.backend == 'ppt':
            if sorted(dims) not in ([2,2], [2,3]):
                raise ValueError('PPT is not an exact separability oracle in these dimensions')
            raw = dict(chi=upper, history=[upper], converged=True, status='analytic',
                       target_kind='exact_capped_visibility')
        elif self.backend == 'external':
            raw = self.provider(rho, dims=dims, seed=state_seed, N=self.N,
                                max_iter=self.max_iter, tol=self.tol)
            if not isinstance(raw, dict) or raw.get('status') not in ('optimal','validated','analytic'):
                raise RuntimeError('External backend must return a successful, validated result dictionary')
            if raw.get('target_kind') not in ('inner_polytope_lower_estimate','exact_capped_visibility'):
                raise ValueError('External target_kind missing or unsupported; outer bounds are not lower targets')
        else:
            try:
                from .adaptive import adaptive_polytope_bipartite
            except ImportError:
                from adaptive import adaptive_polytope_bipartite
            value, history, converged = adaptive_polytope_bipartite(
                rho, dA, dB, N=self.N, max_iter=self.max_iter, tol=self.tol, seed=state_seed)
            raw = dict(chi=value, history=history, converged=converged,
                       status='validated', target_kind='inner_polytope_lower_estimate')
        chi = float(raw['chi'])
        if not np.isfinite(chi) or chi < -1e-6:
            raise RuntimeError('Oracle failed or returned invalid visibility')
        # Ohst uses t>=0; this project explicitly learns min(1,t).
        chi = float(np.clip(chi, 0, 1))
        if chi > upper + 1e-4:
            raise RuntimeError('Lower estimate exceeds the PPT upper bound; inspect solver output')
        result = dict(chi=chi, chi_upper_ppt=upper, label=label, evidence=evidence,
                      min_pt_eigenvalue=min_pt, target_kind=raw['target_kind'],
                      status=raw['status'], history=[float(v) for v in raw.get('history', [])],
                      converged=bool(raw.get('converged', False)),
                      seed=state_seed, backend=self.backend, backend_version=self.backend_version,
                      N=self.N, max_iter=self.max_iter, cache_hit=False,
                      seconds=time.perf_counter()-start)
        result['iterations'] = len(result['history'])
        for diagnostic in ('validation_residual', 'upstream_chi'):
            if diagnostic in raw:
                result[diagnostic] = float(raw[diagnostic])
        self.elapsed_seconds += result['seconds']
        if path:
            encoded = dict(result, label=result['label'].name)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(encoded, indent=2, allow_nan=False))
            temporary.replace(path)
        return result

    def visibility(self, rho, dA, dB, seed=None):
        result = self.query(rho, dA, dB, seed)
        return result['chi'], result['history'], result['converged']

    def get_chi(self, rho, dA, dB, seed=None):
        return self.query(rho, dA, dB, seed)['chi']
