"""Adapter to the actual Julia/Convex implementation in Ohst's cited repository.

Uses pinned upstream functions, then validates a primal decomposition on the
adapted polytope. It does not use the inconsistent top-level README wrapper.
One persistent Julia runtime is reused for all queries in a Python process.
"""
import os
from pathlib import Path
import subprocess
import numpy as np

UPSTREAM_COMMIT = '035d401fc239339ae95104170ce15c60dbe26567'
KVANT_COMMIT = '44caf9384075b5db3f3b8efe1e469947f4a08287'
BRIDGE_VERSION = f'ohst-{UPSTREAM_COMMIT}-bridge-v1'
_runtime = None


def _revision(path):
    return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'], text=True).strip()


def _get_runtime():
    global _runtime
    if _runtime is not None:
        return _runtime
    raw = os.environ.get('OHST_REPO')
    if not raw:
        raise RuntimeError('Set OHST_REPO to the pinned quantum-correlations checkout; see README.md')
    repo = Path(raw).expanduser().resolve()
    if _revision(repo) != UPSTREAM_COMMIT or _revision(repo/'lib/kvant') != KVANT_COMMIT:
        raise RuntimeError('Unexpected upstream revision; use setup_ohst.py to obtain the inspected versions')
    for root, path in [(repo,'src/entanglement/EntanglementWithBlochPolytope.jl'),
                       (repo/'lib/kvant','julia/MultiStates.jl')]:
        dirty = subprocess.check_output(['git','-C',str(root),'diff','HEAD','--',path],text=True)
        if dirty:
            raise RuntimeError(f'Upstream source was modified: {path}')
    try:
        from juliacall import Main as jl
    except ImportError as exc:
        raise RuntimeError('Install requirements-ohst.txt and initialize Julia dependencies per README.md') from exc
    jl.seval('push!')(jl.LOAD_PATH,str(repo/'lib/kvant/julia'))
    jl.include(str(Path(__file__).with_name('ohst_bridge.jl')))
    jl.seval('Base.include')(jl.AL4QEDOhst, str(repo/'src/entanglement/EntanglementWithBlochPolytope.jl'))
    _runtime = jl.AL4QEDOhst
    return _runtime


def query(rho, *, dims, seed, N, max_iter, tol):
    solver = os.environ.get('OHST_SOLVER','SCS').upper()
    if solver not in ('SCS','MOSEK'):
        raise ValueError('OHST_SOLVER must be SCS or MOSEK')
    if np.allclose(rho,np.eye(len(rho))/len(rho),rtol=0,atol=1e-14):
        # Uncapped upstream visibility is unbounded at I/d; capped target is 1.
        return dict(chi=1.,status='analytic',target_kind='exact_capped_visibility',converged=True,history=[1.])
    raw = _get_runtime().query(np.asarray(rho,dtype=np.complex128),list(dims),
                              int(seed),int(N),int(max_iter),float(tol),solver)
    return dict(chi=float(raw[0]),status='validated',target_kind='inner_polytope_lower_estimate',
                converged=False, history=[], validation_residual=float(raw[1]),
                upstream_chi=float(raw[2]))


def provenance():
    import hashlib
    versions = tuple(str(v) for v in _get_runtime().runtime_info())
    return dict(julia=versions[0], Convex=versions[1], SCS=versions[2], MosekTools=versions[3],
                bridge_sha256=hashlib.sha256(Path(__file__).read_bytes()+
                    Path(__file__).with_suffix('.jl').read_bytes()).hexdigest())
