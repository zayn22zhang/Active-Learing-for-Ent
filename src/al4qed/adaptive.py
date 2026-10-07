"""
Adaptive polytope algorithm for bipartite separability certification.
Ohst et al., SciPost Phys. 16, 063 (2024)
"""

import numpy as np
import cvxpy as cp


def partial_transpose(rho, dims, subsys):
    dA, dB = dims
    rho_r = rho.reshape(dA, dB, dA, dB)
    if subsys == 0:
        rho_pt = rho_r.transpose(2, 1, 0, 3)
    else:
        rho_pt = rho_r.transpose(0, 3, 2, 1)
    return rho_pt.reshape(dA * dB, dA * dB)


def swap_systems(rho, dA, dB):
    """Swap A and B systems: ρ_AB -> ρ_BA"""
    return rho.reshape(dA, dB, dA, dB).transpose(1, 0, 3, 2).reshape(dA * dB, dA * dB)


def random_pure_state(d, rng=None):
    """Haar-random pure state density matrix (rank 1)."""
    if rng is None:
        rng = np.random.default_rng()
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())


def random_inner_polytope(d, N, rng=None):
    """Random inner polytope of Bloch sphere using pure states."""
    if rng is None:
        rng = np.random.default_rng()
    if N < d:
        raise ValueError('N must be at least the local dimension')
    # Include a basis simplex so I/d is always in the initial polytope.
    basis = [np.diag(np.eye(d)[i]).astype(complex) for i in range(d)]
    vertices = basis[:]
    # Whole orthonormal bases put I/d in each block's convex hull.
    # This avoids an initial polytope that touches I/d only on one face.
    while len(vertices)+d <= N:
        g = rng.normal(size=(d,d)) + 1j*rng.normal(size=(d,d))
        q,_ = np.linalg.qr(g)
        vertices.extend(np.outer(q[:,i],q[:,i].conj()) for i in range(d))
    vertices.extend(random_pure_state(d,rng) for _ in range(N-len(vertices)))
    return vertices


def bipartite_visibility_sdp(rho_AB, polytope_A, dB, verbose=False):
    """
    Compute χ_P(ρ^AB) = max t such that:
    t ρ + (1-t)I/d = Σ_λ σ_λ ⊗ τ_λ
    Returns (χ, τ_list)
    """
    dA = polytope_A[0].shape[0]
    d = dA * dB
    N = len(polytope_A)

    t = cp.Variable(nonneg=True)
    tau = [cp.Variable((dB, dB), hermitian=True) for _ in range(N)]

    constraints = [t >= 0, t <= 1]
    for lam in range(N):
        constraints.append(tau[lam] >> 0)

    I_d = np.eye(d, dtype=complex) / d
    lhs = t * cp.Constant(rho_AB) + (1 - t) * cp.Constant(I_d)
    rhs = sum(cp.kron(cp.Constant(polytope_A[lam]), tau[lam]) for lam in range(N))
    constraints.append(lhs == rhs)
    constraints.append(sum(cp.trace(tau[lam]) for lam in range(N)) == 1.0)

    prob = cp.Problem(cp.Maximize(t), constraints)

    installed = cp.installed_solvers()
    failures = []
    for solver in (cp.MOSEK, cp.CLARABEL, cp.SCS):
        if solver not in installed:
            continue
        try:
            options = dict(eps=1e-7, max_iters=50000) if solver == cp.SCS else {}
            prob.solve(solver=solver, verbose=verbose, **options)
        except cp.error.SolverError as exc:
            failures.append(f'{solver}: {exc}')
            continue
        if prob.status != cp.OPTIMAL or t.value is None or any(x.value is None for x in tau):
            failures.append(f'{solver}: {prob.status}')
            continue
        value = float(t.value)
        values = [x.value for x in tau]
        reconstructed = sum(np.kron(a,b) for a,b in zip(polytope_A, values))
        target = value*rho_AB + (1-value)*np.eye(d)/d
        residual = np.linalg.norm(target-reconstructed, 'fro')
        min_eig = min(np.linalg.eigvalsh((x+x.conj().T)/2).min() for x in values)
        if (not np.isfinite(value) or not np.isfinite(residual) or not np.isfinite(min_eig)
                or value < -1e-6 or value > 1+1e-6 or residual > 1e-5 or min_eig < -1e-6):
            failures.append(f'{solver}: residual={residual}, min_eig={min_eig}')
            continue
        return float(np.clip(value, 0, 1)), values
    raise RuntimeError('No solver returned a validated optimal solution: '+'; '.join(failures))


def normalise_tau(tau_list):
    """Convert τ̃ matrices to normalized density matrices (polytope vertices)."""
    polytope = []
    for tau in tau_list:
        tr = np.real(np.trace(tau))
        if tr > 1e-9:
            eigenvalues, eigenvectors = np.linalg.eigh((tau+tau.conj().T)/2)
            eigenvalues = np.maximum(eigenvalues, 0)
            vertex = (eigenvectors * eigenvalues) @ eigenvectors.conj().T
            polytope.append(vertex / np.trace(vertex))
        else:
            d = tau.shape[0]
            polytope.append(np.eye(d, dtype=complex) / d)
    return polytope


def adaptive_polytope_bipartite(rho_AB, dA, dB, N=200, max_iter=15, tol=1e-4,
                                 verbose=False, seed=None):
    """
    Main adaptive polytope algorithm for bipartite separability.
    
    Returns
    -------
    chi : float
        Lower bound on true visibility
    history : list
        Convergence history
    converged : bool
        True if converged before max_iter
    """
    if max_iter < 1 or N < max(dA, dB):
        raise ValueError('Require max_iter >=1 and N >= max(dA,dB)')
    best_chi = 0.0
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    polytope_A = random_inner_polytope(dA, N, rng)
    rho = rho_AB.copy()

    chi_prev = 0.0
    history = []
    converged = False

    for iteration in range(max_iter):
        # Step 1: fix Alice polytope, optimize over Bob
        chi1, tau_B = bipartite_visibility_sdp(rho, polytope_A, dB, verbose=verbose)
        polytope_B = normalise_tau(tau_B)

        # Step 2: swap systems and fix Bob polytope, optimize over Alice
        rho_swapped = swap_systems(rho, dA, dB)
        chi2, tau_A = bipartite_visibility_sdp(rho_swapped, polytope_B, dA, verbose=verbose)

        # Update Alice polytope
        polytope_A_new = normalise_tau(tau_A)
        while len(polytope_A_new) < N:
            polytope_A_new.append(random_pure_state(dA, rng))
        polytope_A = polytope_A_new[:N]

        # Current visibility = χ₂ (after swap)
        chi_current = chi2
        best_chi = max(best_chi, chi1, chi2)
        history.append(chi_current)

        if verbose:
            print(f"  Iter {iteration+1:2d}: χ₁={chi1:.5f}, χ₂={chi2:.5f} -> χ={chi_current:.5f}")

        if abs(chi_current - chi_prev) < tol and iteration > 0:
            converged = True
            if verbose:
                print(f"  Converged after {iteration+1} iterations.")
            break
        chi_prev = chi_current

    return best_chi, history, converged