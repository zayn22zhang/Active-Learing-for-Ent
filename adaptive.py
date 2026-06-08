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
    return [random_pure_state(d, rng) for _ in range(N)]


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

    try:
        prob.solve(solver=cp.MOSEK, verbose=verbose)
    except:
        prob.solve(solver=cp.SCS, verbose=verbose, eps=1e-6, max_iters=20000)

    if prob.status in ['optimal', 'optimal_inaccurate']:
        chi_val = float(t.value) if t.value is not None else 0.0
        tau_vals = [tau[lam].value if tau[lam].value is not None else np.zeros((dB, dB), dtype=complex)
                    for lam in range(N)]
        return chi_val, tau_vals
    else:
        return 0.0, [np.eye(dB, dtype=complex) / dB for _ in range(N)]


def normalise_tau(tau_list):
    """Convert τ̃ matrices to normalized density matrices (polytope vertices)."""
    polytope = []
    for tau in tau_list:
        tr = np.real(np.trace(tau))
        if tr > 1e-9:
            polytope.append(tau / tr)
        else:
            d = tau.shape[0]
            polytope.append(np.eye(d, dtype=complex) / d)
    return polytope


def adaptive_polytope_bipartite(rho_AB, dA, dB, N=200, max_iter=15, tol=1e-4,
                                 verbose=False, seed=None):
    """
    Main adaptive polytope algorithm for bipartite separability.
    Returns (χ, history) where χ is a lower bound on the true visibility.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    polytope_A = random_inner_polytope(dA, N, rng)
    rho = rho_AB.copy()

    chi_prev = 0.0
    history = []

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
        history.append(chi_current)

        if verbose:
            print(f"  Iter {iteration+1:2d}: χ₁={chi1:.5f}, χ₂={chi2:.5f} -> χ={chi_current:.5f}")

        if abs(chi_current - chi_prev) < tol and iteration > 0:
            if verbose:
                print(f"  Converged after {iteration+1} iterations.")
            break
        chi_prev = chi_current

    return chi_current, history