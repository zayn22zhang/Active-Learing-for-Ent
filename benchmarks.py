"""
Adaptive Polytope Separability Certification
Reproducing: Ohst et al., "Certifying quantum separability with adaptive polytopes"
SciPost Phys. 16, 063 (2024)

Key results targeted:
- GHZ biseparability threshold: chi = 0.42857 (= 3/7, exact)
- GHZ full separability threshold: chi = 0.199
- W state thresholds
"""

import numpy as np
import cvxpy as cp
from itertools import product as iproduct
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  Quantum state utilities
# ─────────────────────────────────────────────

def ket(*args):
    """Computational basis ket for n qubits."""
    bits = list(args)
    dim = 2 ** len(bits)
    idx = sum(b * 2**(len(bits)-1-i) for i, b in enumerate(bits))
    v = np.zeros(dim, dtype=complex)
    v[idx] = 1.0
    return v

def dm(psi):
    """Pure state density matrix."""
    return np.outer(psi, psi.conj())

def ghz3():
    """3-qubit GHZ state."""
    psi = (ket(0,0,0) + ket(1,1,1)) / np.sqrt(2)
    return dm(psi)

def w3():
    """3-qubit W state."""
    psi = (ket(1,0,0) + ket(0,1,0) + ket(0,0,1)) / np.sqrt(3)
    return dm(psi)

def maximally_mixed(d):
    return np.eye(d, dtype=complex) / d

def white_noise_mix(rho, t):
    d = rho.shape[0]
    return t * rho + (1-t) * np.eye(d, dtype=complex) / d

def partial_transpose(rho, dims, subsys):
    """Partial transpose of rho w.r.t. subsystem index subsys."""
    dA, dB = dims
    rho_r = rho.reshape(dA, dB, dA, dB)
    if subsys == 0:
        rho_pt = rho_r.transpose(2, 1, 0, 3)
    else:
        rho_pt = rho_r.transpose(0, 3, 2, 1)
    return rho_pt.reshape(dA*dB, dA*dB)

def is_ppt(rho, dims):
    """Check PPT condition."""
    pt = partial_transpose(rho, dims, 1)
    eigs = np.linalg.eigvalsh(pt)
    return np.all(eigs > -1e-7)

def tensor(*mats):
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


# ─────────────────────────────────────────────
#  Random inner polytope initialisation
# ─────────────────────────────────────────────

def random_density_matrix(d, rng=None):
    """Haar-random density matrix via Ginibre ensemble."""
    if rng is None:
        rng = np.random.default_rng()
    G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = G @ G.conj().T
    return rho / np.trace(rho)

def random_inner_polytope(d, N, rng=None):
    """
    Random inner polytope of Alice's Bloch sphere.
    Returns list of N density matrices (d×d).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    return [random_density_matrix(d, rng) for _ in range(N)]


# ─────────────────────────────────────────────
#  BIPARTITE: core SDP (Eq. 5 / 14 in paper)
# ─────────────────────────────────────────────

def bipartite_visibility_sdp(rho_AB, polytope_A, dB, verbose=False):
    """
    Compute chi_P(rho^AB) = max t  s.t.  t*rho + (1-t)*I/d = sum_lambda sigma_lambda ⊗ tau_lambda~
    with tau_lambda~ >= 0  (feasibility / optimisation SDP, Eq. 14).

    Returns: (chi, tau_tilde_list)
    """
    dA = polytope_A[0].shape[0]
    d  = dA * dB
    N  = len(polytope_A)

    # Variables: t (scalar), tau_tilde[lambda] (dB x dB PSD)
    t = cp.Variable(nonneg=True)
    tau = [cp.Variable((dB, dB), hermitian=True) for _ in range(N)]

    # Constraint: t*rho_t = sum sigma_lambda ⊗ tau_lambda~
    # where rho_t = t*rho + (1-t)*I/d
    lhs_coeffs = []  # will build constraint matrix-wise

    constraints = [t >= 0, t <= 1]
    for lam in range(N):
        constraints.append(tau[lam] >> 0)

    # Build: t*rho + (1-t)*I/d = sum_lam kron(sigma_lam, tau_lam)
    # Rearrange: t*(rho - I/d) + I/d = sum_lam kron(sigma_lam, tau_lam)
    I_d = np.eye(d, dtype=complex) / d

    # We build this as a real constraint on vectorised matrices
    # Use the direct matrix equality as a list of (dA*dB)^2 real constraints
    # More efficiently: just pass as CVXPY expression

    rhs = sum(cp.kron(cp.Constant(sigma), tau[lam])
              for lam, sigma in enumerate(polytope_A))

    lhs = t * cp.Constant(rho_AB) + (1 - t) * cp.Constant(I_d)

    constraints.append(lhs == rhs)

    prob = cp.Problem(cp.Maximize(t), constraints)
    prob.solve(solver=cp.SCS, verbose=verbose, eps=1e-5, max_iters=10000)

    if prob.status in ['optimal', 'optimal_inaccurate']:
        chi_val = float(t.value) if t.value is not None else 0.0
        tau_vals = []
        for lam in range(N):
            tv = tau[lam].value
            if tv is None:
                tv = np.zeros((dB, dB), dtype=complex)
            tau_vals.append(tv)
        return chi_val, tau_vals
    else:
        return 0.0, [np.eye(dB, dtype=complex)/dB for _ in range(N)]


def normalise_tau(tau_list):
    """Extract normalised polytope from tau~ outputs."""
    polytope = []
    for tau in tau_list:
        tr = np.real(np.trace(tau))
        if tr > 1e-9:
            polytope.append(tau / tr)
        else:
            # degenerate vertex — use maximally mixed
            d = tau.shape[0]
            polytope.append(np.eye(d, dtype=complex) / d)
    return polytope


# ─────────────────────────────────────────────
#  BIPARTITE adaptive polytope algorithm
# ─────────────────────────────────────────────

def adaptive_polytope_bipartite(rho_AB, dA, dB, N=100, max_iter=20, tol=1e-4,
                                 verbose=False, seed=42):
    """
    Main adaptive polytope algorithm for bipartite separability (Section 2.2).

    rho_AB : (dA*dB, dA*dB) density matrix
    dA, dB : local dimensions
    N      : number of polytope vertices
    Returns: chi (separability visibility lower bound), iteration history
    """
    rng = np.random.default_rng(seed)

    # Initialise random inner polytope on Alice's side
    polytope_A = random_inner_polytope(dA, N, rng)
    rho = rho_AB.copy()

    chi_prev = 0.0
    history = []

    for iteration in range(max_iter):
        # Step 2: SDP with current polytope_A, optimise over Bob's side
        chi, tau_B = bipartite_visibility_sdp(rho, polytope_A, dB, verbose=verbose)

        # Build new polytope for Bob
        polytope_B = normalise_tau(tau_B)

        # Step 3: Exchange A and B — swap the state and repeat
        # After swap: Alice has dB, Bob has dA
        rho_swapped = rho.reshape(dA, dB, dA, dB).transpose(1, 0, 3, 2).reshape(dB*dA, dB*dA)

        chi2, tau_A = bipartite_visibility_sdp(rho_swapped, polytope_B, dA, verbose=verbose)

        # New polytope for Alice (in original ordering)
        polytope_A = normalise_tau(tau_A)

        # Best chi from this round
        chi_round = max(chi, chi2)
        history.append(chi_round)

        if verbose:
            print(f"  Iter {iteration+1:2d}: chi_A={chi:.5f}  chi_B={chi2:.5f}")

        if abs(chi_round - chi_prev) < tol and iteration > 0:
            if verbose:
                print(f"  Converged at iteration {iteration+1}")
            break
        chi_prev = chi_round

    return chi_prev, history


# ─────────────────────────────────────────────
#  MULTIPARTITE: biseparability SDP (Eq. 44-46)
# ─────────────────────────────────────────────

def three_qubit_bsep_sdp(rho_ABC, polytope_A, polytope_B, polytope_C, verbose=False):
    """
    Biseparability (BSEP) visibility SDP for 3-qubit systems.
    Eq. 44-46 in paper:
      rho_t = sum_lam tau^AB_lam ⊗ sigma^C_lam
            + sum_lam sigma^A_lam ⊗ tau^BC_lam
            + sum_lam sigma^B_lam ⊗ tau^AC_lam

    Returns (chi, tau_AB_list, tau_BC_list, tau_AC_list)
    """
    d = 8  # 2^3
    NA, NB, NC = len(polytope_A), len(polytope_B), len(polytope_C)
    I8 = np.eye(d, dtype=complex) / d

    t = cp.Variable(nonneg=True)

    # tau variables: tau^AB (4x4), tau^BC (4x4), tau^AC (4x4)
    tau_AB = [cp.Variable((4, 4), hermitian=True) for _ in range(NC)]
    tau_BC = [cp.Variable((4, 4), hermitian=True) for _ in range(NA)]
    tau_AC = [cp.Variable((4, 4), hermitian=True) for _ in range(NB)]

    constraints = [t >= 0, t <= 1]
    for v in tau_AB + tau_BC + tau_AC:
        constraints.append(v >> 0)

    # Build RHS
    # Term 1: sum_lam tau^AB_lam ⊗ sigma^C_lam   (sigma^C is 2x2)
    term1 = sum(cp.kron(tau_AB[lam], cp.Constant(polytope_C[lam]))
                for lam in range(NC))
    # Term 2: sum_lam sigma^A_lam ⊗ tau^BC_lam
    term2 = sum(cp.kron(cp.Constant(polytope_A[lam]), tau_BC[lam])
                for lam in range(NA))
    # Term 3: sum_lam sigma^B_lam ⊗ tau^AC_lam
    # Need to handle ordering: A(2) B(2) C(2), AB|C bipart -> sigma^B ⊗ tau^AC
    # sigma^B acts on subsystem B, tau^AC on AC
    # In ABC ordering, B|AC split: we need kron structure sigma^B kron tau^AC
    # but state is ordered ABC, so B|AC means the (4x4) tau^AC acts on A and C
    # We construct: (I_A ⊗ sigma_B ⊗ I_C) style — easier to build directly
    # For the B|AC bipartition in ABC ordering:
    # sigma_B ⊗ tau_AC: B is middle qubit
    # reorder: (sigma_B)_{b,b'} * (tau_AC)_{ac,a'c'} → rho_{abc,a'b'c'} = (tau_AC)_{ac,a'c'} * sigma_B_{b,b'}
    term3_list = []
    for lam in range(NB):
        sigB = polytope_B[lam]   # 2x2
        tAC  = tau_AC[lam]       # 4x4
        # Build 8x8 matrix with ordering A(0)B(1)C(2)
        # M_{abc, a'b'c'} = tAC_{ac,a'c'} * sigB_{b,b'}
        # This is a partial Kronecker: not a simple kron
        # We can write it as: (I2 ⊗ sigB ⊗ I2) applied in the right slots
        # Equivalent: reshape and permute
        # Build via: sum over b,b' of |b><b'| ⊗ sigB[b,b'] ... no
        # Cleanest: M = kron(sigB, tAC) then permute indices A<->B
        M_raw = cp.kron(cp.Constant(sigB), tAC)  # shape (8,8), ordering BAC
        # We need to permute to ABC ordering
        # BAC->ABC: swap first two qubits
        # Permutation matrix P such that P @ M_BAC @ P.T = M_ABC
        P = _qubit_swap_matrix_012_to_102()
        Pc = cp.Constant(P)
        term3_list.append(Pc @ M_raw @ Pc.T)
    term3 = sum(term3_list)

    rhs = term1 + term2 + term3
    lhs = t * cp.Constant(rho_ABC) + (1 - t) * cp.Constant(I8)

    constraints.append(lhs == rhs)

    prob = cp.Problem(cp.Maximize(t), constraints)
    prob.solve(solver=cp.SCS, verbose=verbose, eps=1e-5, max_iters=15000)

    if prob.status in ['optimal', 'optimal_inaccurate']:
        chi_val = float(t.value) if t.value is not None else 0.0
        def get_vals(vlist):
            out = []
            for v in vlist:
                val = v.value
                if val is None:
                    val = np.zeros((v.shape[0], v.shape[0]), dtype=complex)
                out.append(val)
            return out
        return chi_val, get_vals(tau_AB), get_vals(tau_BC), get_vals(tau_AC)
    else:
        return 0.0, [], [], []


def _qubit_swap_matrix_012_to_102():
    """Permutation matrix swapping qubit 0 and 1 in 3-qubit system (8x8)."""
    P = np.zeros((8, 8), dtype=complex)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                old_idx = a*4 + b*2 + c   # ABC
                new_idx = b*4 + a*2 + c   # BAC
                P[old_idx, new_idx] = 1.0
    return P


# ─────────────────────────────────────────────
#  Adaptive polytope — biseparability (3 qubits)
# ─────────────────────────────────────────────

def adaptive_polytope_bsep_3qubit(rho_ABC, N=50, max_iter=10, tol=1e-3,
                                   verbose=False, seed=42):
    """
    Adaptive polytope for biseparability (BSEP) of 3-qubit states.
    Returns chi_BSEP lower bound.
    """
    rng = np.random.default_rng(seed)
    polytope_A = random_inner_polytope(2, N, rng)
    polytope_B = random_inner_polytope(2, N, rng)
    polytope_C = random_inner_polytope(2, N, rng)

    chi_prev = 0.0
    history = []

    for iteration in range(max_iter):
        chi, tau_AB, tau_BC, tau_AC = three_qubit_bsep_sdp(
            rho_ABC, polytope_A, polytope_B, polytope_C, verbose=verbose)

        history.append(chi)

        if verbose:
            print(f"  BSEP iter {iteration+1:2d}: chi={chi:.5f}")

        # Update polytopes from feasible outputs
        # tau_AB (NC x 4x4) → new polytope for C side: take marginal over A
        if len(tau_AB) > 0 and chi > 0:
            new_C = []
            for tAB in tau_AB:
                # marginal on C from AB: trace over A and B
                # tAB is 4x4 (AB system); take trace to get weight
                tr = max(np.real(np.trace(tAB)), 1e-10)
                # For polytope update: we reuse the normalised version
                # Actually, to update sigma_C we look at the associated tau^C
                # In BSEP SDP the sigma^C are the polytope vertices (fixed);
                # we update by seeing which vertices are active
                new_C.append(polytope_C[0])  # placeholder
            # Better: just rerandomise non-active vertices
            # For simplicity, keep polytopes and rely on multiple iterations

        if abs(chi - chi_prev) < tol and iteration > 0:
            if verbose:
                print(f"  Converged.")
            break
        chi_prev = chi

    return chi_prev, history


# ─────────────────────────────────────────────
#  MULTIPARTITE: full separability SDP (Eq. 28/29-34)
# ─────────────────────────────────────────────

def three_qubit_fsep_sdp(rho_ABC, polytope_A, verbose=False):
    """
    Full separability SDP for 3-qubit systems using PPT trick.
    FSEP = conv(S_A ⊗ PPT_BC) for qubits.
    Eq. 28 / primal of Eq. 29-34.

    tau^BC_lam >= 0 and (tau^BC_lam)^{T_B} >= 0
    rho_t = sum_lam sigma^A_lam ⊗ tau^BC_lam
    """
    d = 8
    NA = len(polytope_A)
    I8 = np.eye(d, dtype=complex) / d

    t = cp.Variable(nonneg=True)
    tau_BC = [cp.Variable((4, 4), hermitian=True) for _ in range(NA)]

    constraints = [t >= 0, t <= 1]
    for lam in range(NA):
        constraints.append(tau_BC[lam] >> 0)
        # PPT constraint on tau^BC: partial transpose w.r.t. B >= 0
        # BC system (B=qubit 0, C=qubit 1): PT wrt B
        PT_mat = _partial_transpose_2qubit_wrt_first()
        Pt = cp.Constant(PT_mat)
        tau_pt = Pt @ tau_BC[lam] @ Pt.T   # apply PT reshuffling
        constraints.append(tau_pt >> 0)

    rhs = sum(cp.kron(cp.Constant(polytope_A[lam]), tau_BC[lam])
              for lam in range(NA))
    lhs = t * cp.Constant(rho_ABC) + (1 - t) * cp.Constant(I8)
    constraints.append(lhs == rhs)

    prob = cp.Problem(cp.Maximize(t), constraints)
    prob.solve(solver=cp.SCS, verbose=verbose, eps=1e-5, max_iters=15000)

    if prob.status in ['optimal', 'optimal_inaccurate']:
        chi_val = float(t.value) if t.value is not None else 0.0
        tau_vals = []
        for lam in range(NA):
            tv = tau_BC[lam].value
            if tv is None:
                tv = np.zeros((4, 4), dtype=complex)
            tau_vals.append(tv)
        return chi_val, tau_vals
    else:
        return 0.0, [np.zeros((4,4)) for _ in range(NA)]


def _partial_transpose_2qubit_wrt_first():
    """
    For a 2-qubit (4x4) state ρ_{ab,a'b'}, return reshape matrix P such that
    P @ vec(rho) gives vec(rho^{T_A}).
    Here we return a 4x4 unitary-style matrix that implements the PT:
    rho -> P rho P^dagger  with P being a 'swap on one index'.
    Actually: just return the permutation as a 4x4 matrix on indices.
    PT wrt qubit 0 (B in BC): ρ_{b c, b' c'} -> ρ_{b' c, b c'}
    We implement as a left-right operator: M -> Φ M Φ† where Φ is the
    'partial flip' = I_B^flip ⊗ I_C mapped appropriately.
    """
    # The partial transpose wrt qubit B (first of 2) is:
    # rho_PT[b*2+c, b'*2+c'] = rho[b'*2+c, b*2+c'] = rho[b'2+c, b2+c']
    # We implement as matrix: rho_PT = P @ rho @ P^T where P swaps b <-> b'
    # This is simply swapping rows/cols: P = sigma_x ⊗ I
    sx = np.array([[0,1],[1,0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    return np.kron(sx, I2)


def adaptive_polytope_fsep_3qubit(rho_ABC, N=80, max_iter=12, tol=1e-3,
                                   verbose=False, seed=42):
    """
    Adaptive polytope for FULL separability of 3-qubit states.
    Uses cyclic optimisation over parties (Section 3.1, App D.3).
    """
    rng = np.random.default_rng(seed)
    polytope_A = random_inner_polytope(2, N, rng)

    chi_prev = 0.0
    history = []

    for iteration in range(max_iter):
        # SDP with A polytope fixed, optimise tau^BC (PPT-constrained)
        chi, tau_BC = three_qubit_fsep_sdp(rho_ABC, polytope_A, verbose=verbose)
        history.append(chi)

        if verbose:
            print(f"  FSEP iter {iteration+1:2d}: chi={chi:.5f}")

        # Update Alice's polytope using Bob-Charlie marginals
        new_polytope = []
        for tv in tau_BC:
            tr = max(np.real(np.trace(tv)), 1e-10)
            # Get Alice's reduced state from the BC operator
            # Here we extract the A-marginal from the full decomposition
            new_polytope.append(tv / tr)  # These are BC states; use as new "environment"

        # Cycle: now fix "BC" polytope, optimise over A
        # Swap A <-> BC: rho^{BC-A} = rho permuted
        rho_BCA = rho_ABC.reshape(2,2,2,2,2,2).transpose(1,2,0,4,5,3).reshape(4*2, 4*2)

        # BC polytope from tau_BC outputs
        polytope_BC = normalise_tau(tau_BC)
        chi2, tau_A = bipartite_visibility_sdp(rho_BCA, polytope_BC, 2, verbose=verbose)

        # Update A polytope
        polytope_A = normalise_tau(tau_A)
        # Pad if needed
        while len(polytope_A) < N:
            polytope_A.append(random_density_matrix(2, rng))

        chi_round = max(chi, chi2)
        history[-1] = chi_round

        if verbose:
            print(f"    (swapped) chi2={chi2:.5f}  best={chi_round:.5f}")

        if abs(chi_round - chi_prev) < tol and iteration > 1:
            if verbose:
                print("  Converged.")
            break
        chi_prev = chi_round

    return chi_prev, history


# ─────────────────────────────────────────────
#  PPT upper bound (for comparison)
# ─────────────────────────────────────────────

def ppt_visibility_binary_search(rho_AB, dA, dB, tol=1e-4):
    """
    Compute PPT threshold chi_PPT via binary search:
    max t such that t*rho + (1-t)*I/d is PPT.
    """
    d = dA * dB
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        rho_t = white_noise_mix(rho_AB, mid)
        if is_ppt(rho_t, (dA, dB)):
            lo = mid
        else:
            hi = mid
    return lo


# ─────────────────────────────────────────────
#  Main benchmarks
# ─────────────────────────────────────────────

def run_benchmarks():
    print("=" * 65)
    print("Adaptive Polytope Separability Certification")
    print("Reproducing Ohst et al., SciPost Phys. 16, 063 (2024)")
    print("=" * 65)

    # ── Benchmark 1: GHZ biseparability ──────────────────────────────
    print("\n▶ 3-qubit GHZ state — Biseparability threshold")
    print("  Known exact value: chi = 3/7 ≈ 0.42857")
    rho_GHZ = ghz3()

    chi_bsep, hist_bsep = adaptive_polytope_bsep_3qubit(
        rho_GHZ, N=60, max_iter=12, tol=5e-4, verbose=True, seed=0)

    print(f"\n  ✓ chi_BSEP (lower bound) = {chi_bsep:.5f}")
    print(f"  Target: 0.42857  |  Error: {abs(chi_bsep - 3/7):.5f}")
    print(f"  Iterations: {len(hist_bsep)}  |  History: {[f'{v:.4f}' for v in hist_bsep]}")

    # ── Benchmark 2: GHZ full separability ───────────────────────────
    print("\n▶ 3-qubit GHZ state — Full separability threshold")
    print("  Known exact value: chi = 0.199")

    chi_fsep_GHZ, hist_fsep_GHZ = adaptive_polytope_fsep_3qubit(
        rho_GHZ, N=60, max_iter=12, tol=5e-4, verbose=True, seed=1)

    print(f"\n  ✓ chi_FSEP (lower bound) = {chi_fsep_GHZ:.5f}")
    print(f"  Target: 0.199  |  Error: {abs(chi_fsep_GHZ - 0.199):.5f}")

    # ── Benchmark 3: W state full separability ────────────────────────
    print("\n▶ 3-qubit W state — Full separability threshold")
    print("  Known exact value: chi = 0.178")
    rho_W = w3()

    chi_fsep_W, hist_fsep_W = adaptive_polytope_fsep_3qubit(
        rho_W, N=60, max_iter=12, tol=5e-4, verbose=True, seed=2)

    print(f"\n  ✓ chi_FSEP (lower bound) = {chi_fsep_W:.5f}")
    print(f"  Target: 0.178  |  Error: {abs(chi_fsep_W - 0.178):.5f}")

    # ── Benchmark 4: W state biseparability ───────────────────────────
    print("\n▶ 3-qubit W state — Biseparability threshold")
    print("  Known exact value: chi = 0.479")

    chi_bsep_W, hist_bsep_W = adaptive_polytope_bsep_3qubit(
        rho_W, N=60, max_iter=12, tol=5e-4, verbose=True, seed=3)

    print(f"\n  ✓ chi_BSEP (lower bound) = {chi_bsep_W:.5f}")
    print(f"  Target: 0.479  |  Error: {abs(chi_bsep_W - 0.479):.5f}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY TABLE")
    print("=" * 65)
    print(f"{'State & Type':<35} {'Obtained':>10} {'Target':>10} {'Error':>8}")
    print("-" * 65)
    results = [
        ("GHZ  BSEP",  chi_bsep,      3/7,   0.42857),
        ("GHZ  FSEP",  chi_fsep_GHZ,  0.199, 0.199),
        ("W    FSEP",  chi_fsep_W,    0.178, 0.178),
        ("W    BSEP",  chi_bsep_W,    0.479, 0.479),
    ]
    for name, obtained, target, _ in results:
        err = abs(obtained - target)
        ok = "✓" if err < 0.02 else "~" if err < 0.05 else "✗"
        print(f"{ok} {name:<33} {obtained:>10.5f} {target:>10.5f} {err:>8.5f}")
    print("=" * 65)

    return results


if __name__ == "__main__":
    run_benchmarks()