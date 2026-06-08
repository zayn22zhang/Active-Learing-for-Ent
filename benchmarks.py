"""
Adaptive Polytope Separability Certification – MULTI-SEED ANALYSIS
Diagnosing χ(a) fluctuations: test multiple initial polytopes per a
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================
#  Core algorithms (same as final version)
# ============================================================

def partial_transpose(rho, dims, subsys):
    dA, dB = dims
    rho_r = rho.reshape(dA, dB, dA, dB)
    if subsys == 0:
        rho_pt = rho_r.transpose(2, 1, 0, 3)
    else:
        rho_pt = rho_r.transpose(0, 3, 2, 1)
    return rho_pt.reshape(dA * dB, dA * dB)

def swap_systems(rho, dA, dB):
    return rho.reshape(dA, dB, dA, dB).transpose(1, 0, 3, 2).reshape(dA * dB, dA * dB)

def random_pure_state(d, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())

def random_inner_polytope(d, N, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return [random_pure_state(d, rng) for _ in range(N)]

def bipartite_visibility_sdp(rho_AB, polytope_A, dB, verbose=False):
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
    polytope = []
    for tau in tau_list:
        tr = np.real(np.trace(tau))
        if tr > 1e-9:
            polytope.append(tau / tr)
        else:
            d = tau.shape[0]
            polytope.append(np.eye(d, dtype=complex) / d)
    return polytope

def adaptive_polytope_bipartite(rho_AB, dA, dB, N=100, max_iter=20, tol=1e-4,
                                 verbose=False, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    polytope_A = random_inner_polytope(dA, N, rng)
    rho = rho_AB.copy()

    chi_prev = 0.0
    history = []

    for iteration in range(max_iter):
        chi1, tau_B = bipartite_visibility_sdp(rho, polytope_A, dB, verbose=verbose)
        polytope_B = normalise_tau(tau_B)

        rho_swapped = swap_systems(rho, dA, dB)
        chi2, tau_A = bipartite_visibility_sdp(rho_swapped, polytope_B, dA, verbose=verbose)
        polytope_A_new = normalise_tau(tau_A)
        
        chi_current = chi2
        history.append(chi_current)

        while len(polytope_A_new) < N:
            polytope_A_new.append(random_pure_state(dA, rng))
        polytope_A = polytope_A_new[:N]

        if abs(chi_current - chi_prev) < tol and iteration > 0:
            break
        chi_prev = chi_current

    return chi_current, history


# ============================================================
#  Horodecki 3x3 – Eq. (48)
# ============================================================

def horodecki_3x3(a):
    N = 8*a + 1
    b = np.sqrt(1 - a*a) / 2

    M = np.array([
        [a, 0, 0, 0, a, 0, 0, 0, a],
        [0, a, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, a, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, a, 0, 0, 0, 0, 0],
        [a, 0, 0, 0, a, 0, 0, 0, a],
        [0, 0, 0, 0, 0, a, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, (1+a)/2, 0, b],
        [0, 0, 0, 0, 0, 0, 0, a, 0],
        [a, 0, 0, 0, a, 0, b, 0, (1+a)/2]
    ], dtype=complex)

    return M / N


# ============================================================
#  Multi-seed scan: take max over seeds (since χ is a lower bound)
# ============================================================

def scan_multi_seed(a_vals, seeds, N=200, max_iter=15):
    print("\n" + "="*70)
    print("MULTI-SEED SCAN (taking χ_max per a)")
    print(f"Seeds: {seeds}")
    print(f"N = {N}, max_iter = {max_iter}")
    print("="*70)
    
    results = {}
    all_traces = {}  # store all seed results for diagnosis
    
    for a in a_vals:
        rho = horodecki_3x3(a)
        chi_list = []
        
        for seed in seeds:
            chi, _ = adaptive_polytope_bipartite(rho, 3, 3, N=N, max_iter=max_iter,
                                                 tol=1e-4, verbose=False, seed=seed)
            chi_list.append(chi)
        
        chi_max = max(chi_list)
        chi_min = min(chi_list)
        chi_mean = np.mean(chi_list)
        chi_std = np.std(chi_list)
        
        results[a] = {
            'max': chi_max,
            'min': chi_min,
            'mean': chi_mean,
            'std': chi_std,
            'all': chi_list
        }
        
        print(f"a={a:.3f}: max={chi_max:.5f}, min={chi_min:.5f}, mean={chi_mean:.5f}, std={chi_std:.5f}")
    
    return results


# ============================================================
#  Plot comparison: single seed vs multi-seed max
# ============================================================

def plot_comparison(single_seed_results, multi_seed_results, a_vals):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Single seed (original)
    ax1 = axes[0]
    ax1.plot(a_vals, single_seed_results, 'r-', linewidth=2, label='Single seed (seed=42)')
    ax1.set_xlabel('Parameter a', fontsize=12)
    ax1.set_ylabel('Visibility χ(a)', fontsize=12)
    ax1.set_title('Single Seed (Original)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right: Multi-seed max with error bars (min-max range)
    ax2 = axes[1]
    a_vals_list = list(multi_seed_results.keys())
    chi_max = [multi_seed_results[a]['max'] for a in a_vals_list]
    chi_min = [multi_seed_results[a]['min'] for a in a_vals_list]
    chi_mean = [multi_seed_results[a]['mean'] for a in a_vals_list]
    
    ax2.plot(a_vals_list, chi_max, 'b-', linewidth=2, label='χ_max (over seeds)')
    ax2.fill_between(a_vals_list, chi_min, chi_max, alpha=0.3, color='blue', label='Min-Max range')
    ax2.plot(a_vals_list, chi_mean, 'g--', linewidth=1.5, label='χ_mean')
    ax2.set_xlabel('Parameter a', fontsize=12)
    ax2.set_ylabel('Visibility χ(a)', fontsize=12)
    ax2.set_title('Multi-Seed (max over 10 seeds)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('multi_seed_comparison.png', dpi=150)
    plt.show()
    
    print("\n→ Comparison plot saved to 'multi_seed_comparison.png'")


# ============================================================
#  Diagnose problematic a values
# ============================================================

def diagnose_outliers(multi_seed_results, threshold_std=0.05):
    print("\n" + "="*70)
    print("DIAGNOSIS: High-variance a values (possible local optima)")
    print("="*70)
    
    outliers = []
    for a, stats in multi_seed_results.items():
        if stats['std'] > threshold_std:
            outliers.append((a, stats['std'], stats['min'], stats['max']))
            print(f"a={a:.3f}: std={stats['std']:.5f}, range=[{stats['min']:.5f}, {stats['max']:.5f}]")
    
    if not outliers:
        print("No high-variance points found (std ≤ {threshold_std})")
    else:
        print(f"\n→ {len(outliers)} points show seed-dependent variance > {threshold_std}")
        print("  These indicate initialization sensitivity and possible suboptimal solutions.")
    
    return outliers


# ============================================================
#  Main
# ============================================================

def main():
    print("=" * 70)
    print("ADAPTIVE POLYTOPE – MULTI-SEED DIAGNOSIS")
    print("Testing initialization sensitivity for Horodecki family")
    print("=" * 70)
    
    # Parameters
    a_vals = np.linspace(0.2, 1.0, 17)  # 17 points for smoother curve
    seeds = list(range(10))  # 10 different initial polytopes
    N = 200
    max_iter = 15
    
    # Run single seed baseline (seed=42, as before)
    print("\nRunning single seed baseline (seed=42)...")
    single_seed_results = []
    for a in a_vals:
        rho = horodecki_3x3(a)
        chi, _ = adaptive_polytope_bipartite(rho, 3, 3, N=N, max_iter=max_iter,
                                             tol=1e-4, verbose=False, seed=42)
        single_seed_results.append(chi)
        print(f"a={a:.3f}: χ={chi:.5f}")
    
    # Run multi-seed scan
    multi_seed_results = scan_multi_seed(a_vals, seeds, N=N, max_iter=max_iter)
    
    # Plot comparison
    plot_comparison(single_seed_results, multi_seed_results, a_vals)
    
    # Diagnose outliers
    outliers = diagnose_outliers(multi_seed_results, threshold_std=0.05)
    
    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
    If multi-seed max curve is smooth and single-seed had dips:
        → The original dips were due to bad initialization
        → Use multi-seed max for final figure
    
    If multi-seed max still has dips:
        → May need larger N (e.g., 300-500)
        → Consider MOSEK for better numerical stability
    
    If min-max range is large:
        → Algorithm is sensitive to initial polytope
        → Paper likely uses multiple runs or better initialization
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()