#!/usr/bin/env python3
"""
Benchmark script for adaptive polytope oracle.
Verifies Horodecki state properties and generates Figure 3(a) red curve.
Uses χ_mean with error bars for honest representation.
Includes threshold calibration study.
"""

import numpy as np
import matplotlib.pyplot as plt

from adaptive import swap_systems, partial_transpose
from states import horodecki_3x3, werner_state, isotropic_state
from oracle import AdaptivePolytopeOracle, OracleLabel


def verify_horodecki(a=0.3):
    """Print Horodecki state properties for verification."""
    print("\n" + "=" * 60)
    print("Horodecki State Verification (a=0.3)")
    print("=" * 60)
    
    rho = horodecki_3x3(a)
    
    print(f"Trace = {np.trace(rho):.12f}")
    eigs = np.linalg.eigvalsh(rho)
    print(f"Min eigenvalue = {np.min(eigs):.12f}")
    
    pt = partial_transpose(rho, (3, 3), 1)
    eigs_pt = np.linalg.eigvalsh(pt)
    print(f"Min PPT eigenvalue = {np.min(eigs_pt):.12f}")
    
    swap_diff = np.linalg.norm(rho - swap_systems(swap_systems(rho, 3, 3), 3, 3))
    print(f"Swap consistency = {swap_diff:.2e}")
    
    rank = np.linalg.matrix_rank(rho, tol=1e-8)
    print(f"Rank = {rank}")


def generate_figure3a_red_curve(a_vals=None, N=200, max_iter=15, n_seeds=10):
    """
    Generate Figure 3(a) red curve using MEAN over multiple seeds.
    Error bars represent ±1 standard deviation.
    """
    if a_vals is None:
        a_vals = np.linspace(0.2, 1.0, 41)
    
    oracle = AdaptivePolytopeOracle(N=N, max_iter=max_iter, ent_threshold=0.99)
    
    chi_mean_vals = []
    chi_std_vals = []
    chi_all = []  # Store all χ values for histogram analysis
    
    print(f"\nScanning a ∈ [0.2, 1.0] with {len(a_vals)} points, {n_seeds} seeds each...")
    print("(Using χ_mean, not χ_max, for honest representation)")
    
    for a in a_vals:
        rho = horodecki_3x3(a)
        chi_list = []
        
        for seed in range(n_seeds):
            chi = oracle.get_chi(rho, 3, 3, seed=seed)  # Returns continuous χ
            chi_list.append(chi)
            chi_all.append(chi)
        
        chi_mean_vals.append(np.mean(chi_list))
        chi_std_vals.append(np.std(chi_list))
        
        if len(a_vals) <= 20:
            print(f"a={a:.3f}: mean={chi_mean_vals[-1]:.5f}, std={chi_std_vals[-1]:.5f}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: χ(a) with error bars
    ax1.plot(a_vals, chi_mean_vals, 'r-', linewidth=2, label=f'χ_mean (N={N}, {n_seeds} seeds)')
    ax1.fill_between(a_vals, 
                     np.array(chi_mean_vals) - np.array(chi_std_vals),
                     np.array(chi_mean_vals) + np.array(chi_std_vals),
                     alpha=0.2, color='red', label='±1 std')
    ax1.set_xlabel('Parameter a', fontsize=12)
    ax1.set_ylabel('Visibility χ(a)', fontsize=12)
    ax1.set_title('Horodecki 3×3 Family – Figure 3(a) Red Curve', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right: Distribution of χ values (histogram)
    ax2.hist(chi_all, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=oracle.ent_threshold, color='red', linestyle='--', 
                label=f'τ_ent = {oracle.ent_threshold}')
    ax2.set_xlabel('χ value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of χ over all (a, seed)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure_3a_red_curve.png', dpi=150)
    plt.show()
    
    return a_vals, chi_mean_vals, chi_std_vals


def calibrate_threshold():
    """
    Study the effect of ent_threshold on classification.
    Helps choose a reasonable τ value.
    """
    print("\n" + "=" * 60)
    print("Threshold Calibration Study")
    print("=" * 60)
    
    a_vals = [0.3, 0.5, 0.7, 0.9, 1.0]
    thresholds = [0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
    
    print("\nFraction of Horodecki states certified as ENTANGLED for each threshold:")
    print("a\\τ\t" + "\t".join([f"{t:.3f}" for t in thresholds]))
    
    for a in a_vals:
        rho = horodecki_3x3(a)
        row = [f"{a:.1f}"]
        for tau in thresholds:
            oracle = AdaptivePolytopeOracle(ent_threshold=tau)
            oracle.N = 200
            oracle.max_iter = 15
            
            ent_count = 0
            for seed in range(10):
                chi = oracle.get_chi(rho, 3, 3, seed=seed)
                if chi < tau:
                    ent_count += 1
            fraction = ent_count / 10
            row.append(f"{fraction:.2f}")
        print("\t".join(row))
    
    print("\nRecommendation: τ = 0.99 gives conservative ENTANGLED certification")
    print("(Only states with very strong evidence are labeled entangled)")


def benchmark_werner_isotropic():
    """
    Benchmark on Werner and Isotropic states (known exact thresholds).
    These states are separable for p ≤ 1/2, entangled for p > 1/2.
    For p > 1/2, χ_true = 1/(2p) ??? Actually known: χ = p for p≤0.5? 
    Here we just study χ behavior.
    """
    print("\n" + "=" * 60)
    print("Werner / Isotropic Benchmark")
    print("=" * 60)
    print("(χ is a lower bound, not the exact threshold)")
    
    oracle = AdaptivePolytopeOracle(N=200, max_iter=15)
    
    for name, state_fn in [("Werner", werner_state), ("Isotropic", isotropic_state)]:
        print(f"\n{name} state (2x2):")
        for p in [0.3, 0.5, 0.7, 0.9]:
            rho = state_fn(p)
            chi = oracle.get_chi(rho, 2, 2, seed=42)
            print(f"  p={p:.1f}: χ={chi:.5f}")


if __name__ == "__main__":
    verify_horodecki()
    generate_figure3a_red_curve()
    calibrate_threshold()
    benchmark_werner_isotropic()