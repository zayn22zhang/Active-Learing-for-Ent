#!/usr/bin/env python3
"""
Convergence analysis of adaptive polytope algorithm.
Generates two key figures:
1. χ vs iteration for a fixed state (shows adaptive refinement)
2. Final χ distribution over multiple seeds (shows stability)
"""

import numpy as np
import matplotlib.pyplot as plt
from states import horodecki_3x3
from adaptive import adaptive_polytope_bipartite


def plot_convergence_history(a=0.3, N=200, max_iter=15, seed=42):
    """
    Figure 1: χ vs iteration for a fixed state.
    This shows how adaptive refinement improves the polytope.
    """
    print("\n" + "=" * 60)
    print(f"Convergence History: Horodecki a={a}, N={N}, seed={seed}")
    print("=" * 60)
    
    rho = horodecki_3x3(a)
    chi_final, history, converged = adaptive_polytope_bipartite(
        rho, 3, 3, N=N, max_iter=max_iter, tol=1e-4, verbose=True, seed=seed
    )
    
    print(f"\nFinal χ = {chi_final:.5f}")
    print(f"Converged: {converged}")
    print(f"History: {[f'{h:.4f}' for h in history]}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    iterations = range(1, len(history) + 1)
    plt.plot(iterations, history, 'bo-', linewidth=2, markersize=8, label='χ₂ (after swap)')
    plt.axhline(y=chi_final, color='r', linestyle='--', alpha=0.7, label=f'Final χ = {chi_final:.4f}')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Visibility χ', fontsize=12)
    plt.title(f'Adaptive Polytope Convergence\nHorodecki a={a}, N={N}, seed={seed}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('convergence_history.png', dpi=150)
    plt.show()
    
    return history, chi_final, converged


def plot_multi_seed_stability(a=0.3, N=200, max_iter=15, n_seeds=50):
    """
    Figure 2: Distribution of final χ over multiple random seeds.
    This proves the algorithm converges to a stable value regardless of initialization.
    """
    print("\n" + "=" * 60)
    print(f"Multi-Seed Stability: Horodecki a={a}, N={N}, {n_seeds} seeds")
    print("=" * 60)
    
    rho = horodecki_3x3(a)
    chi_values = []
    histories = []
    
    for seed in range(n_seeds):
        chi, history, converged = adaptive_polytope_bipartite(
            rho, 3, 3, N=N, max_iter=max_iter, tol=1e-4, verbose=False, seed=seed
        )
        chi_values.append(chi)
        histories.append(history)
        
        if seed % 10 == 0:
            print(f"  seed={seed:2d}: χ={chi:.5f}, iterations={len(history)}")
    
    chi_mean = np.mean(chi_values)
    chi_std = np.std(chi_values)
    chi_min = np.min(chi_values)
    chi_max = np.max(chi_values)
    
    print(f"\nStatistics over {n_seeds} seeds:")
    print(f"  mean = {chi_mean:.5f}")
    print(f"  std  = {chi_std:.5f}")
    print(f"  min  = {chi_min:.5f}")
    print(f"  max  = {chi_max:.5f}")
    print(f"  range = {chi_max - chi_min:.5f}")
    
    # Plot histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Histogram of final χ
    ax1.hist(chi_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=chi_mean, color='r', linestyle='--', linewidth=2, label=f'mean = {chi_mean:.4f}')
    ax1.axvline(x=chi_mean + chi_std, color='gray', linestyle=':', linewidth=1.5, label=f'±1σ')
    ax1.axvline(x=chi_mean - chi_std, color='gray', linestyle=':', linewidth=1.5)
    ax1.set_xlabel('Final χ', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Distribution over {n_seeds} random seeds\nHorodecki a={a}, N={N}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Convergence trajectories (first 10 seeds)
    for i in range(min(10, n_seeds)):
        ax2.plot(range(1, len(histories[i]) + 1), histories[i], 'o-', alpha=0.5, markersize=4)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Visibility χ', fontsize=12)
    ax2.set_title('Convergence trajectories (first 10 seeds)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_seed_stability.png', dpi=150)
    plt.show()
    
    return chi_values, histories


def compare_different_a_values(a_vals=[0.3, 0.5, 0.7, 0.9], N=200, n_seeds=30):
    """
    Compare stability across different a values.
    """
    print("\n" + "=" * 60)
    print(f"Stability across different a values (N={N}, {n_seeds} seeds each)")
    print("=" * 60)
    print("  a     mean χ     std χ     min χ     max χ")
    print("  -     ------     -----     -----     -----")
    
    results = {}
    for a in a_vals:
        rho = horodecki_3x3(a)
        chi_list = []
        for seed in range(n_seeds):
            chi, _, _ = adaptive_polytope_bipartite(
                rho, 3, 3, N=N, max_iter=15, tol=1e-4, verbose=False, seed=seed
            )
            chi_list.append(chi)
        
        results[a] = {
            'mean': np.mean(chi_list),
            'std': np.std(chi_list),
            'min': np.min(chi_list),
            'max': np.max(chi_list)
        }
        print(f"  {a:.1f}   {results[a]['mean']:.5f}   {results[a]['std']:.5f}   "
              f"{results[a]['min']:.5f}   {results[a]['max']:.5f}")
    
    return results


if __name__ == "__main__":
    # Figure 1: Convergence history for a single seed
    history, chi_final, converged = plot_convergence_history(a=0.3, N=200, max_iter=15, seed=42)
    
    # Figure 2: Multi-seed stability
    chi_values, histories = plot_multi_seed_stability(a=0.3, N=200, max_iter=15, n_seeds=50)
    
    # Figure 3: Compare across different a values
    results = compare_different_a_values(a_vals=[0.3, 0.5, 0.7, 0.9], N=200, n_seeds=30)