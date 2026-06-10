"""
Oracle interface for separability certification.
Wraps the adaptive polytope algorithm with two-output classification.

Mathematical property:
    χ(ρ) is a LOWER BOUND on the true separability visibility.
    Therefore:
        - χ < τ  → ρ is ENTANGLED (rigorous)
        - χ ≥ τ  → UNKNOWN (cannot certify separability)
"""

from enum import Enum
from typing import List, Optional, Tuple
from adaptive import adaptive_polytope_bipartite


class OracleLabel(Enum):
    """Two possible outputs from the oracle (no SEPARABLE)."""
    ENTANGLED = 0   # χ < τ: rigorously entangled
    UNKNOWN = 1     # χ ≥ τ: cannot certify (may be separable or entangled)


class AdaptivePolytopeOracle:
    """
    Oracle for bipartite separability using adaptive polytope algorithm.
    
    The oracle returns a lower bound χ on the true visibility.
    For ML training, prefer using the continuous χ value over discrete labels.
    """
    
    def __init__(self, N=200, max_iter=15, tol=1e-4, ent_threshold=0.99):
        self.N = N
        self.max_iter = max_iter
        self.tol = tol
        self.ent_threshold = ent_threshold
    
    def visibility(self, rho, dA, dB, seed=None) -> Tuple[float, List[float], bool]:
        """
        Compute visibility χ (lower bound) using adaptive polytope.
        
        Returns
        -------
        chi : float
            Lower bound on the separability visibility
        history : list
            Convergence history (χ₂ at each iteration)
        converged : bool
            Whether algorithm converged before max_iter
        """
        # FIXED: receive 3 return values from adaptive_polytope_bipartite
        chi, history, converged = adaptive_polytope_bipartite(
            rho, dA, dB,
            N=self.N,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=False,
            seed=seed
        )
        return chi, history, converged
    
    def query(self, rho, dA, dB, seed=None) -> dict:
        """
        Query the oracle for entanglement certification.
        
        Returns
        -------
        dict
            - 'chi': float, computed visibility lower bound
            - 'label': OracleLabel, ENTANGLED or UNKNOWN
            - 'history': list, convergence history
            - 'iterations': int, number of iterations taken
            - 'converged': bool, whether algorithm converged
            - 'N': int, number of vertices used
            - 'max_iter': int, maximum iterations
            - 'ent_threshold': float, threshold used
            - 'seed': int or None, random seed used
        """
        chi, history, converged = self.visibility(rho, dA, dB, seed=seed)
        
        if chi < self.ent_threshold:
            label = OracleLabel.ENTANGLED
        else:
            label = OracleLabel.UNKNOWN
        
        return {
            'chi': chi,
            'label': label,
            'history': history,
            'iterations': len(history),
            'converged': converged,
            'N': self.N,
            'max_iter': self.max_iter,
            'ent_threshold': self.ent_threshold,
            'seed': seed
        }
    
    def get_chi(self, rho, dA, dB, seed=None) -> float:
        """Convenience method: returns only χ (continuous value)."""
        chi, _, _ = self.visibility(rho, dA, dB, seed=seed)
        return chi