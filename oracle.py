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
from adaptive import adaptive_polytope_bipartite


class OracleLabel(Enum):
    """Two possible outputs from the oracle (no SEPARABLE)."""
    ENTANGLED = 0   # χ < τ: rigorously entangled
    UNKNOWN = 1     # χ ≥ τ: cannot certify (may be separable or entangled)


class AdaptivePolytopeOracle:
    """
    Oracle for bipartite separability using adaptive polytope algorithm.
    
    The oracle returns a lower bound χ on the true visibility.
    Due to the polytope approximation, χ is a LOWER BOUND, so:
        - Small χ (< τ) implies entanglement (rigorous)
        - Large χ (≥ τ) does NOT imply separability
    
    Parameters
    ----------
    N : int
        Number of polytope vertices (default: 200)
    max_iter : int
        Maximum adaptive iterations (default: 15)
    tol : float
        Convergence tolerance (default: 1e-4)
    ent_threshold : float
        χ < ent_threshold → ENTANGLED (default: 0.99)
    """
    
    def __init__(self, N=200, max_iter=15, tol=1e-4, ent_threshold=0.99):
        self.N = N
        self.max_iter = max_iter
        self.tol = tol
        self.ent_threshold = ent_threshold
    
    def visibility(self, rho, dA, dB, seed=None):
        """
        Compute visibility χ (lower bound) using adaptive polytope.
        
        Returns
        -------
        chi : float
            Lower bound on the separability visibility
        history : list
            Convergence history
        """
        chi, history = adaptive_polytope_bipartite(
            rho, dA, dB,
            N=self.N,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=False,
            seed=seed
        )
        return chi, history
    
    def query(self, rho, dA, dB, seed=None):
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
        """
        chi, history = self.visibility(rho, dA, dB, seed=seed)
        
        if chi < self.ent_threshold:
            label = OracleLabel.ENTANGLED
        else:
            label = OracleLabel.UNKNOWN
        
        return {
            'chi': chi,
            'label': label,
            'history': history,
            'iterations': len(history),
            'converged': len(history) < self.max_iter
        }
    
    def query_batch(self, states, dims_list, seeds=None):
        """
        Batch query multiple states.
        
        Parameters
        ----------
        states : list of (rho, dA, dB) or list of rho with shared dims
        dims_list : list of (dA, dB) or single tuple
        seeds : list of seeds or None
        
        Returns
        -------
        list of query results
        """
        results = []
        for i, item in enumerate(states):
            if isinstance(item, tuple) and len(item) == 3:
                rho, dA, dB = item
            else:
                rho = item
                if isinstance(dims_list, tuple) and len(dims_list) == 2:
                    dA, dB = dims_list
                else:
                    dA, dB = dims_list[i]
            
            seed = seeds[i] if seeds is not None else None
            results.append(self.query(rho, dA, dB, seed=seed))
        
        return results
    
    def certify_entangled(self, rho, dA, dB, seed=None):
        """
        Convenience method: returns True if state is certified entangled.
        """
        result = self.query(rho, dA, dB, seed=seed)
        return result['label'] == OracleLabel.ENTANGLED