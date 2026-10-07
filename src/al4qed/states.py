"""
Quantum states for benchmarking.
"""

import numpy as np


def horodecki_3x3(a):
    """
    Horodecki 3x3 PPT-entangled state (bound entangled).
    Appendix E, Eq. (48) of Ohst et al.
    """
    if not 0 <= a <= 1:
        raise ValueError('a must lie in [0,1]')
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


def werner_state(p, d=2):
    """Werner state: p|Ψ⁻⟩⟨Ψ⁻| + (1-p)I/d² (d=2 only)."""
    if not 0 <= p <= 1:
        raise ValueError('p must lie in [0,1]')
    if d != 2:
        raise ValueError("Werner state only implemented for d=2")
    psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    proj = np.outer(psi_minus, psi_minus.conj())
    I4 = np.eye(4, dtype=complex) / 4
    return p * proj + (1-p) * I4


def isotropic_state(p, d=2):
    """White-noise maximally entangled state in d x d, p in [0,1]."""
    if not 0 <= p <= 1 or int(d) != d or d < 2:
        raise ValueError('Require p in [0,1] and integer d>=2')
    psi = np.eye(d).reshape(-1) / np.sqrt(d)
    return p*np.outer(psi, psi.conj()) + (1-p)*np.eye(d*d)/(d*d)
