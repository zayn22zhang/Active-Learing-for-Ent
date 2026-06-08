"""
Quantum states for benchmarking.
"""

import numpy as np


def horodecki_3x3(a):
    """
    Horodecki 3x3 PPT-entangled state (bound entangled).
    Appendix E, Eq. (48) of Ohst et al.
    """
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
    if d != 2:
        raise ValueError("Werner state only implemented for d=2")
    psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    proj = np.outer(psi_minus, psi_minus.conj())
    I4 = np.eye(4, dtype=complex) / 4
    return p * proj + (1-p) * I4


def isotropic_state(p, d=2):
    """Isotropic state: p|Φ⁺⟩⟨Φ⁺| + (1-p)I/d² (d=2 only)."""
    if d != 2:
        raise ValueError("Isotropic state only implemented for d=2")
    phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    proj = np.outer(phi_plus, phi_plus.conj())
    I4 = np.eye(4, dtype=complex) / 4
    return p * proj + (1-p) * I4