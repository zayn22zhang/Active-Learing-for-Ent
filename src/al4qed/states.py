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


def ghz_w_3x3(p_g, p_w):
    """Two-qutrit G/W/white-noise mixture, not a three-party GHZ/W state.

    G=(|00>+|11>+|22>)/sqrt(3), W=(|01>+|10>)/sqrt(2).
    rho=p_g*|G><G|+p_w*|W><W|+(1-p_g-p_w)*I_9/9.
    """
    weights = np.asarray([p_g, p_w], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() > 1:
        raise ValueError('Require finite p_g,p_w >= 0 and p_g+p_w <= 1')
    g = np.eye(3).reshape(-1) / np.sqrt(3)
    w = np.zeros(9); w[[1, 3]] = 1 / np.sqrt(2)
    return p_g*np.outer(g,g) + p_w*np.outer(w,w) + (1-p_g-p_w)*np.eye(9)/9
