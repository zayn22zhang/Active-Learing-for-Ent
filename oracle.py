"""
Reusable quantum entanglement oracle built from the Ohst-style hierarchy code.

This module is intentionally conservative:
- PPT negativity is reported as a rigorous entanglement certificate.
- A successful hierarchy decomposition is reported as certified separability/BSEP.
- Failure of an inner hierarchy is useful as a strong numerical oracle label, but is
  marked as numerical rather than a theorem-level entanglement certificate.

Main use cases
--------------
1. Three-qubit GME-style oracle:

    from oracle import EntanglementOracle
    from ohst_reproduction import ghz_state

    oracle = EntanglementOracle(vertices=120, epsilon=5e-4)
    result = oracle.classify_three_qubit(ghz_state())
    print(result.label, result.oracle_label, result.diagnostics)

2. Bipartite C^2 tensor C^4 oracle:

    result = oracle.classify_bipartite_2x4(rho)

Labels
------
- oracle_label = 1: entangled/GME according to the requested oracle mode.
- oracle_label = 0: not entangled / not GME certified by the hierarchy.
- oracle_label = -1: inconclusive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ohst_reproduction import (
    BipartiteSeparableHierarchy,
    BlochPolytope,
    BSEPHierarchy,
    WitnessExtractor,
    partial_transpose_8x8_np,
)


@dataclass
class OracleResult:
    label: str
    oracle_label: int
    is_entangled: Optional[bool]
    is_gme: Optional[bool]
    certificate: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "oracle_label": self.oracle_label,
            "is_entangled": self.is_entangled,
            "is_gme": self.is_gme,
            "certificate": self.certificate,
            "diagnostics": self.diagnostics,
        }


class EntanglementOracle:
    """Strong numerical oracle for small quantum entanglement tasks.

    Parameters
    ----------
    vertices:
        Number of Fibonacci Bloch-sphere samples. The actual polytope includes two
        extra pole states, so total vertices are ``vertices + 2``.
    epsilon:
        Frobenius feasibility tube used by the Ohst-style inner hierarchy.
    ppt_tol:
        Numerical tolerance for PPT eigenvalue negativity.
    witness_tol:
        Numerical tolerance for witness objective negativity.
    ppt_strengthen_2x4:
        Adds PPT constraints to residual 4x4 blocks in the bipartite C^2 x C^4
        hierarchy. Useful when the C^4 side is interpreted as two qubits.
    """

    def __init__(
        self,
        vertices: int = 120,
        epsilon: float = 5e-4,
        ppt_tol: float = 1e-8,
        witness_tol: float = 1e-8,
        ppt_strengthen_2x4: bool = True,
    ):
        self.vertices = vertices
        self.epsilon = epsilon
        self.ppt_tol = ppt_tol
        self.witness_tol = witness_tol
        self.ppt_strengthen_2x4 = ppt_strengthen_2x4

        self.polytope = BlochPolytope.fibonacci(vertices)
        self._bsep = None
        self._witness = None
        self._bipartite_2x4 = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, rho: np.ndarray, mode: str = "three_qubit") -> OracleResult:
        if mode in ("three_qubit", "gme", "bsep"):
            return self.classify_three_qubit(rho)
        if mode in ("bipartite_2x4", "2x4"):
            return self.classify_bipartite_2x4(rho)
        raise ValueError(f"Unknown oracle mode: {mode}")

    def classify_three_qubit(self, rho: np.ndarray) -> OracleResult:
        """Classify an 8x8 state as a three-qubit GME/BSEP oracle task.

        Interpretation:
        - If the BSEP hierarchy contains rho, rho is certified biseparable by an
          explicit finite-polytope decomposition. For a GME task this is label 0.
        - If hierarchy containment fails, the oracle returns label 1 as a strong
          numerical GME label. The result reports witness negativity when available.
        - PPT negativity across a single subsystem is included as an additional
          rigorous entanglement diagnostic, but by itself it is not a GME proof.
        """
        rho = self._validated_density_matrix(rho, dim=8)
        ppt_mins = self.ppt_min_eigenvalues_three_qubit(rho)
        npt_partitions = [k for k, v in ppt_mins.items() if v < -self.ppt_tol]

        bsep = self._get_bsep()
        contains_bsep = bsep.contains(rho)
        diagnostics: Dict[str, Any] = {
            "mode": "three_qubit_bsep_gme_oracle",
            "vertices": len(self.polytope.vertices),
            "epsilon": self.epsilon,
            "ppt_min_eigenvalues": ppt_mins,
            "npt_partitions": npt_partitions,
            "bsep_solver": bsep.last_diagnostics,
        }

        if contains_bsep:
            return OracleResult(
                label="biseparable_certified_by_ohst_inner_hierarchy",
                oracle_label=0,
                is_entangled=bool(npt_partitions) if npt_partitions else None,
                is_gme=False,
                certificate="finite_polytope_bsep_decomposition",
                diagnostics=diagnostics,
            )

        witness_data = self._get_witness().extract(rho)
        witness_obj = None
        certificate_strength = None
        if witness_data is not None:
            witness_obj = float(np.real(witness_data["obj"]))
            certificate_strength = -witness_obj
            diagnostics["witness_obj"] = witness_obj
            diagnostics["certificate_strength"] = certificate_strength
            diagnostics["witness_solver_status"] = witness_data.get("status")

        if witness_obj is not None and witness_obj < -self.witness_tol:
            label = "gme_entangled_by_ohst_numerical_oracle"
            certificate = "negative_finite_polytope_witness"
        else:
            label = "gme_entangled_by_bsep_infeasibility_oracle"
            certificate = "bsep_inner_hierarchy_infeasible"

        return OracleResult(
            label=label,
            oracle_label=1,
            is_entangled=True if npt_partitions else None,
            is_gme=True,
            certificate=certificate,
            diagnostics=diagnostics,
        )

    def classify_bipartite_2x4(self, rho: np.ndarray) -> OracleResult:
        """Classify an 8x8 C^2 tensor C^4 bipartite state.

        This is useful for additional validation and for ordinary bipartite tasks.
        PPT negativity is a rigorous entanglement certificate. A successful hierarchy
        decomposition is a rigorous finite-polytope separability certificate. PPT+not
        contained is reported as inconclusive because PPT entanglement requires a
        stronger/independent certificate.
        """
        rho = self._validated_density_matrix(rho, dim=8)
        min_pt = self.ppt_min_eigenvalue_bipartite(rho, dims=(2, 4), subsystem=0)
        diagnostics: Dict[str, Any] = {
            "mode": "bipartite_2x4_sep_oracle",
            "vertices": len(self.polytope.vertices),
            "epsilon": self.epsilon,
            "ppt_min_eigenvalue_A": min_pt,
            "ppt_strengthen_2x4": self.ppt_strengthen_2x4,
        }

        if min_pt < -self.ppt_tol:
            return OracleResult(
                label="entangled_certified_by_npt",
                oracle_label=1,
                is_entangled=True,
                is_gme=None,
                certificate="negative_partial_transpose",
                diagnostics=diagnostics,
            )

        sep = self._get_bipartite_2x4()
        contains_sep = sep.contains(rho)
        diagnostics["sep_solver"] = sep.last_diagnostics

        if contains_sep:
            return OracleResult(
                label="separable_certified_by_2x4_inner_hierarchy",
                oracle_label=0,
                is_entangled=False,
                is_gme=None,
                certificate="finite_polytope_sep_decomposition",
                diagnostics=diagnostics,
            )

        return OracleResult(
            label="inconclusive_ppt_not_certified_separable",
            oracle_label=-1,
            is_entangled=None,
            is_gme=None,
            certificate="ppt_but_not_in_current_inner_hierarchy",
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Diagnostics / utility methods
    # ------------------------------------------------------------------
    def ppt_min_eigenvalues_three_qubit(self, rho: np.ndarray) -> Dict[str, float]:
        rho = self._validated_density_matrix(rho, dim=8)
        out = {}
        for subsystem in ("A", "B", "C"):
            pt = partial_transpose_8x8_np(rho, subsystem)
            pt = (pt + pt.conj().T) / 2
            out[subsystem] = float(np.linalg.eigvalsh(pt).min())
        return out

    @staticmethod
    def ppt_min_eigenvalue_bipartite(
        rho: np.ndarray,
        dims=(2, 4),
        subsystem: int = 0,
    ) -> float:
        rho = np.asarray(rho, dtype=complex)
        d0, d1 = dims
        if rho.shape != (d0 * d1, d0 * d1):
            raise ValueError(f"Expected shape {(d0 * d1, d0 * d1)}, got {rho.shape}")
        tensor = rho.reshape((d0, d1, d0, d1))
        if subsystem == 0:
            pt = tensor.transpose((2, 1, 0, 3))
        elif subsystem == 1:
            pt = tensor.transpose((0, 3, 2, 1))
        else:
            raise ValueError("subsystem must be 0 or 1")
        pt = pt.reshape((d0 * d1, d0 * d1))
        pt = (pt + pt.conj().T) / 2
        return float(np.linalg.eigvalsh(pt).min())

    # ------------------------------------------------------------------
    # Lazy hierarchy construction
    # ------------------------------------------------------------------
    def _get_bsep(self) -> BSEPHierarchy:
        if self._bsep is None:
            self._bsep = BSEPHierarchy(self.polytope, epsilon=self.epsilon)
        return self._bsep

    def _get_witness(self) -> WitnessExtractor:
        if self._witness is None:
            self._witness = WitnessExtractor(self.polytope)
        return self._witness

    def _get_bipartite_2x4(self) -> BipartiteSeparableHierarchy:
        if self._bipartite_2x4 is None:
            self._bipartite_2x4 = BipartiteSeparableHierarchy(
                self.polytope,
                dB=4,
                epsilon=self.epsilon,
                ppt_strengthen_rest=self.ppt_strengthen_2x4,
            )
        return self._bipartite_2x4

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validated_density_matrix(
        rho: np.ndarray,
        dim: int,
        trace_tol: float = 1e-7,
        hermitian_tol: float = 1e-7,
        psd_tol: float = 1e-7,
    ) -> np.ndarray:
        rho = np.asarray(rho, dtype=complex)
        if rho.shape != (dim, dim):
            raise ValueError(f"Expected a {dim}x{dim} density matrix, got {rho.shape}")
        if not np.allclose(rho, rho.conj().T, atol=hermitian_tol):
            raise ValueError("Input is not Hermitian within tolerance.")
        tr = np.trace(rho)
        if abs(tr - 1.0) > trace_tol:
            raise ValueError(f"Trace must be 1; got {tr}")
        eig_min = float(np.linalg.eigvalsh((rho + rho.conj().T) / 2).min())
        if eig_min < -psd_tol:
            raise ValueError(f"Input is not positive semidefinite; min eigenvalue={eig_min}")
        return (rho + rho.conj().T) / 2


if __name__ == "__main__":
    from ohst_reproduction import ghz_state, target_state

    oracle = EntanglementOracle(vertices=30, epsilon=1e-3)
    rho = target_state(0.5, ghz_state)
    result = oracle.classify_three_qubit(rho)
    print(result.as_dict())
