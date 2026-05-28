# ============================================
Reproduction of:
Ohst et al. (2024)

Features:
- BSEP hierarchy
- FBSEP hierarchy
- Dual witness extraction
- Adaptive refinement
- GHZ convergence benchmarks
- Horodecki validation benchmark

Status:
Frozen reproducibility version (v1.0)
# ============================================

import argparse
import csv
import time
import warnings

import cvxpy as cp
import numpy as np
from scipy.linalg import eigh
from scipy.stats import unitary_group

warnings.filterwarnings("ignore")


# ============================================
# 0. Solver configuration
# ============================================
if "MOSEK" not in cp.installed_solvers():
    raise RuntimeError("MOSEK is required. Please install MOSEK and cvxpy[mosek].")

SOLVER = cp.MOSEK

MOSEK_PARAMS = {
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-9,
    "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-9,
    "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-9,
    "MSK_IPAR_INTPNT_MAX_ITERATIONS": 800,
    "MSK_IPAR_INTPNT_STARTING_POINT": 1,
    "MSK_IPAR_LOG": 0,
}


# ============================================
# 1. Partial transpose permutation matrices
# ============================================
def build_pt_16_permutation():
    P = np.zeros((16, 16), dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    row_in = 2 * i + j
                    col_in = 2 * k + l
                    row_out = 2 * i + l
                    col_out = 2 * k + j
                    P[row_out * 4 + col_out, row_in * 4 + col_in] = 1.0
    return P


def build_pt_64_permutation(subsys):
    P = np.zeros((64, 64), dtype=float)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for ap in range(2):
                    for bp in range(2):
                        for cp in range(2):
                            row_in = (a << 2) | (b << 1) | c
                            col_in = (ap << 2) | (bp << 1) | cp
                            if subsys == "A":
                                row_out = (ap << 2) | (b << 1) | c
                                col_out = (a << 2) | (bp << 1) | cp
                            elif subsys == "B":
                                row_out = (a << 2) | (bp << 1) | c
                                col_out = (ap << 2) | (b << 1) | cp
                            else:
                                row_out = (a << 2) | (b << 1) | cp
                                col_out = (ap << 2) | (bp << 1) | c
                            P[row_out * 8 + col_out, row_in * 8 + col_in] = 1.0
    return P


PT_16 = build_pt_16_permutation()
PT_64_A = build_pt_64_permutation("A")
PT_64_B = build_pt_64_permutation("B")
PT_64_C = build_pt_64_permutation("C")


def partial_transpose_4x4(X):
    vec_x = cp.reshape(X, (16, 1), order="C")
    vec_pt = PT_16 @ vec_x
    pt = cp.reshape(vec_pt, (4, 4), order="C")
    return (pt + pt.H) / 2


def partial_transpose_8x8_np(Y, subsys):
    if subsys == "A":
        perm = PT_64_A
    elif subsys == "B":
        perm = PT_64_B
    else:
        perm = PT_64_C
    vec = Y.reshape(64, order="C")
    vec_pt = perm @ vec
    return vec_pt.reshape((8, 8), order="C")


# ============================================
# 2. Three-qubit subsystem permutations
# ============================================
def perm_matrix(perm):
    P = np.zeros((8, 8), dtype=float)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                idx_orig = (a << 2) | (b << 1) | c
                arr = [a, b, c]
                new_arr = [arr[perm[0]], arr[perm[1]], arr[perm[2]]]
                idx_new = (new_arr[0] << 2) | (new_arr[1] << 1) | new_arr[2]
                P[idx_new, idx_orig] = 1.0
    return P


P_BAC = perm_matrix((1, 0, 2))
P_CAB = perm_matrix((2, 0, 1))


# ============================================
# 3. Quantum states
# ============================================
def ghz_state():
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[-1] = 1 / np.sqrt(2)
    return np.outer(psi, psi.conj())


def w_state():
    psi = np.zeros(8, dtype=complex)
    psi[1] = psi[2] = psi[4] = 1 / np.sqrt(3)
    return np.outer(psi, psi.conj())


def phase_ghz_state(phi):
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[-1] = np.exp(1j * phi) / np.sqrt(2)
    return np.outer(psi, psi.conj())


def random_haar_state():
    v = unitary_group.rvs(8)[:, 0]
    return np.outer(v, v.conj())


def horodecki_2x4_state(a=0.25):
    """Horodecki PPT entangled state on C^2 tensor C^4."""
    norm = 8 * a + 1
    rho = np.zeros((8, 8), dtype=complex)
    for i in range(6):
        rho[i, i] = a / norm
    rho[6, 6] = (1 + a) / (2 * norm)
    rho[7, 7] = (1 + a) / (2 * norm)
    rho[0, 5] = a / norm
    rho[5, 0] = a / norm
    return rho / np.trace(rho)


def target_state(p, pure_fn, dim=8):
    return p * pure_fn() + (1 - p) * np.eye(dim, dtype=complex) / dim


# ============================================
# 4. Polytope geometry
# ============================================
class BlochPolytope:
    def __init__(self, vertices):
        self.vertices = list(vertices)

    @staticmethod
    def fibonacci(N=50):
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        verts = []
        phi = (1 + np.sqrt(5)) / 2
        for k in range(N):
            z = 1 - 2 * (k + 0.5) / N
            r = np.sqrt(max(0, 1 - z * z))
            theta = 2 * np.pi * k / phi
            x, y = r * np.cos(theta), r * np.sin(theta)
            rho = (np.eye(2) + x * sx + y * sy + z * sz) / 2
            verts.append(rho)
        verts.append((np.eye(2) + sz) / 2)
        verts.append((np.eye(2) - sz) / 2)
        return BlochPolytope(verts)

    def add_vertices(self, new_verts):
        existing = {self._bloch_hash(v) for v in self.vertices}
        for v in new_verts:
            key = self._bloch_hash(v)
            if key not in existing:
                self.vertices.append(v)
                existing.add(key)

    def _bloch_hash(self, rho):
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        b = [
            np.trace(rho @ sx).real,
            np.trace(rho @ sy).real,
            np.trace(rho @ sz).real,
        ]
        return tuple(np.round(b, decimals=6))

    def prune(self, max_vertices=300):
        if len(self.vertices) <= max_vertices:
            return

        kept = []
        for v in self.vertices:
            duplicate = False
            bv = np.array(self._bloch_hash(v))
            for u in kept:
                bu = np.array(self._bloch_hash(u))
                if np.linalg.norm(bv - bu) < 0.05:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(v)

        self.vertices = kept[:max_vertices]


# ============================================
# 5. CVXPY helpers
# ============================================
def kron_embedding(sigma, T):
    a, b = sigma[0, 0], sigma[0, 1]
    c, d = sigma[1, 0], sigma[1, 1]
    return cp.bmat([[a * T, b * T], [c * T, d * T]])


def kron_I_right(d_right, sigma):
    a, b = sigma[0, 0], sigma[0, 1]
    c, d = sigma[1, 0], sigma[1, 1]
    eye = np.eye(d_right, dtype=complex)
    return cp.bmat([[a * eye, b * eye], [c * eye, d * eye]])


def balanced_sum(terms):
    if not terms:
        return 0
    if len(terms) == 1:
        return terms[0]
    mid = len(terms) // 2
    return balanced_sum(terms[:mid]) + balanced_sum(terms[mid:])


def solve_problem(problem):
    problem.solve(
        solver=SOLVER,
        warm_start=True,
        mosek_params=MOSEK_PARAMS,
        verbose=False,
    )
    return problem.status


def is_optimal(status):
    return status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


# ============================================
# 6. Multipartite BSEP hierarchy, inner approximation
# ============================================
class BSEPHierarchy:
    def __init__(self, polytope, epsilon=1e-3):
        self.polytope = polytope
        self.epsilon = epsilon
        self.rho_param = cp.Parameter((8, 8), complex=True, hermitian=True)
        self.last_diagnostics = {}
        self._build_problem()

    def _build_problem(self):
        nv = len(self.polytope.vertices)
        T_ABC = [cp.Variable((4, 4), hermitian=True) for _ in range(nv)]
        T_BAC = [cp.Variable((4, 4), hermitian=True) for _ in range(nv)]
        T_CAB = [cp.Variable((4, 4), hermitian=True) for _ in range(nv)]

        w = cp.Variable(nonneg=True)
        constraints = [3 * w == 1]

        rho_abc = balanced_sum(
            [kron_embedding(sig, T) for sig, T in zip(self.polytope.vertices, T_ABC)]
        )
        constraints.append(sum(cp.trace(T) for T in T_ABC) == w)

        rho_bac = balanced_sum(
            [
                P_BAC @ kron_embedding(sig, T) @ P_BAC.T
                for sig, T in zip(self.polytope.vertices, T_BAC)
            ]
        )
        constraints.append(sum(cp.trace(T) for T in T_BAC) == w)

        rho_cab = balanced_sum(
            [
                P_CAB @ kron_embedding(sig, T) @ P_CAB.T
                for sig, T in zip(self.polytope.vertices, T_CAB)
            ]
        )
        constraints.append(sum(cp.trace(T) for T in T_CAB) == w)

        rho_approx = rho_abc + rho_bac + rho_cab

        for Tlist in [T_ABC, T_BAC, T_CAB]:
            for T in Tlist:
                constraints.append(T >> 0)
                constraints.append(partial_transpose_4x4(T) >> 0)

        residual = rho_approx - self.rho_param
        real_part = cp.reshape(cp.real(residual), (64, 1), order="C")
        imag_part = cp.reshape(cp.imag(residual), (64, 1), order="C")
        constraints.append(cp.sum_squares(real_part) + cp.sum_squares(imag_part) <= self.epsilon**2)

        self.problem = cp.Problem(cp.Minimize(0), constraints)

    def contains(self, rho):
        self.rho_param.value = rho
        try:
            status = solve_problem(self.problem)
            self.last_diagnostics = {
                "status": status,
                "solve_time": getattr(self.problem.solver_stats, "solve_time", None),
                "vertices": len(self.polytope.vertices),
                "ppt_strengthen_rest": getattr(self, "ppt_strengthen_rest", False),
            }
            return is_optimal(status)
        except Exception as exc:
            self.last_diagnostics = {
                "status": "exception",
                "error": str(exc),
                "vertices": len(self.polytope.vertices),
            }
            return False

    def threshold(self, pure_fn, max_iter=40, tol=1e-7):
        lo, hi = 0.0, 1.0
        last_infeasible = None
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            rho = target_state(mid, pure_fn)
            if self.contains(rho):
                lo = mid
            else:
                hi = mid
                last_infeasible = rho
            if hi - lo < tol:
                break
        return lo, last_infeasible


# ============================================
# 7. Exact FBSEP hierarchy: SEP(A|BC) cap SEP(B|AC) cap SEP(C|AB)
# ============================================
class FBSEPHierarchy:
    def __init__(self, polytope):
        self.polytope = polytope
        self.rho_param = cp.Parameter((8, 8), complex=True, hermitian=True)
        self.last_diagnostics = {}
        self._build_problem()

    def _build_problem(self):
        nv = len(self.polytope.vertices)
        T_ABC = [cp.Variable((4, 4), hermitian=True) for _ in range(nv)]
        T_BAC = [cp.Variable((4, 4), hermitian=True) for _ in range(nv)]
        T_CAB = [cp.Variable((4, 4), hermitian=True) for _ in range(nv)]
        rho_shared = cp.Variable((8, 8), hermitian=True)

        rho_abc = balanced_sum(
            [kron_embedding(sig, T) for sig, T in zip(self.polytope.vertices, T_ABC)]
        )
        rho_bac = balanced_sum(
            [
                P_BAC @ kron_embedding(sig, T) @ P_BAC.T
                for sig, T in zip(self.polytope.vertices, T_BAC)
            ]
        )
        rho_cab = balanced_sum(
            [
                P_CAB @ kron_embedding(sig, T) @ P_CAB.T
                for sig, T in zip(self.polytope.vertices, T_CAB)
            ]
        )

        constraints = [
            rho_abc == rho_shared,
            rho_bac == rho_shared,
            rho_cab == rho_shared,
            rho_shared == self.rho_param,
            sum(cp.trace(T) for T in T_ABC) == 1,
            sum(cp.trace(T) for T in T_BAC) == 1,
            sum(cp.trace(T) for T in T_CAB) == 1,
        ]

        for Tlist in [T_ABC, T_BAC, T_CAB]:
            for T in Tlist:
                constraints.append(T >> 0)
                constraints.append(partial_transpose_4x4(T) >> 0)

        self.problem = cp.Problem(cp.Minimize(0), constraints)

    def contains(self, rho):
        self.rho_param.value = rho
        try:
            status = solve_problem(self.problem)
            self.last_diagnostics = {
                "status": status,
                "solve_time": getattr(self.problem.solver_stats, "solve_time", None),
                "vertices": len(self.polytope.vertices),
            }
            return is_optimal(status)
        except Exception as exc:
            self.last_diagnostics = {
                "status": "exception",
                "error": str(exc),
                "vertices": len(self.polytope.vertices),
            }
            return False


# ============================================
# 8. Multipartite dual witness extraction
# ============================================
class WitnessExtractor:
    def __init__(self, polytope):
        assert len(polytope.vertices) > 0
        self.polytope = polytope
        self.n_vertices = len(polytope.vertices)
        self.rho_param = cp.Parameter((8, 8), complex=True, hermitian=True)
        self._build_problem()

    def _build_problem(self):
        self.P = cp.Variable((8, 8), hermitian=True)
        self.Q = cp.Variable((8, 8), hermitian=True)
        constraints = [self.P >> 0, self.Q >> 0]

        def pt_on_rest_blocks(Y):
            blocks = [[None for _ in range(2)] for __ in range(2)]
            for i in range(2):
                for j in range(2):
                    block = Y[4 * i : 4 * i + 4, 4 * j : 4 * j + 4]
                    blocks[i][j] = partial_transpose_4x4(block)
            return cp.vstack([cp.hstack(blocks[i]) for i in range(2)])

        self.W = self.P + pt_on_rest_blocks(self.Q)
        constraints.append(cp.trace(self.W) == 1)

        for sigma in self.polytope.vertices:
            constraints.append(cp.real(cp.trace(kron_I_right(4, sigma) @ self.W)) >= 0)

        for Pmat in [P_BAC, P_CAB]:
            W_perm = Pmat.T @ self.W @ Pmat
            for sigma in self.polytope.vertices:
                constraints.append(cp.real(cp.trace(kron_I_right(4, sigma) @ W_perm)) >= 0)

        objective = cp.Minimize(cp.real(cp.trace(self.W @ self.rho_param)))
        self.problem = cp.Problem(objective, constraints)

    def extract(self, rho):
        if len(self.polytope.vertices) != self.n_vertices:
            raise RuntimeError("WitnessExtractor polytope changed after build; rebuild before extract().")
        self.rho_param.value = rho
        try:
            status = solve_problem(self.problem)
            if is_optimal(status):
                return {
                    "W": self.W.value,
                    "obj": self.problem.value,
                    "status": status,
                    "solve_time": getattr(self.problem.solver_stats, "solve_time", None),
                }
        except Exception:
            pass
        return None

    @staticmethod
    def exposed_vertex(W):
        H = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                H[i, j] = np.trace(W[4 * i : 4 * i + 4, 4 * j : 4 * j + 4])
        H = (H + H.conj().T) / 2
        _, v = eigh(H)
        psi = v[:, 0]
        return np.outer(psi, psi.conj())


# ============================================
# 9. Multipartite adaptive refinement
# ============================================
class AdaptiveBSEP:
    def __init__(self, polytope, epsilon=1e-3, max_vertices=300):
        self.polytope = polytope
        self.epsilon = epsilon
        self.max_vertices = max_vertices
        self.hierarchy = BSEPHierarchy(polytope, epsilon)
        self.witness = WitnessExtractor(polytope)
        self.history = []

    def refine(self, pure_fn, max_rounds=5):
        thresholds = []
        print("\nround | vertices | threshold | delta     | witness_obj | cert_strength | solve_s")
        print("-" * 84)
        for rnd in range(max_rounds):
            start = time.time()
            thr, last_infeasible = self.hierarchy.threshold(pure_fn)
            elapsed = time.time() - start
            delta = np.nan if not thresholds else abs(thr - thresholds[-1])
            thresholds.append(thr)

            witness_obj = np.nan
            if last_infeasible is not None:
                wdata = self.witness.extract(last_infeasible)
                if wdata is not None:
                    witness_obj = wdata["obj"]
            else:
                wdata = None

            certificate_strength = -witness_obj if np.isfinite(witness_obj) else np.nan
            row = {
                "round": rnd + 1,
                "vertices": len(self.polytope.vertices),
                "threshold": thr,
                "delta": delta,
                "witness_obj": witness_obj,
                "certificate_strength": certificate_strength,
                "solve_time": elapsed,
            }
            self.history.append(row)
            print(
                f"{rnd + 1:5d} | {len(self.polytope.vertices):8d} | {thr:9.6f} | "
                f"{delta:8.2e} | {witness_obj:11.3e} | "
                f"{certificate_strength:13.3e} | {elapsed:7.2f}"
            )

            if rnd > 1 and delta < 1e-5:
                break
            if last_infeasible is None or wdata is None or wdata["obj"] >= -1e-8:
                break

            W = wdata["W"]
            new_verts = [
                WitnessExtractor.exposed_vertex(W),
                WitnessExtractor.exposed_vertex(P_BAC.T @ W @ P_BAC),
                WitnessExtractor.exposed_vertex(P_CAB.T @ W @ P_CAB),
            ]
            self.polytope.add_vertices(new_verts)
            self.polytope.prune(self.max_vertices)
            self.hierarchy = BSEPHierarchy(self.polytope, self.epsilon)
            self.witness = WitnessExtractor(self.polytope)

        return thresholds


# ============================================
# 10. Genuine bipartite dA x dB adaptive hierarchy
# ============================================
class BipartiteSeparableHierarchy:
    """Inner polytope hierarchy for SEP(C^2 tensor C^dB).

    Set ppt_strengthen_rest=True when dB=4 should be interpreted as C^2 tensor C^2
    and each residual block T_i should also satisfy T_i^Gamma >= 0. This is a
    hierarchy-strengthening option, not part of the basic C^2 tensor C^4 SEP test.
    """

    def __init__(self, polytope, dB=4, epsilon=1e-7, ppt_strengthen_rest=False):
        self.polytope = polytope
        self.dA = 2
        self.dB = dB
        self.dim = self.dA * self.dB
        self.epsilon = epsilon
        self.ppt_strengthen_rest = ppt_strengthen_rest
        if self.ppt_strengthen_rest and self.dB != 4:
            raise ValueError("ppt_strengthen_rest is currently implemented only for dB=4 = 2 x 2.")
        self.rho_param = cp.Parameter((self.dim, self.dim), complex=True, hermitian=True)
        self.last_diagnostics = {}
        self._build_problem()

    def _build_problem(self):
        T_vars = [cp.Variable((self.dB, self.dB), hermitian=True) for _ in self.polytope.vertices]
        rho_approx = balanced_sum(
            [kron_embedding(sig, T) for sig, T in zip(self.polytope.vertices, T_vars)]
        )

        constraints = [sum(cp.trace(T) for T in T_vars) == 1]
        for T in T_vars:
            constraints.append(T >> 0)
            if self.ppt_strengthen_rest:
                constraints.append(partial_transpose_4x4(T) >> 0)

        residual = rho_approx - self.rho_param
        real_part = cp.reshape(cp.real(residual), (self.dim * self.dim, 1), order="C")
        imag_part = cp.reshape(cp.imag(residual), (self.dim * self.dim, 1), order="C")
        constraints.append(cp.sum_squares(real_part) + cp.sum_squares(imag_part) <= self.epsilon**2)

        self.problem = cp.Problem(cp.Minimize(0), constraints)

    def contains(self, rho):
        self.rho_param.value = rho
        try:
            status = solve_problem(self.problem)
            self.last_diagnostics = {
                "status": status,
                "solve_time": getattr(self.problem.solver_stats, "solve_time", None),
                "vertices": len(self.polytope.vertices),
                "ppt_strengthen_rest": self.ppt_strengthen_rest,
            }
            return is_optimal(status)
        except Exception as exc:
            self.last_diagnostics = {
                "status": "exception",
                "error": str(exc),
                "vertices": len(self.polytope.vertices),
                "ppt_strengthen_rest": self.ppt_strengthen_rest,
            }
            return False

    def threshold(self, entangled_fn, max_iter=40, tol=1e-7):
        lo, hi = 0.0, 1.0
        last_infeasible = None
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            rho = target_state(mid, entangled_fn, dim=self.dim)
            if self.contains(rho):
                lo = mid
            else:
                hi = mid
                last_infeasible = rho
            if hi - lo < tol:
                break
        return lo, last_infeasible


class BipartiteWitnessExtractor:
    def __init__(self, polytope, dB=4):
        assert len(polytope.vertices) > 0
        self.polytope = polytope
        self.n_vertices = len(polytope.vertices)
        self.dB = dB
        self.dim = 2 * dB
        self.rho_param = cp.Parameter((self.dim, self.dim), complex=True, hermitian=True)
        self._build_problem()

    def _build_problem(self):
        self.W = cp.Variable((self.dim, self.dim), hermitian=True)
        constraints = [cp.trace(self.W) == 1]
        for sigma in self.polytope.vertices:
            constraints.append(cp.real(cp.trace(kron_I_right(self.dB, sigma) @ self.W)) >= 0)
        objective = cp.Minimize(cp.real(cp.trace(self.W @ self.rho_param)))
        self.problem = cp.Problem(objective, constraints)

    def extract(self, rho):
        if len(self.polytope.vertices) != self.n_vertices:
            raise RuntimeError("BipartiteWitnessExtractor polytope changed after build; rebuild before extract().")
        self.rho_param.value = rho
        try:
            status = solve_problem(self.problem)
            if is_optimal(status):
                return {
                    "W": self.W.value,
                    "obj": self.problem.value,
                    "status": status,
                    "solve_time": getattr(self.problem.solver_stats, "solve_time", None),
                }
        except Exception:
            pass
        return None

    def exposed_vertex(self, W):
        H = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                H[i, j] = np.trace(W[self.dB * i : self.dB * i + self.dB, self.dB * j : self.dB * j + self.dB])
        H = (H + H.conj().T) / 2
        _, v = eigh(H)
        psi = v[:, 0]
        return np.outer(psi, psi.conj())


class BipartiteAdaptiveHierarchy:
    def __init__(self, polytope, dB=4, epsilon=1e-7, ppt_strengthen_rest=False, max_vertices=300):
        self.polytope = polytope
        self.dB = dB
        self.epsilon = epsilon
        self.ppt_strengthen_rest = ppt_strengthen_rest
        self.max_vertices = max_vertices
        self.hierarchy = BipartiteSeparableHierarchy(
            polytope, dB=dB, epsilon=epsilon, ppt_strengthen_rest=ppt_strengthen_rest
        )
        self.witness = BipartiteWitnessExtractor(polytope, dB=dB)
        self.history = []

    def refine(self, entangled_fn, max_rounds=5):
        thresholds = []
        print("\nround | vertices | threshold | delta     | witness_obj | cert_strength | solve_s")
        print("-" * 84)
        for rnd in range(max_rounds):
            start = time.time()
            thr, last_infeasible = self.hierarchy.threshold(entangled_fn)
            elapsed = time.time() - start
            delta = np.nan if not thresholds else abs(thr - thresholds[-1])
            thresholds.append(thr)

            wdata = self.witness.extract(last_infeasible) if last_infeasible is not None else None
            witness_obj = wdata["obj"] if wdata is not None else np.nan

            certificate_strength = -witness_obj if np.isfinite(witness_obj) else np.nan
            row = {
                "round": rnd + 1,
                "vertices": len(self.polytope.vertices),
                "threshold": thr,
                "delta": delta,
                "witness_obj": witness_obj,
                "certificate_strength": certificate_strength,
                "solve_time": elapsed,
            }
            self.history.append(row)
            print(
                f"{rnd + 1:5d} | {len(self.polytope.vertices):8d} | {thr:9.6f} | "
                f"{delta:8.2e} | {witness_obj:11.3e} | "
                f"{certificate_strength:13.3e} | {elapsed:7.2f}"
            )

            if rnd > 1 and delta < 1e-5:
                break
            if last_infeasible is None or wdata is None or wdata["obj"] >= -1e-8:
                break

            self.polytope.add_vertices([self.witness.exposed_vertex(wdata["W"])])
            self.polytope.prune(self.max_vertices)
            self.hierarchy = BipartiteSeparableHierarchy(
                self.polytope,
                dB=self.dB,
                epsilon=self.epsilon,
                ppt_strengthen_rest=self.ppt_strengthen_rest,
            )
            self.witness = BipartiteWitnessExtractor(self.polytope, dB=self.dB)

        return thresholds


# ============================================
# 11. Outer PPT bound
# ============================================
def outer_ppt_bound(pure_fn, max_iter=40):
    def is_ppt_all(p):
        rho = target_state(p, pure_fn)
        eigs = []
        for subsys in ["A", "B", "C"]:
            rho_pt = partial_transpose_8x8_np(rho, subsys)
            eigs.append(np.linalg.eigvalsh((rho_pt + rho_pt.conj().T) / 2).min())
        return min(eigs) >= -1e-8

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if is_ppt_all(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return lo


def print_status(label, hierarchy):
    diag = getattr(hierarchy, "last_diagnostics", {})
    status = diag.get("status", "unknown")
    solve_time = diag.get("solve_time", None)
    if solve_time is None:
        print(f"{label}: status={status}")
    else:
        print(f"{label}: status={status}, solve_time={solve_time:.3f}s")


def timed_threshold(label, hierarchy, state_fn):
    start = time.time()
    threshold, _ = hierarchy.threshold(state_fn)
    elapsed = time.time() - start
    diag = getattr(hierarchy, "last_diagnostics", {})
    return {
        "label": label,
        "vertices": len(hierarchy.polytope.vertices),
        "threshold": threshold,
        "solve_time_s": elapsed,
        "status": "search_completed",
        "last_solver_status": diag.get("status", "unknown"),
        "ppt_strengthened": diag.get("ppt_strengthen_rest", False),
    }


def run_threshold_benchmark_table(
    vertex_counts=(50, 100),
    epsilon=1e-3,
    bipartite_epsilon=1e-7,
    out_csv="ohst_threshold_benchmark.csv",
):
    rows = []
    print("\nBenchmark table: GHZ / W thresholds plus Horodecki additional validation")
    print("vertices | problem                  | threshold | gap       | time_s   | search_status      | last_solver        | ppt")
    print("-" * 119)

    for nv in vertex_counts:
        multi_poly = BlochPolytope.fibonacci(nv)
        multi = BSEPHierarchy(multi_poly, epsilon=epsilon)
        for label, state_fn in [("GHZ multipartite BSEP", ghz_state), ("W multipartite BSEP", w_state)]:
            row = timed_threshold(label, multi, state_fn)
            theory = theory_for_label(label)
            row["theory"] = theory
            row["gap_to_theory"] = theory - row["threshold"] if theory is not None else None
            rows.append(row)
            gap = row["gap_to_theory"]
            gap_text = f"{gap:9.3e}" if gap is not None else "      n/a"
            print(
                f"{row['vertices']:8d} | {row['label']:<24} | {row['threshold']:9.6f} | "
                f"{gap_text} | {row['solve_time_s']:8.2f} | {row['status']:<18} | "
                f"{row['last_solver_status']:<18} | {row['ppt_strengthened']}"
            )

        for ppt_flag in (False, True):
            bip_poly = BlochPolytope.fibonacci(nv)
            bip = BipartiteSeparableHierarchy(
                bip_poly, dB=4, epsilon=bipartite_epsilon, ppt_strengthen_rest=ppt_flag
            )
            label = "Validation: Horodecki 2x4" + (" PPT-rest" if ppt_flag else "")
            row = timed_threshold(label, bip, lambda: horodecki_2x4_state(0.25))
            row["theory"] = None
            row["gap_to_theory"] = None
            rows.append(row)
            print(
                f"{row['vertices']:8d} | {row['label']:<24} | {row['threshold']:9.6f} | "
                f"      n/a | {row['solve_time_s']:8.2f} | {row['status']:<18} | "
                f"{row['last_solver_status']:<18} | {row['ppt_strengthened']}"
            )

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "vertices",
                "threshold",
                "theory",
                "gap_to_theory",
                "solve_time_s",
                "status",
                "last_solver_status",
                "ppt_strengthened",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nBenchmark CSV saved: {out_csv}")
    return rows


THEORETICAL_GHZ_BSEP = 3.0 / 7.0
THEORETICAL_W_THRESHOLD = 0.4790


def theory_for_label(label):
    if label.startswith("GHZ"):
        return THEORETICAL_GHZ_BSEP
    if label.startswith("W"):
        return THEORETICAL_W_THRESHOLD
    return None


def parse_float_list(raw):
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def parse_int_list(raw):
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {path}")


def run_ghz_vertex_sweep(
    vertex_counts=(50, 100, 150, 200, 300),
    epsilon=5e-4,
    max_iter=50,
    out_csv="ohst_ghz_vertex_sweep.csv",
):
    rows = []
    print("\nEXPERIMENT 1: GHZ vertex sweep")
    print("N_input | vertices | epsilon | threshold | gap_to_3/7 | time_s | last_solver")
    print("-" * 88)
    for n_input in vertex_counts:
        poly = BlochPolytope.fibonacci(n_input)
        hier = BSEPHierarchy(poly, epsilon=epsilon)
        start = time.time()
        threshold, _ = hier.threshold(ghz_state, max_iter=max_iter)
        elapsed = time.time() - start
        diag = hier.last_diagnostics
        row = {
            "n_input": n_input,
            "vertices": len(poly.vertices),
            "epsilon": epsilon,
            "threshold": threshold,
            "gap_to_theory": THEORETICAL_GHZ_BSEP - threshold,
            "solve_time_s": elapsed,
            "last_solver_status": diag.get("status", "unknown"),
        }
        rows.append(row)
        print(
            f"{n_input:7d} | {len(poly.vertices):8d} | {epsilon:7.1e} | {threshold:9.6f} | "
            f"{row['gap_to_theory']:10.3e} | {elapsed:6.1f} | {row['last_solver_status']}"
        )
    if len(rows) >= 2:
        assert rows[-1]["threshold"] >= rows[0]["threshold"] - 1e-4, (
            "GHZ vertex sweep failed monotonic reproduction check: "
            f"first={rows[0]['threshold']:.6f}, last={rows[-1]['threshold']:.6f}"
        )
    write_csv(
        out_csv,
        rows,
        ["n_input", "vertices", "epsilon", "threshold", "gap_to_theory", "solve_time_s", "last_solver_status"],
    )
    return rows


def run_ghz_epsilon_sweep(
    vertices=250,
    epsilons=(1e-3, 5e-4, 2e-4, 1e-4),
    max_iter=50,
    out_csv="ohst_ghz_epsilon_sweep.csv",
):
    rows = []
    print("\nEXPERIMENT 2: GHZ epsilon sweep")
    print("N_input | vertices | epsilon | threshold | gap_to_3/7 | time_s | last_solver")
    print("-" * 88)
    for epsilon in epsilons:
        poly = BlochPolytope.fibonacci(vertices)
        hier = BSEPHierarchy(poly, epsilon=epsilon)
        start = time.time()
        threshold, _ = hier.threshold(ghz_state, max_iter=max_iter)
        elapsed = time.time() - start
        diag = hier.last_diagnostics
        row = {
            "n_input": vertices,
            "vertices": len(poly.vertices),
            "epsilon": epsilon,
            "threshold": threshold,
            "gap_to_theory": THEORETICAL_GHZ_BSEP - threshold,
            "solve_time_s": elapsed,
            "last_solver_status": diag.get("status", "unknown"),
        }
        rows.append(row)
        print(
            f"{vertices:7d} | {len(poly.vertices):8d} | {epsilon:7.1e} | {threshold:9.6f} | "
            f"{row['gap_to_theory']:10.3e} | {elapsed:6.1f} | {row['last_solver_status']}"
        )
    if len(rows) >= 2:
        assert rows[-1]["threshold"] >= rows[0]["threshold"] - 1e-4, (
            "GHZ epsilon sweep failed monotonic reproduction check: "
            f"first={rows[0]['threshold']:.6f}, last={rows[-1]['threshold']:.6f}"
        )
    write_csv(
        out_csv,
        rows,
        ["n_input", "vertices", "epsilon", "threshold", "gap_to_theory", "solve_time_s", "last_solver_status"],
    )
    return rows


def run_ghz_adaptive_experiment(
    initial_vertices=250,
    epsilon=5e-4,
    max_rounds=8,
    max_vertices=300,
    out_csv="ohst_ghz_adaptive_rounds.csv",
):
    print("\nEXPERIMENT 3: GHZ adaptive refinement")
    adaptive = AdaptiveBSEP(
        BlochPolytope.fibonacci(initial_vertices),
        epsilon=epsilon,
        max_vertices=max_vertices,
    )
    adaptive.refine(ghz_state, max_rounds=max_rounds)
    rows = []
    for row in adaptive.history:
        rows.append({
            "round": row["round"],
            "vertices": row["vertices"],
            "epsilon": epsilon,
            "threshold": row["threshold"],
            "gap_to_theory": THEORETICAL_GHZ_BSEP - row["threshold"],
            "delta": row["delta"],
            "witness_obj": row["witness_obj"],
            "certificate_strength": row["certificate_strength"],
            "solve_time_s": row["solve_time"],
        })
    write_csv(
        out_csv,
        rows,
        [
            "round",
            "vertices",
            "epsilon",
            "threshold",
            "gap_to_theory",
            "delta",
            "witness_obj",
            "certificate_strength",
            "solve_time_s",
        ],
    )
    return rows


def run_ghz_convergence_suite(
    vertex_counts=(50, 100, 150, 200, 300),
    fixed_vertices=250,
    epsilons=(1e-3, 5e-4, 2e-4, 1e-4),
    epsilon=5e-4,
    adaptive_rounds=8,
    max_vertices=300,
):
    run_ghz_vertex_sweep(vertex_counts=vertex_counts, epsilon=epsilon)
    run_ghz_epsilon_sweep(vertices=fixed_vertices, epsilons=epsilons)
    run_ghz_adaptive_experiment(
        initial_vertices=fixed_vertices,
        epsilon=epsilon,
        max_rounds=adaptive_rounds,
        max_vertices=max_vertices,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Ohst et al. 2024 hierarchy reproduction")
    parser.add_argument(
        "--ghz-convergence-only",
        action="store_true",
        help="Run only the GHZ convergence experiments and write CSV tables.",
    )
    parser.add_argument(
        "--ghz-convergence",
        action="store_true",
        help="Run the normal reproduction and then the GHZ convergence experiments.",
    )
    parser.add_argument(
        "--vertex-counts",
        default="50,100,150,200,300",
        help="Comma-separated N values for the GHZ vertex sweep.",
    )
    parser.add_argument(
        "--fixed-vertices",
        type=int,
        default=250,
        help="Input N for the GHZ epsilon sweep and adaptive run.",
    )
    parser.add_argument(
        "--epsilons",
        default="1e-3,5e-4,2e-4,1e-4",
        help="Comma-separated epsilon values for the GHZ epsilon sweep.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=5e-4,
        help="Primary epsilon for vertex and adaptive GHZ experiments.",
    )
    parser.add_argument(
        "--adaptive-rounds",
        type=int,
        default=8,
        help="Maximum adaptive refinement rounds for the GHZ experiment.",
    )
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=300,
        help="Maximum retained polytope vertices during adaptive refinement.",
    )
    return parser.parse_args()


# ============================================
# 12. Main reproduction entry point
# ============================================
if __name__ == "__main__":
    args = parse_args()
    if args.ghz_convergence_only:
        run_ghz_convergence_suite(
            vertex_counts=parse_int_list(args.vertex_counts),
            fixed_vertices=args.fixed_vertices,
            epsilons=parse_float_list(args.epsilons),
            epsilon=args.epsilon,
            adaptive_rounds=args.adaptive_rounds,
            max_vertices=args.max_vertices,
        )
        raise SystemExit(0)

    print("=" * 80)
    print("Ohst et al. (2024) hierarchy reproduction")
    print("Theoretical GHZ BSEP threshold = 0.42857")
    print("=" * 80)

    poly = BlochPolytope.fibonacci(200)
    print(f"Initial multipartite vertices: {len(poly.vertices)}")

    hier = BSEPHierarchy(poly, epsilon=1e-3)
    thr_ghz, _ = hier.threshold(ghz_state)
    outer_ghz = outer_ppt_bound(ghz_state)
    print(f"\nGHZ: Inner = {thr_ghz:.6f}, Outer PPT = {outer_ghz:.6f}")
    print_status("GHZ inner diagnostic", hier)

    thr_w, _ = hier.threshold(w_state)
    outer_w = outer_ppt_bound(w_state)
    print(f"W  : Inner = {thr_w:.6f}, Outer PPT = {outer_w:.6f} (expected ~0.479)")
    print_status("W inner diagnostic", hier)

    fbsep = FBSEPHierarchy(poly)
    is_fb = fbsep.contains(ghz_state())
    print(f"\nExact FBSEP intersection: GHZ is FBSEP? {is_fb} (should be False)")
    print_status("FBSEP diagnostic", fbsep)

    print("\nMultipartite adaptive refinement, GHZ, start 50 vertices")
    adapt_poly = BlochPolytope.fibonacci(50)
    adaptive = AdaptiveBSEP(adapt_poly, epsilon=1e-3)
    adapt_thrs = adaptive.refine(ghz_state, max_rounds=4)
    print(f"Final multipartite adaptive threshold = {adapt_thrs[-1]:.6f}")

    print("\nAdditional validation benchmark: bipartite 2x4 Horodecki-type state")
    bip_poly = BlochPolytope.fibonacci(100)
    bip_hier = BipartiteSeparableHierarchy(bip_poly, dB=4, epsilon=1e-7)
    thr_horo, _ = bip_hier.threshold(lambda: horodecki_2x4_state(0.25))
    print(f"Additional validation: Horodecki 2x4 white-noise threshold, fixed polytope = {thr_horo:.6f}")
    print_status("Horodecki fixed-polytope diagnostic", bip_hier)

    bip_hier_ppt = BipartiteSeparableHierarchy(
        BlochPolytope.fibonacci(100), dB=4, epsilon=1e-7, ppt_strengthen_rest=True
    )
    thr_horo_ppt, _ = bip_hier_ppt.threshold(lambda: horodecki_2x4_state(0.25))
    print(f"Additional validation: Horodecki 2x4 PPT-strengthened residual threshold = {thr_horo_ppt:.6f}")
    print_status("Horodecki PPT-strengthened diagnostic", bip_hier_ppt)

    print("\nAdditional validation: bipartite adaptive refinement, Horodecki 2x4")
    bip_adapt = BipartiteAdaptiveHierarchy(
        BlochPolytope.fibonacci(50), dB=4, epsilon=1e-7, ppt_strengthen_rest=True, max_vertices=300
    )
    bip_thrs = bip_adapt.refine(lambda: horodecki_2x4_state(0.25), max_rounds=4)
    print(f"Final additional-validation Horodecki threshold = {bip_thrs[-1]:.6f}")

    run_threshold_benchmark_table(vertex_counts=(50, 100), out_csv="ohst_threshold_benchmark.csv")

    if args.ghz_convergence:
        run_ghz_convergence_suite(
            vertex_counts=parse_int_list(args.vertex_counts),
            fixed_vertices=args.fixed_vertices,
            epsilons=parse_float_list(args.epsilons),
            epsilon=args.epsilon,
            adaptive_rounds=args.adaptive_rounds,
            max_vertices=args.max_vertices,
        )

    print("\n" + "=" * 80)
    print("Numerical reproduction completed.")
    print("Horodecki-type state is reported only as an additional validation benchmark.")
    print("FBSEP now uses exact shared-state equality constraints.")
    print("Bipartite hierarchy includes optional PPT-strengthened residual blocks.")
    print("Benchmark automation writes ohst_threshold_benchmark.csv.")
    print("GHZ convergence experiments are available with --ghz-convergence-only.")
