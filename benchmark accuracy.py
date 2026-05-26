import numpy as np
import cvxpy as cp
from scipy.linalg import eigh
import warnings
warnings.filterwarnings("ignore")

# ============================================
# 1. 量子态定义
# ============================================
def ghz_state(n=3):
    dim = 2**n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1/np.sqrt(2)
    psi[-1] = 1/np.sqrt(2)
    return np.outer(psi, psi.conj())

def ghz_noise_state(p):
    return p * ghz_state(3) + (1-p) * np.eye(8)/8

# ============================================
# 2. 固定多胞形顶点（高密度）
# ============================================
def fibonacci_vertices(N=500):
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    verts = []
    phi = (1+np.sqrt(5))/2
    for k in range(N):
        z = 1 - 2*(k+0.5)/N
        r = np.sqrt(max(0, 1-z*z))
        theta = 2*np.pi*k/phi
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        bloch = np.array([x,y,z])
        rho = (np.eye(2) + bloch[0]*sx + bloch[1]*sy + bloch[2]*sz)/2
        verts.append(rho)
    # 添加两极
    verts.append((np.eye(2)+sz)/2)
    verts.append((np.eye(2)-sz)/2)
    return verts

# ============================================
# 3. 部分转置（子系统 B 或 C）
# ============================================
def partial_transpose_4x4_cvx(X, system=0):
    Y_raw = np.empty((4,4), dtype=object)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    if system == 0:   # 转置 B
                        row_out = 2*k + j
                        col_out = 2*i + l
                    else:             # 转置 C
                        row_out = 2*i + l
                        col_out = 2*k + j
                    row_in = 2*i + j
                    col_in = 2*k + l
                    Y_raw[row_out][col_out] = X[row_in, col_in]
    PT = cp.bmat(Y_raw.tolist())
    return (PT + PT.H) / 2

# ============================================
# 4. 置换矩阵（B|AC 和 C|AB 分区）
# ============================================
P_BAC = np.zeros((8,8))
for a in range(2):
    for b in range(2):
        for c in range(2):
            col = (b<<2) | (a<<1) | c   # B A C
            row = (a<<2) | (b<<1) | c   # A B C
            P_BAC[row, col] = 1.0

P_CAB = np.zeros((8,8))
for a in range(2):
    for b in range(2):
        for c in range(2):
            col = (a<<2) | (b<<1) | c   # A B C
            row = (c<<2) | (a<<1) | b   # C A B
            P_CAB[row, col] = 1.0

# ============================================
# 5. BSEP 可行性 SDP（三分区，固定多胞形）
# ============================================
def is_biseparable(rho, vertices, epsilon=1e-3):
    nv = len(vertices)
    T_ABC = [cp.Variable((4,4), hermitian=True) for _ in range(nv)]
    T_BAC = [cp.Variable((4,4), hermitian=True) for _ in range(nv)]
    T_CAB = [cp.Variable((4,4), hermitian=True) for _ in range(nv)]

    wA = cp.Variable(nonneg=True)
    wB = cp.Variable(nonneg=True)
    wC = cp.Variable(nonneg=True)
    constraints = [wA + wB + wC == 1]

    rho_approx = 0
    # A|BC
    for lam in range(nv):
        sigma = vertices[lam] / cp.trace(vertices[lam])
        rho_approx += cp.kron(sigma, T_ABC[lam])
    constraints.append(sum(cp.trace(T) for T in T_ABC) == wA)

    # B|AC
    for lam in range(nv):
        sigma = vertices[lam] / cp.trace(vertices[lam])
        raw = cp.kron(sigma, T_BAC[lam])
        rho_approx += P_BAC @ raw @ P_BAC.T
    constraints.append(sum(cp.trace(T) for T in T_BAC) == wB)

    # C|AB
    for lam in range(nv):
        sigma = vertices[lam] / cp.trace(vertices[lam])
        raw = cp.kron(sigma, T_CAB[lam])
        rho_approx += P_CAB @ raw @ P_CAB.T
    constraints.append(sum(cp.trace(T) for T in T_CAB) == wC)

    # PPT 约束（统一转置 B 子系统）
    for Tlist in [T_ABC, T_BAC, T_CAB]:
        for T in Tlist:
            constraints.append(T >> 0)
            constraints.append(partial_transpose_4x4_cvx(T, system=0) >> 0)

    constraints.append(cp.norm(rho_approx - rho, 'fro') <= epsilon)

    prob = cp.Problem(cp.Minimize(0), constraints)
    solver = cp.MOSEK if 'MOSEK' in cp.installed_solvers() else cp.SCS
    try:
        if solver == cp.MOSEK:
            prob.solve(solver=cp.MOSEK, verbose=False)
        else:
            prob.solve(solver=cp.SCS, eps=1e-5, max_iters=5000, verbose=False)
    except:
        return False
    return prob.status in ['optimal', 'optimal_inaccurate']

# ============================================
# 6. 二分搜索阈值（固定顶点集和 epsilon）
# ============================================
def compute_threshold(vertices, epsilon=1e-3, tol=1e-6, max_iter=45):
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = (lo+hi)/2
        rho = ghz_noise_state(mid)
        if is_biseparable(rho, vertices, epsilon):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo

# ============================================
# 7. 主程序：固定顶点，逐步收紧 epsilon
# ============================================
if __name__ == "__main__":
    print("="*70)
    print("Fixed high-density polytope (no adaptive addition)")
    print("Target exact BSEP bound for GHZ = 0.42857")
    print("="*70)
    vertices = fibonacci_vertices(500)
    print(f"Using {len(vertices)} vertices (Fibonacci + poles)")
    epsilons = [1e-3, 5e-4, 2e-4, 1e-4]
    thresholds = []
    for eps in epsilons:
        print(f"\n--- epsilon = {eps:.0e} ---")
        thr = compute_threshold(vertices, epsilon=eps, tol=1e-6)
        thresholds.append(thr)
        print(f"Threshold = {thr:.6f}")
    print("\n" + "="*70)
    print("Results summary:")
    for eps, thr in zip(epsilons, thresholds):
        print(f"epsilon = {eps:.0e} : threshold = {thr:.6f}")
    print("Theoretical exact BSEP bound = 0.42857")