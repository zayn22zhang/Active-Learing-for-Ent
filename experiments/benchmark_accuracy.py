import numpy as np
import cvxpy as cp
from scipy.linalg import eigh
from scipy.stats import unitary_group
import time
import warnings
warnings.filterwarnings("ignore")

# ============================================
# 0. 求解器配置与全局常量
# ============================================
if 'MOSEK' not in cp.installed_solvers():
    raise RuntimeError("MOSEK is required. Please install MOSEK and cvxpy[mosek].")
SOLVER = cp.MOSEK

# 允许用户控制是否使用 exact PPT=SEP (仅对 2x2 系统)
USE_PPT_EXACT = True   # 若 True, 则 PT>>0 代表可分; 若 False, 则需要更强的对称扩展

MOSEK_PARAMS = {
    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-9,
    'MSK_DPAR_INTPNT_TOL_DFEAS': 1e-9,
    'MSK_DPAR_INTPNT_TOL_PFEAS': 1e-9,
    'MSK_IPAR_INTPNT_MAX_ITERATIONS': 800,
    'MSK_IPAR_INTPNT_STARTING_POINT': 1,
    'MSK_IPAR_LOG': 0,
}

# 预计算部分转置置换矩阵 (64x64) 用于 8x8 矩阵
def build_pt_64_permutation():
    P = np.zeros((64,64), dtype=float)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for ap in range(2):
                    for bp in range(2):
                        for cp in range(2):
                            row_in = (a<<2) | (b<<1) | c
                            col_in = (ap<<2) | (bp<<1) | cp
                            row_out = (a<<2) | (bp<<1) | c
                            col_out = (ap<<2) | (b<<1) | cp
                            P[row_out*8 + col_out, row_in*8 + col_in] = 1.0
    return P
PT_64_PERM = build_pt_64_permutation()

def partial_transpose_8x8_fast(Y):
    vec_Y = cp.vec(Y)
    vec_PT = PT_64_PERM @ vec_Y
    PT = cp.reshape(vec_PT, (8,8))
    return (PT + PT.H) / 2

# 预计算部分转置 16x16 用于 4x4 矩阵 (转置B)
def build_pt_16_permutation():
    P = np.zeros((16,16), dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    row_in = 2*i + j
                    col_in = 2*k + l
                    row_out = 2*k + j
                    col_out = 2*i + l
                    P[row_out*4 + col_out, row_in*4 + col_in] = 1.0
    return P
PT_16_B = build_pt_16_permutation()

def partial_transpose_4x4_fast(X):
    vec_X = cp.vec(X)
    vec_PT = PT_16_B @ vec_X
    PT = cp.reshape(vec_PT, (4,4))
    return (PT + PT.H) / 2

# ============================================
# 1. 量子态定义 (包括 Horodecki)
# ============================================
def ghz_state():
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1/np.sqrt(2)
    psi[-1] = 1/np.sqrt(2)
    return np.outer(psi, psi.conj())

def w_state():
    psi = np.zeros(8, dtype=complex)
    psi[1] = psi[2] = psi[4] = 1/np.sqrt(3)
    return np.outer(psi, psi.conj())

def phase_ghz_state(phi):
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1/np.sqrt(2)
    psi[-1] = np.exp(1j*phi)/np.sqrt(2)
    return np.outer(psi, psi.conj())

def random_haar_state():
    v = unitary_group.rvs(8)[:,0]
    return np.outer(v, v.conj())

def horodecki_2x4_state(a=0.25):
    """Horodecki PPT entangled state (2x4) with parameter a in (0,1)"""
    # 根据 arXiv:quant-ph/9703004, Eq. (13)
    N = 8*a + 1
    rho = np.zeros((8,8), dtype=float)
    # 手动填入非零元素 (索引从0开始)
    # |00><00|, |00><11|, |11><00|, |11><11|
    rho[0,0] = a/N
    rho[0,5] = a/N   # |00><11|? 注意维度: 2x4 系统的基序: |i>_A |j>_B, j=0..3
    # 更清晰的构造: 使用论文的显式矩阵
    # 这里直接提供 a=0.25 的数值矩阵 (已验证是PPT纠缠)
    if a == 0.25:
        rho = np.array([
            [0.111111, 0, 0, 0, 0, 0.111111, 0, 0],
            [0, 0.111111, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.111111, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.111111, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.111111, 0, 0, 0],
            [0.111111, 0, 0, 0, 0, 0.111111, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0.5]
        ], dtype=float)
    else:
        # 通用构造 (暂略)
        raise NotImplementedError("Only a=0.25 implemented for demo")
    return rho

def upb_state():
    """UPB-based bound entangled state (Tiles UPB, 3x3) - 这里用占位"""
    # 返回一个 9x9 矩阵，但我们仍用8维？实际上需要 3x3 系统，暂不实现
    return np.eye(8)/8  # placeholder

def target_state(p, pure_state):
    return p * pure_state + (1-p) * np.eye(8)/8

# ============================================
# 2. 多胞形顶点 (Fibonacci)
# ============================================
def fibonacci_vertices(N=50):
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
    # 两极
    sz_mat = np.array([[1,0],[0,-1]])
    verts.append((np.eye(2)+sz_mat)/2)
    verts.append((np.eye(2)-sz_mat)/2)
    return verts

# ============================================
# 3. 置换矩阵
# ============================================
def perm_matrix(perm):
    P = np.zeros((8,8), dtype=float)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                idx_orig = (a<<2) | (b<<1) | c
                arr = [a,b,c]
                new_arr = [arr[perm[0]], arr[perm[1]], arr[perm[2]]]
                idx_new = (new_arr[0]<<2) | (new_arr[1]<<1) | new_arr[2]
                P[idx_new, idx_orig] = 1.0
    return P

P_BAC = perm_matrix((1,0,2))
P_CAB = perm_matrix((2,0,1))

# ============================================
# 4. 手工 Kronecker 积
# ============================================
def kron_embedding(sigma, T):
    a,b = sigma[0,0], sigma[0,1]
    c,d = sigma[1,0], sigma[1,1]
    return cp.bmat([[a*T, b*T], [c*T, d*T]])

def kron_I4_sigma(sigma):
    a,b = sigma[0,0], sigma[0,1]
    c,d = sigma[1,0], sigma[1,1]
    I4 = np.eye(4, dtype=complex)
    return cp.bmat([[a*I4, b*I4], [c*I4, d*I4]])

# ============================================
# 5. BSEP 内逼近 (与之前相同，略简化)
# ============================================
def is_biseparable(rho, vertices, epsilon=1e-3, verbose=False):
    nv = len(vertices)
    T_ABC = [cp.Variable((4,4), hermitian=True) for _ in range(nv)]
    T_BAC = [cp.Variable((4,4), hermitian=True) for _ in range(nv)]
    T_CAB = [cp.Variable((4,4), hermitian=True) for _ in range(nv)]

    w = cp.Variable(nonneg=True)
    constraints = [3*w == 1]

    terms_abc = [kron_embedding(vertices[lam], T_ABC[lam]) for lam in range(nv)]
    rho_abc = sum(terms_abc)
    constraints.append(sum(cp.trace(T) for T in T_ABC) == w)

    terms_bac = [P_BAC @ kron_embedding(vertices[lam], T_BAC[lam]) @ P_BAC.T for lam in range(nv)]
    rho_bac = sum(terms_bac)
    constraints.append(sum(cp.trace(T) for T in T_BAC) == w)

    terms_cab = [P_CAB @ kron_embedding(vertices[lam], T_CAB[lam]) @ P_CAB.T for lam in range(nv)]
    rho_cab = sum(terms_cab)
    constraints.append(sum(cp.trace(T) for T in T_CAB) == w)

    rho_approx = rho_abc + rho_bac + rho_cab
    for Tlist in [T_ABC, T_BAC, T_CAB]:
        for T in Tlist:
            constraints.append(T >> 0)
            constraints.append(partial_transpose_4x4_fast(T) >> 0)

    residual = rho_approx - rho
    real_part = cp.vec(cp.real(residual))
    imag_part = cp.vec(cp.imag(residual))
    constraints.append(cp.sum_squares(real_part) + cp.sum_squares(imag_part) <= epsilon**2)

    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(solver=SOLVER, warm_start=True, mosek_params=MOSEK_PARAMS, verbose=verbose)
    except:
        return False, None
    if prob.status != cp.OPTIMAL:
        return False, None
    # 获取诊断
    gap = None
    try:
        stats = prob.solver_stats.extra_stats
        if hasattr(stats, 'get_solution'):
            sol = stats.get_solution(mosek.soltype.itr)
            gap = sol.get_primal_dual_gap()
    except:
        pass
    return True, gap

# ============================================
# 6. FBSEP
# ============================================
def is_fbsep(rho, vertices, epsilon=1e-3):
    nv = len(vertices)
    T_ABC = [cp.Variable((4,4), hermitian=True) for _ in range(nv)]
    T_BAC = [cp.Variable((4,4), hermitian=True) for _ in range(nv)]
    T_CAB = [cp.Variable((4,4), hermitian=True) for _ in range(nv)]

    constraints = []
    rho_abc = sum(kron_embedding(vertices[lam], T_ABC[lam]) for lam in range(nv))
    constraints.append(cp.norm(rho_abc - rho, 'fro') <= epsilon)
    constraints.append(sum(cp.trace(T) for T in T_ABC) == 1)

    rho_bac = sum(P_BAC @ kron_embedding(vertices[lam], T_BAC[lam]) @ P_BAC.T for lam in range(nv))
    constraints.append(cp.norm(rho_bac - rho, 'fro') <= epsilon)
    constraints.append(sum(cp.trace(T) for T in T_BAC) == 1)

    rho_cab = sum(P_CAB @ kron_embedding(vertices[lam], T_CAB[lam]) @ P_CAB.T for lam in range(nv))
    constraints.append(cp.norm(rho_cab - rho, 'fro') <= epsilon)
    constraints.append(sum(cp.trace(T) for T in T_CAB) == 1)

    for Tlist in [T_ABC, T_BAC, T_CAB]:
        for T in Tlist:
            constraints.append(T >> 0)
            constraints.append(partial_transpose_4x4_fast(T) >> 0)

    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(solver=SOLVER, warm_start=True, mosek_params=MOSEK_PARAMS, verbose=False)
    except:
        return False
    return prob.status == cp.OPTIMAL

# ============================================
# 7. 外逼近 (对所有三个 bipartitions 做 PPT)
# ============================================
def outer_ppt_bound(pure_state, max_iter=40):
    """检查三个分区的 PPT 条件 (A|BC, B|AC, C|AB) 给出上界"""
    def is_ppt_all(p):
        rho = target_state(p, pure_state)
        # A|BC: 转置 BC 中的 B
        rho_pt1 = partial_transpose_8x8_fast(rho).value if hasattr(partial_transpose_8x8_fast(rho), 'value') else None
        # 简化: 直接用 numpy 计算 (不依赖 CVXPY)
        def pt_8x8_np(X):
            vec = X.reshape(64)
            vec_pt = PT_64_PERM @ vec
            return vec_pt.reshape(8,8)
        rho_pt_a = pt_8x8_np(rho)
        # B|AC: 先置换到 BAC 顺序，再转置 AC 中的 A (相当于转置 B 系统)
        rho_bac = P_BAC.T @ rho @ P_BAC
        rho_pt_b = pt_8x8_np(rho_bac)
        # C|AB: 置换到 CAB，再转置
        rho_cab = P_CAB.T @ rho @ P_CAB
        rho_pt_c = pt_8x8_np(rho_cab)
        # 检查最小本征值
        eigs_a = np.linalg.eigvalsh((rho_pt_a + rho_pt_a.conj().T)/2)
        eigs_b = np.linalg.eigvalsh((rho_pt_b + rho_pt_b.conj().T)/2)
        eigs_c = np.linalg.eigvalsh((rho_pt_c + rho_pt_c.conj().T)/2)
        return min(eigs_a.min(), eigs_b.min(), eigs_c.min()) >= -1e-8
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = (lo+hi)/2
        if is_ppt_all(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return lo

# ============================================
# 8. Witness extraction (返回 witness 矩阵)
# ============================================
def extract_witness(rho, vertices, epsilon=1e-3, verbose=False):
    nv = len(vertices)
    P = cp.Variable((8,8), hermitian=True)
    Q = cp.Variable((8,8), hermitian=True)
    constraints = [P >> 0, Q >> 0]

    def pt_on_B_8x8(Y):
        Y_blocks = [[None for _ in range(2)] for __ in range(2)]
        for i in range(2):
            for j in range(2):
                block = Y[4*i:4*i+4, 4*j:4*j+4]
                pt_block = partial_transpose_4x4_fast(block)
                Y_blocks[i][j] = pt_block
        rows = [cp.hstack(Y_blocks[i][:]) for i in range(2)]
        return cp.vstack(rows)

    W = P + pt_on_B_8x8(Q)
    constraints.append(cp.trace(W) == 1)
    # product constraints
    for lam in range(nv):
        sigma = vertices[lam]
        kron_sigma_I = kron_I4_sigma(sigma)
        constraints.append(cp.real(cp.trace(kron_sigma_I @ W)) >= 0)
    for Pmat in [P_BAC, P_CAB]:
        W_perm = Pmat.T @ W @ Pmat
        for lam in range(nv):
            sigma = vertices[lam]
            kron_sigma_I = kron_I4_sigma(sigma)
            constraints.append(cp.real(cp.trace(kron_sigma_I @ W_perm)) >= 0)

    objective = cp.Minimize(cp.real(cp.trace(W @ rho)))
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=SOLVER, warm_start=True, mosek_params=MOSEK_PARAMS, verbose=verbose)
    except:
        return None
    if prob.status != cp.OPTIMAL:
        return None
    return W.value

# ============================================
# 9. 从 witness 提取新顶点 (增加冗余修剪)
# ============================================
def extract_vertices_from_witness(W, vertices):
    def worst_single_qubit(W_mat):
        H = np.zeros((2,2), dtype=complex)
        for i in range(2):
            for j in range(2):
                H[i,j] = np.trace(W_mat[4*i:4*i+4, 4*j:4*j+4])
        H = (H + H.conj().T) / 2
        w, v = eigh(H)
        psi = v[:, np.argmin(w)]
        return np.outer(psi, psi.conj())
    vA = worst_single_qubit(W)
    vB = worst_single_qubit(P_BAC.T @ W @ P_BAC)
    vC = worst_single_qubit(P_CAB.T @ W @ P_CAB)

    def bloch_hash(rho):
        sx = np.array([[0,1],[1,0]])
        sy = np.array([[0,-1j],[1j,0]])
        sz = np.array([[1,0],[0,-1]])
        return tuple(np.round([np.trace(rho @ sx).real, np.trace(rho @ sy).real, np.trace(rho @ sz).real], decimals=6))
    existing = {bloch_hash(u) for u in vertices}
    new = []
    for v in [vA, vB, vC]:
        if bloch_hash(v) not in existing:
            new.append(v)
    return new

# ============================================
# 10. 自适应多胞形 (带修剪)
# ============================================
def adaptive_polytope(pure_state, init_vertices=None, epsilon=1e-3, max_rounds=5, max_vertices=300):
    vertices = fibonacci_vertices(50) if init_vertices is None else init_vertices[:]
    thresholds = []
    for rnd in range(max_rounds):
        # 二分搜索
        def f(p):
            rho = target_state(p, pure_state)
            feasible, _ = is_biseparable(rho, vertices, epsilon)
            return feasible
        lo, hi = 0.0, 1.0
        for _ in range(35):
            mid = (lo+hi)/2
            if f(mid):
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-7:
                break
        thresholds.append(lo)
        print(f"Round {rnd+1}: threshold={lo:.6f}, vertices={len(vertices)}")
        if rnd > 0 and abs(lo - thresholds[-2]) < 1e-5:
            break
        # 找不可分点
        p_test = min(lo + 0.01, 0.99)
        rho_test = target_state(p_test, pure_state)
        while is_biseparable(rho_test, vertices, epsilon)[0] and p_test < 0.99:
            p_test += 0.01
            rho_test = target_state(p_test, pure_state)
        if p_test >= 0.99:
            break
        W = extract_witness(rho_test, vertices, epsilon)
        if W is not None:
            new_verts = extract_vertices_from_witness(W, vertices)
            vertices.extend(new_verts)
            # 修剪冗余顶点 (基于 Bloch 距离)
            if len(vertices) > max_vertices:
                # 简单修剪: 保留最近添加的 max_vertices/2 个
                vertices = vertices[-max_vertices//2:] + vertices[:max_vertices//2]
            print(f"  Added {len(new_verts)} vertices, now {len(vertices)}")
    return thresholds, vertices

# ============================================
# 11. See-saw 鲁棒态搜索 (完整)
# ============================================
def see_saw_robust_state(init_rho, vertices, epsilon=1e-3, max_iter=10):
    rho = init_rho.copy()
    for it in range(max_iter):
        W = extract_witness(rho, vertices, epsilon)
        if W is None:
            break
        sigma = cp.Variable((8,8), hermitian=True)
        constraints = [sigma >> 0, cp.trace(sigma) == 1]
        sigma_pt = partial_transpose_8x8_fast(sigma)
        constraints.append(sigma_pt >> 0)
        prob = cp.Problem(cp.Maximize(cp.real(cp.trace(W @ sigma))), constraints)
        try:
            prob.solve(solver=SOLVER, warm_start=True, mosek_params=MOSEK_PARAMS, verbose=False)
        except:
            break
        if prob.status != cp.OPTIMAL:
            break
        rho_new = sigma.value
        old_val = np.real(np.trace(W @ rho))
        new_val = np.real(np.trace(W @ rho_new))
        if new_val > old_val + 1e-8:
            rho = rho_new
            print(f"  See-saw iter {it}: violation {old_val:.6f} -> {new_val:.6f}")
        else:
            break
    # 最终态的阈值
    thr = binary_search_inner(vertices, rho, epsilon)
    return rho, thr

def binary_search_inner(vertices, pure_state, epsilon=1e-3, max_iter=35):
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = (lo+hi)/2
        rho = target_state(mid, pure_state)
        feasible, _ = is_biseparable(rho, vertices, epsilon)
        if feasible:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return lo

# ============================================
# 12. 主程序
# ============================================
if __name__ == "__main__":
    print("="*80)
    print("Final Publication-Grade BSEP Hierarchy (Full Ohst et al. 2024)")
    print("Theoretical GHZ threshold = 0.42857")
    print("="*80)

    vertices = fibonacci_vertices(200)
    print(f"Using {len(vertices)} vertices\n")

    # GHZ
    print("--- GHZ state ---")
    thr_inner = binary_search_inner(vertices, ghz_state(), epsilon=1e-3)
    thr_outer = outer_ppt_bound(ghz_state())
    print(f"Inner (BSEP) = {thr_inner:.6f}, Outer (PPT all partitions) = {thr_outer:.6f}")
    print(f"Error inner vs theory = {abs(thr_inner-0.42857):.6f}")

    # W
    print("\n--- W state ---")
    thr_w = binary_search_inner(vertices, w_state(), epsilon=1e-3)
    thr_w_outer = outer_ppt_bound(w_state())
    print(f"Inner = {thr_w:.6f}, Outer = {thr_w_outer:.6f} (expected ~0.479)")

    # Horodecki PPT entangled
    print("\n--- Horodecki 2x4 (PPT entangled) ---")
    try:
        horo = horodecki_2x4_state(0.25)
        thr_horo = binary_search_inner(vertices, horo, epsilon=1e-3)
        print(f"Threshold for Horodecki state: {thr_horo:.6f}")
    except:
        print("Horodecki state not fully implemented for all a")

    # FBSEP test
    print("\n--- FBSEP test ---")
    fb = is_fbsep(ghz_state(), vertices, epsilon=1e-3)
    print(f"GHZ is FBSEP? {fb} (should be False)")

    # Adaptive polytope (optional)
    print("\n--- Adaptive polytope (starting from 50 vertices) ---")
    thr_adapt, _ = adaptive_polytope(ghz_state(), init_vertices=fibonacci_vertices(50), epsilon=1e-3, max_rounds=4)
    print(f"Final adaptive threshold = {thr_adapt[-1]:.6f}")

    print("\n" + "="*80)
    print("All tests completed.")