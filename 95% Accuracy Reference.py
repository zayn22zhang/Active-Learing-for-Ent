import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import cvxpy as cp
import os

# =====================================================================
# 1. 量子态构造 (通用 N-Qubit, 来自代码1的泛化思路)
# =====================================================================
def get_nqubit_mixed_state(n, p1, p2):
    """
    泛化 N-Qubit 密度矩阵
    p1: GHZ 比例, p2: W 比例, (1-p1-p2): 噪声比例
    """
    dim = 2**n
    rho_ghz = np.zeros((dim, dim), dtype=complex)
    rho_ghz[0, 0] = 0.5
    rho_ghz[0, dim-1] = 0.5
    rho_ghz[dim-1, 0] = 0.5
    rho_ghz[dim-1, dim-1] = 0.5

    rho_w = np.zeros((dim, dim), dtype=complex)
    w_indices = [1 << i for i in range(n)]
    for i in w_indices:
        for j in w_indices:
            rho_w[i, j] = 1.0 / n

    rho_noise = np.eye(dim, dtype=complex) / float(dim)
    return p1 * rho_ghz + p2 * rho_w + (1.0 - p1 - p2) * rho_noise


# =====================================================================
# 2. SDP Oracle —— 真实判定 (来自代码2，泛化到 N-Qubit)
#    对每种二分割 (bipartition) 检验 biseparability
# =====================================================================
def _get_all_bipartitions(n):
    """枚举所有非平凡二分割 (subsystem A | rest)，返回 A 的比特位列表"""
    bipartitions = []
    for mask in range(1, 2**n - 1):
        complement = ((1 << n) - 1) ^ mask
        # 避免重复 (A|B) == (B|A)
        if mask < complement:
            bipartitions.append(mask)
    return bipartitions


def _partial_transpose_matrix(rho_real, party_mask, n):
    """
    对密度矩阵 rho 的子系统 party_mask 做部分转置（numpy 版本，用于构建 cvxpy 约束）。
    返回重排后的 (dim x dim) numpy 索引映射，供 cvxpy 线性组合使用。
    """
    dim = 2**n
    PT = np.zeros((dim, dim), dtype=object)
    for i in range(dim):
        for j in range(dim):
            # 对属于 party_mask 的比特位做转置
            i2, j2 = i, j
            for bit in range(n):
                if party_mask & (1 << bit):
                    # 交换 i 和 j 在该比特位上的值
                    bi = (i >> bit) & 1
                    bj = (j >> bit) & 1
                    if bi != bj:
                        i2 = i2 ^ (1 << bit)  # flip bit in i2
                        j2 = j2 ^ (1 << bit)  # flip bit in j2
            PT[i2, j2] = rho_real[i, j]
    return PT


def is_biseparable_sdp(rho, n, tol=1e-5):
    """
    用 SDP 判断 rho 是否为 biseparable (返回 0) 或 GME (返回 1)。
    判据：若 rho 可分解为各二分割的可分态之和，则为 biseparable。
    即: rho = sum_k sigma_k，其中 sigma_k 在 bipartition k 下 PPT。
    """
    dim = 2**n
    rho_r = rho.real
    bipartitions = _get_all_bipartitions(n)
    num_bp = len(bipartitions)

    # 为每个二分割创建一个半正定矩阵变量
    R_vars = [cp.Variable((dim, dim), symmetric=True) for _ in range(num_bp)]

    constraints = [cp.sum(R_vars) == rho_r]  # 分解约束

    for k, bp_mask in enumerate(bipartitions):
        constraints.append(R_vars[k] >> 0)  # 半正定
        # 构建部分转置的线性表达式
        PT_expr_rows = []
        for i2 in range(dim):
            row = []
            for j2 in range(dim):
                # 找到哪个 (i,j) 的部分转置映射到 (i2,j2)
                # 即逆映射：部分转置是自逆的，所以再做一次PT即可
                # 直接构造：PT[i2,j2] = R_vars[k][original_i, original_j]
                # 反查：对 (i2,j2) 做同样的 bit-flip 得到 (i,j)
                i_orig, j_orig = i2, j2
                for bit in range(n):
                    if bp_mask & (1 << bit):
                        bi = (i2 >> bit) & 1
                        bj = (j2 >> bit) & 1
                        if bi != bj:
                            i_orig = i_orig ^ (1 << bit)
                            j_orig = j_orig ^ (1 << bit)
                row.append(R_vars[k][i_orig, j_orig])
            PT_expr_rows.append(row)
        PT_expr = cp.bmat(PT_expr_rows)
        constraints.append(PT_expr >> 0)  # PPT 约束

    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(solver=cp.SCS, eps=1e-6, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate']:
            return 0  # biseparable
        else:
            return 1  # GME
    except Exception:
        return 1  # 求解失败保守判为 GME


# =====================================================================
# 3. 为每个 qubit 数预计算基础采样点和标签 (SDP Oracle)
# =====================================================================
def build_pool_with_sdp(n_qubits, res=25, verbose=True):
    """
    在 (p1, p2) 三角区域采样，用真实 SDP 打标签。
    res 控制网格分辨率（越大越精确但越慢）。
    """
    p1_vals = np.linspace(0, 1, res)
    p2_vals = np.linspace(0, 1, res)
    P1, P2 = np.meshgrid(p1_vals, p2_vals)
    mask = (P1 + P2) <= 1.0 + 1e-9

    points, labels = [], []
    flat_p1 = P1[mask]
    flat_p2 = P2[mask]
    total = len(flat_p1)

    if verbose:
        print(f"  [{n_qubits}-Qubit] SDP 采样 {total} 个点 (res={res})...")

    for idx, (p1, p2) in enumerate(zip(flat_p1, flat_p2)):
        rho = get_nqubit_mixed_state(n_qubits, p1, p2)
        label = is_biseparable_sdp(rho, n_qubits)
        points.append([p1, p2])
        labels.append(label)
        if verbose and (idx + 1) % 50 == 0:
            print(f"    进度: {idx+1}/{total}")

    points = np.array(points)
    labels = np.array(labels)
    if verbose:
        print(f"  [{n_qubits}-Qubit] 完成: biseparable={(labels==0).sum()}, GME={(labels==1).sum()}")
    return points, labels


# =====================================================================
# 4. 核心实验：主动学习 + 准确率追踪 (代码1的验证思路)
# =====================================================================
def run_al_accuracy_experiment(n_qubits, pool_points, pool_labels, max_queries=80):
    """
    在预先计算好的 pool 上运行主动学习，
    追踪每轮对独立测试集的准确率（来自代码1的 benchmark 思路）。
    """
    # 独立测试集：从 pool 中随机留出 20% 作为 hold-out
    np.random.seed(42)
    n_total = len(pool_labels)
    test_size = max(20, int(n_total * 0.2))
    test_idx = np.random.choice(n_total, test_size, replace=False)
    train_pool_mask = np.ones(n_total, dtype=bool)
    train_pool_mask[test_idx] = False

    pool_pts_train = pool_points[train_pool_mask]
    pool_lbs_train = pool_labels[train_pool_mask]
    test_pts = pool_points[test_idx]
    test_lbs = pool_labels[test_idx]

    # 初始化：各取若干种子点（保证两类都有）
    sep_idx = np.where(pool_lbs_train == 0)[0]
    gme_idx = np.where(pool_lbs_train == 1)[0]
    n_init = min(3, len(sep_idx), len(gme_idx))
    labeled_idx = list(sep_idx[:n_init]) + list(gme_idx[:n_init])

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=2000,
        alpha=1e-3,
        random_state=42
    )

    acc_history = []
    sample_counts = []
    last_preds = None
    stable_count = 0
    PATIENCE = 5
    STOP_THRESHOLD = 0.001

    actual_max = min(max_queries, len(pool_pts_train) - 1)

    for i in range(actual_max):
        y_train = pool_lbs_train[labeled_idx]

        # 加权训练（应对类不平衡，来自代码2）
        n0 = np.sum(y_train == 0)
        n1 = np.sum(y_train == 1)
        w0 = 1.0 / n0 if n0 > 0 else 1.0
        w1 = 1.0 / n1 if n1 > 0 else 1.0
        sample_w = np.array([w0 if l == 0 else w1 for l in y_train])
        sample_w /= sample_w.sum()

        try:
            model.fit(pool_pts_train[labeled_idx], y_train, sample_weight=sample_w)
        except Exception:
            break

        # 准确率记录（来自代码1）
        y_pred = model.predict(test_pts)
        acc = accuracy_score(test_lbs, y_pred)
        acc_history.append(acc)
        sample_counts.append(len(labeled_idx))

        # 收敛检测（来自代码2）
        current_preds = model.predict(pool_pts_train)
        if last_preds is not None:
            change_rate = 1.0 - accuracy_score(last_preds, current_preds)
            stable_count = stable_count + 1 if change_rate < STOP_THRESHOLD else 0
        last_preds = current_preds.copy()

        if stable_count >= PATIENCE:
            print(f"  [{n_qubits}-Qubit] 收敛，共 {len(labeled_idx)} 次查询")
            break

        # 最大熵主动查询（来自代码2）
        probs = model.predict_proba(pool_pts_train)
        ents = entropy(probs.T)
        ents[labeled_idx] = -1.0
        next_q = np.argmax(ents)
        labeled_idx.append(next_q)

    return sample_counts, acc_history


# =====================================================================
# 5. 主流程：多 Qubit 对比实验 + 绘图
# =====================================================================
QUBIT_CONFIGS = [2, 3, 4, 5]

# 每个 qubit 数对应的网格分辨率
# 高维 SDP 慢，适当降低分辨率
RES_MAP = {2: 30, 3: 25, 4: 20, 5: 15}
MAX_QUERIES = 80

# --- 步骤 A：用 SDP Oracle 预采样 ---
print("=" * 60)
print("步骤 A：SDP Oracle 预采样（真实量子判定）")
print("=" * 60)

all_pool_data = {}
for n in QUBIT_CONFIGS:
    pts, lbs = build_pool_with_sdp(n, res=RES_MAP[n], verbose=True)
    all_pool_data[n] = (pts, lbs)

# --- 步骤 B：主动学习准确率实验 ---
print("\n" + "=" * 60)
print("步骤 B：主动学习收敛性验证（准确率追踪）")
print("=" * 60)

results = {}
for n in QUBIT_CONFIGS:
    print(f"\n[{n}-Qubit] 开始主动学习...")
    pts, lbs = all_pool_data[n]
    x, y = run_al_accuracy_experiment(n, pts, lbs, max_queries=MAX_QUERIES)
    results[n] = (x, y)
    print(f"  最终准确率: {y[-1]:.4f} (经过 {x[-1]} 次查询)")

# --- 步骤 C：绘图（代码1的 benchmark 风格）---
print("\n" + "=" * 60)
print("步骤 C：绘制准确率对比图")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

# 颜色方案（对标学术图表）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

for idx, n in enumerate(QUBIT_CONFIGS):
    x, y = results[n]
    ax.plot(x, y,
            marker=markers[idx],
            color=colors[idx],
            label=f'{n}-Qubit (SDP Oracle)',
            markersize=5,
            lw=1.8,
            alpha=0.85)

# 95% 参考线（来自代码1）
ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.6,
           label='95% Accuracy Reference')

ax.set_title(
    "Active Learning Sample Efficiency: Accuracy vs. Oracle Queries\n"
    "(SDP Oracle · GHZ-W-Noise Mixed State · 2–5 Qubits)",
    fontsize=12
)
ax.set_xlabel("Number of SDP Oracle Queries (Labeled Data Size)", fontsize=11)
ax.set_ylabel("Prediction Accuracy on Hold-out Test Set", fontsize=11)
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0.4, 1.02)
ax.set_xlim(left=0)

plt.tight_layout()
save_name = 'SDP_Oracle_AL_Accuracy_2to5Qubits.png'
plt.savefig(save_name, dpi=300, bbox_inches='tight')
plt.show()
print(f"\n图表已保存：{save_name}")

# =====================================================================
# 6. （可选）额外输出：3-Qubit 的相空间边界图（来自代码2的可视化）
# =====================================================================
print("\n绘制 3-Qubit GME 边界图（参数空间可视化）...")

n = 3
pts3, lbs3 = all_pool_data[3]

fig2, ax2 = plt.subplots(figsize=(8, 7))

# 背景色区域
from matplotlib.tri import Triangulation
tri = Triangulation(pts3[:, 0], pts3[:, 1])
ax2.tripcolor(tri, lbs3, cmap='RdBu_r', alpha=0.35, vmin=-0.2, vmax=1.2)

ax2.scatter(pts3[lbs3 == 0, 0], pts3[lbs3 == 0, 1],
            c='#2980B9', s=18, alpha=0.5, label='Biseparable')
ax2.scatter(pts3[lbs3 == 1, 0], pts3[lbs3 == 1, 1],
            c='#C0392B', s=18, alpha=0.5, label='GME')

x_line = np.linspace(0, 1, 200)
ax2.plot(x_line, 1 - x_line, 'k--', lw=1.2, alpha=0.4, label='$p_1+p_2=1$')

ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.set_xlabel('GHZ Fraction ($p_1$)', fontsize=12)
ax2.set_ylabel('W Fraction ($p_2$)', fontsize=12)
ax2.set_title('3-Qubit GME Phase Diagram (SDP Oracle)', fontsize=12)
ax2.legend(loc='upper right')
plt.tight_layout()

save_name2 = 'SDP_3Qubit_GME_PhaseDiagram.png'
plt.savefig(save_name2, dpi=200, bbox_inches='tight')
plt.show()
print(f"相图已保存：{save_name2}")
