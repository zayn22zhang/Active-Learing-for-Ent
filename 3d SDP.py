import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
import cvxpy as cp
import os

# 1. 量子态构造（密度算符表示）
def get_3qubit_mixed_state(p1, p2):
    rho_GHZ = np.zeros((8, 8), dtype=complex)
    rho_GHZ[0, 0] = 0.5;  rho_GHZ[0, 7] = 0.5
    rho_GHZ[7, 0] = 0.5;  rho_GHZ[7, 7] = 0.5

    rho_W = np.zeros((8, 8), dtype=complex)
    for i in [1, 2, 4]:
        for j in [1, 2, 4]:
            rho_W[i, j] = 1.0 / 3.0
            
    rho_noise = np.eye(8, dtype=complex) / 8.0

    # 混合态: rho = p1*rho_GHZ + p2*rho_W + (1-p1-p2)*I/8
    return p1 * rho_GHZ + p2 * rho_W + (1.0 - p1 - p2) * rho_noise


# 2. Oracle 判定：SDP
def _pt_indices(which):
    """预计算部分转置的索引映射 (i,j) -> (i2,j2)，基序 |abc>=|4a+2b+c>"""
    mapping = {}
    for a in range(2):
      for b in range(2):
        for c in range(2):
          for ap in range(2):
            for bp in range(2):
              for cp in range(2):
                i = 4*a+2*b+c;   j = 4*ap+2*bp+cp
                if   which == 'A': i2=4*ap+2*b+c;  j2=4*a+2*bp+cp
                elif which == 'B': i2=4*a+2*bp+c;  j2=4*ap+2*b+cp
                else:              i2=4*a+2*b+cp;  j2=4*ap+2*bp+c
                mapping[(i, j)] = (i2, j2)
    return mapping

# 预计算三种划分的索引表（只算一次）
_PT_MAP = {q: _pt_indices(q) for q in ['A', 'B', 'C']}

def _sdp_partial_transpose(X, which):
    """对 cvxpy 矩阵变量 X 构造部分转置的线性表达式"""
    n = 8
    result = np.zeros((n, n), dtype=object)
    for (i, j), (i2, j2) in _PT_MAP[which].items():
        result[i2, j2] = X[i, j]
    return cp.bmat([[result[r, c] for c in range(n)] for r in range(n)])

def is_gme_sdp(rho, tol=1e-5):
    rho_r = rho.real  
    n = 8

    R1 = cp.Variable((n, n), symmetric=True)  # rho_{AB|C}
    R2 = cp.Variable((n, n), symmetric=True)  # rho_{AC|B}
    R3 = cp.Variable((n, n), symmetric=True)  # rho_{A|BC}

    def pt_expr(R, which):
        rows = []
        for i2 in range(n):
            row = []
            for j2 in range(n):
                found = False
                for (i, j), (r, s) in _PT_MAP[which].items():
                    if r == i2 and s == j2:
                        row.append(R[i, j])
                        found = True
                        break
                if not found:
                    row.append(cp.Constant(0))
            rows.append(row)
        return cp.bmat(rows)

    constraints = [
        R1 + R2 + R3 == rho_r,   # 分解
        R1 >> 0, R2 >> 0, R3 >> 0,
        pt_expr(R1, 'C') >> 0,   # T_C(rho_{AB|C}) >= 0
        pt_expr(R2, 'B') >> 0,   # T_B(rho_{AC|B}) >= 0
        pt_expr(R3, 'A') >> 0,   # T_A(rho_{A|BC}) >= 0
    ]

    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(solver=cp.SCS, eps=1e-6, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate']:
            return 0  # biseparable
        else:
            return 1  # GME
    except Exception:
        return 1


# 3. 采样数据
res = 40   
p1_vals = np.linspace(0, 1, res)
p2_vals = np.linspace(0, 1, res)
pp1, pp2 = np.meshgrid(p1_vals, p2_vals)

points, labels = [], []
total = sum(1 for p1,p2 in zip(pp1.ravel(), pp2.ravel()) if p1+p2 <= 1.0+1e-9)
done = 0

print(f"基础采样（SDP），共 {total} 点...")
for p1, p2 in zip(pp1.ravel(), pp2.ravel()):
    if p1 + p2 <= 1.0 + 1e-9:
        rho = get_3qubit_mixed_state(p1, p2)
        points.append([p1, p2])
        labels.append(is_gme_sdp(rho))
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{total}  GME: {sum(labels)}  bisep: {done-sum(labels)}")

points = np.array(points)
labels = np.array(labels)
print(f"采样完成: biseparable={(labels==0).sum()}, GME={(labels==1).sum()}")

# 4. 主动学习 (AL)
# Neural Network
model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=5000,
                      alpha=1e-3, random_state=42)

MAX_QUERIES  = 100
STOP_THRESHOLD = 0.001
PATIENCE     = 5

sep_idx = np.where(labels == 0)[0]
ent_idx = np.where(labels == 1)[0]
init_idx = list(sep_idx[:3]) + list(ent_idx[:3])

query_history  = []
last_predictions = None
stable_count   = 0

print("\n>>> 主动学习 <<<")
print("-" * 85)
print(f"{'轮次':^6} | {'查询坐标 (p1, p2)':^20} | {'预测变化率':^12} | {'状态':^20}")
print("-" * 85)

for i in range(MAX_QUERIES):
    n0 = np.sum(labels[init_idx] == 0)
    n1 = np.sum(labels[init_idx] == 1)
    w0 = 1.0 / n0 if n0 > 0 else 1.0
    w1 = 1.0 / n1 if n1 > 0 else 1.0
    sample_weights = np.array([w0 if l == 0 else w1 for l in labels[init_idx]])
    sample_weights /= np.sum(sample_weights)

    try:
        model.fit(points[init_idx], labels[init_idx], sample_weight=sample_weights)
    except Exception as e:
        print(f"R {i+1:02d}  | 训练异常: {e}"); break

    current_predictions = model.predict(points)
    change_rate = 1.0
    if last_predictions is not None:
        change_rate = 1.0 - accuracy_score(last_predictions, current_predictions)
        stable_count = stable_count + 1 if change_rate < STOP_THRESHOLD else 0
    last_predictions = current_predictions.copy()

    probs = model.predict_proba(points)
    ents  = entropy(probs.T)
    ents[init_idx] = -1.0
    q_idx = np.argmax(ents)

    q_p1, q_p2 = points[q_idx]
    status = "Searching..." if stable_count == 0 else f"Stabilizing({stable_count}/{PATIENCE})"
    print(f"R {i+1:02d}  | ({q_p1:.4f}, {q_p2:.4f}) | {change_rate:^12.5f} | {status}")

    init_idx.append(q_idx)
    query_history.append(points[q_idx])

    if stable_count >= PATIENCE:
        print("-" * 85)
        print(f"模型收敛，总采样: {len(init_idx)} 组。")
        break

# 5. 细网格（SDP 直接算，用于背景和边界线）
print("\n>>> 细网格 SDP 判定 <<<")
fine_res = 80  
p1_fine = np.linspace(0, 1, fine_res)
p2_fine = np.linspace(0, 1, fine_res)
P1, P2  = np.meshgrid(p1_fine, p2_fine)
mask    = (P1 + P2) <= 1.0

Z = np.full(P1.shape, np.nan)
total_pts = mask.sum()
count = 0
for idx in np.ndindex(P1.shape):
    if mask[idx]:
        rho = get_3qubit_mixed_state(P1[idx], P2[idx])
        Z[idx] = is_gme_sdp(rho)
        count += 1
        if count % 500 == 0:
            print(f"  进度: {count}/{total_pts}")

print("细网格完成。")


# 6. 可视化
query_history = np.array(query_history)

fig, ax = plt.subplots(figsize=(9, 8))

ax.contourf(P1, P2, Z, levels=[-0.5, 0.5, 1.5],
            colors=['#AED6F1', '#F1948A'], alpha=0.4)
ax.contour(P1, P2, Z, levels=[0.5], colors='black', linewidths=2.5)

ax.scatter(points[labels==0, 0], points[labels==0, 1],
           c='#2980B9', s=12, alpha=0.4, label='Biseparable')
ax.scatter(points[labels==1, 0], points[labels==1, 1],
           c='#C0392B', s=12, alpha=0.4, label='GME')

if query_history.shape[0] > 0:
    ax.scatter(query_history[:, 0], query_history[:, 1],
               c='gold', marker='*', s=220,
               edgecolors='black', linewidths=1.0,
               label='AL Boundary Search', zorder=10)

x_line = np.linspace(0, 1, 200)
ax.plot(x_line, 1 - x_line, 'k--', lw=1.2, alpha=0.4, label='$p_1+p_2=1$')

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel('GHZ Fraction ($p_1$)', fontsize=13)
ax.set_ylabel('W Fraction ($p_2$)', fontsize=13)
ax.set_title('3-Qubit GME Boundary via Active Learning', fontsize=13)
ax.legend(loc='upper right')

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gme_boundary_al.png')
plt.savefig(save_path, dpi=150)
plt.show()
print(f"已保存至: {save_path}")