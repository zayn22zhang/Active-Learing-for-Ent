"""
Neural-network active learning for quantum entanglement detection.

Framework: Fei-style deep semi-supervised / active-learning benchmark.
Oracle: reusable Ohst-style hierarchy oracle from oracle.py.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".matplotlib_cache"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
from oracle import EntanglementOracle
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  量子态构造 —— GHZ-W-White Noise 混合态
#     rho(p1,p2) = p1*rho_GHZ + p2*rho_W + (1-p1-p2)*I/d
# ══════════════════════════════════════════════════════════════════════════════
def ghz_state(n):
    """n-qubit GHZ 密度矩阵"""
    dim = 2**n
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 0.5
    rho[0, dim-1] = 0.5
    rho[dim-1, 0] = 0.5
    rho[dim-1, dim-1] = 0.5
    return rho


def w_state(n):
    """n-qubit W 密度矩阵"""
    dim = 2**n
    rho = np.zeros((dim, dim), dtype=complex)
    w_basis = [1 << k for k in range(n)]   
    for i in w_basis:
        for j in w_basis:
            rho[i, j] = 1.0 / n
    return rho


def mixed_state(n, p1, p2):
    """
    GHZ-W-Noise 混合态
    """
    dim = 2**n
    return (p1 * ghz_state(n)
            + p2 * w_state(n)
            + (1.0 - p1 - p2) * np.eye(dim, dtype=complex) / dim)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Oracle —— reusable Ohst-style oracle.py
# ══════════════════════════════════════════════════════════════════════════════
_ORACLE_CACHE = {}


def get_oracle(n, vertices=80, epsilon=5e-4):
    """Cache one oracle per qubit/parameter setting.

    n=2 uses exact two-qubit PPT separability.
    n=3 uses the Ohst-style BSEP/GME hierarchy from oracle.py.
    """
    key = (n, vertices, epsilon)
    if key not in _ORACLE_CACHE:
        _ORACLE_CACHE[key] = EntanglementOracle(vertices=vertices, epsilon=epsilon)
    return _ORACLE_CACHE[key]


def new_ohst_oracle_label(rho, n, vertices=80, epsilon=5e-4):
    """Return binary label for active learning: 0=separable/biseparable, 1=entangled/GME."""
    oracle = get_oracle(n, vertices=vertices, epsilon=epsilon)
    if n == 2:
        result = oracle.classify_two_qubit(rho)
    elif n == 3:
        result = oracle.classify_three_qubit(rho)
    else:
        raise ValueError("The current oracle-backed benchmark supports n=2 and n=3.")

    if result.oracle_label == -1:
        # For binary NN training, keep uncertain PPT-entangled cases on the positive side.
        # The diagnostics remain available if one wants to filter these later.
        return 1
    return int(result.oracle_label)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  sample pool（用 SDP oracle 打标签）
# ══════════════════════════════════════════════════════════════════════════════
def build_sdp_pool(n, res=25, N_verts=80, epsilon=5e-4, verbose=True):
    """
    在三角区域 {p1>=0, p2>=0, p1+p2<=1} 上均匀采样，
    用 oracle.py 中的新 Ohst-style oracle 打标签。
    """
    p1v = np.linspace(0, 1, res)
    p2v = np.linspace(0, 1, res)
    P1, P2 = np.meshgrid(p1v, p2v)
    mask = (P1 + P2) <= 1.0 + 1e-9

    pts_list, lbs_list = [], []
    flat_p1 = P1[mask]
    flat_p2 = P2[mask]
    total = len(flat_p1)

    rng = np.random.default_rng(42)

    if verbose:
        print(f"  [{n}-qubit] 新 oracle 采样 {total} 点 (res={res}, vertices={N_verts}, epsilon={epsilon})...")

    for k, (p1, p2) in enumerate(zip(flat_p1, flat_p2)):
        rho = mixed_state(n, p1, p2)
        lb = new_ohst_oracle_label(rho, n, vertices=N_verts, epsilon=epsilon)
        pts_list.append([p1, p2])
        lbs_list.append(lb)
        if verbose and (k + 1) % 30 == 0:
            print(f"    {k+1}/{total}  GME={sum(lbs_list)}  bisep={k+1-sum(lbs_list)}")

    pts = np.array(pts_list)
    lbs = np.array(lbs_list)
    if verbose:
        print(f"  [{n}-qubit] 完成: bisep={(lbs==0).sum()}, GME={(lbs==1).sum()}")
    return pts, lbs


# ══════════════════════════════════════════════════════════════════════════════
# 4.  主动学习的Benchmark 实验
# ══════════════════════════════════════════════════════════════════════════════
def run_al_benchmark(n, pool_pts, pool_lbs,
                     max_queries=80,
                     patience=5,
                     stop_threshold=0.001):
    """
    在预计算的 pool 上跑主动学习，追踪 hold-out 准确率。
    返回 (query_counts, acc_history, query_coords)
    """
    np.random.seed(42)
    N = len(pool_lbs)

    # Hold-out 划分（20%）
    test_size = max(20, int(N * 0.2))
    test_idx = np.random.choice(N, test_size, replace=False)
    train_mask = np.ones(N, dtype=bool)
    train_mask[test_idx] = False

    train_pts = pool_pts[train_mask]
    train_lbs = pool_lbs[train_mask]
    test_pts  = pool_pts[test_idx]
    test_lbs  = pool_lbs[test_idx]

    # 初始种子：各类各取 3 个
    sep_idx = np.where(train_lbs == 0)[0]
    gme_idx = np.where(train_lbs == 1)[0]
    n_init = min(3, len(sep_idx), len(gme_idx))
    labeled = list(sep_idx[:n_init]) + list(gme_idx[:n_init])

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=3000,
        alpha=1e-3,
        random_state=42
    )

    acc_history   = []
    query_counts  = []
    query_coords  = []
    last_preds    = None
    stable_count  = 0

    print(f"\n  [{n}-qubit] 主动学习开始 (pool={len(train_pts)}, test={len(test_pts)})")
    print(f"  {'轮':>4} | {'(p1, p2)':^16} | {'变化率':^10} | {'准确率':^8} | 状态")
    print("  " + "-"*65)

    actual_max = min(max_queries, len(train_pts) - 1)

    for i in range(actual_max):
        y = train_lbs[labeled]
        
        n0, n1 = np.sum(y==0), np.sum(y==1)
        w0 = 1.0/n0 if n0 > 0 else 1.0
        w1 = 1.0/n1 if n1 > 0 else 1.0
        sw = np.array([w0 if l==0 else w1 for l in y])
        sw /= sw.sum()

        try:
            model.fit(train_pts[labeled], y, sample_weight=sw)
        except Exception as e:
            print(f"  训练异常: {e}")
            break

        # 准确率
        acc = accuracy_score(test_lbs, model.predict(test_pts))
        acc_history.append(acc)
        query_counts.append(len(labeled))

        # 收敛检测
        cur_preds = model.predict(train_pts)
        change_rate = 1.0
        if last_preds is not None:
            change_rate = 1.0 - accuracy_score(last_preds, cur_preds)
            stable_count = stable_count+1 if change_rate < stop_threshold else 0
        last_preds = cur_preds.copy()

        # 打印进度
        if len(labeled) > n_init * 2:
            p1q, p2q = train_pts[labeled[-1]]
            status = f"收敛中({stable_count}/{patience})" if stable_count > 0 else "搜索中..."
            print(f"  {i+1:>4} | ({p1q:.3f},{p2q:.3f})   | {change_rate:^10.5f} | {acc:^8.4f} | {status}")

        if stable_count >= patience:
            print(f"  [{n}-qubit] 收敛！查询 {len(labeled)} 次，最终准确率 {acc:.4f}")
            break

        # 最大熵主动查询
        probs = model.predict_proba(train_pts)
        ents = entropy(probs.T)
        ents[labeled] = -1.0
        next_q = np.argmax(ents)
        labeled.append(next_q)
        query_coords.append(train_pts[next_q])

    return query_counts, acc_history, np.array(query_coords) if query_coords else np.empty((0,2))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  主流程
# ══════════════════════════════════════════════════════════════════════════════

QUBIT_CONFIGS = [2, 3]        
RES_MAP       = {2: 28, 3: 22, 4: 18, 5: 14}
NVERTS_MAP    = {2: 20, 3: 80}
ORACLE_EPSILON = 5e-4
MAX_QUERIES   = 80



def main():
    print("=" * 70)
    print("New oracle.py Ohst-style Oracle + Neural Network Active Learning Benchmark")
    print("量子态: GHZ-W-White Noise 混合态")
    print("=" * 70)

    # 步骤 A: SDP 预采样
    print("\n【步骤 A】SDP Oracle 预采样")
    all_pools = {}
    for n in QUBIT_CONFIGS:
        pts, lbs = build_sdp_pool(n,
                                  res=RES_MAP[n],
                                  N_verts=NVERTS_MAP[n],
                                  epsilon=ORACLE_EPSILON,
                                  verbose=True)
        all_pools[n] = (pts, lbs)

    # 步骤 B: 主动学习准确率实验
    print("\n【步骤 B】主动学习准确率 Benchmark")
    results = {}
    for n in QUBIT_CONFIGS:
        pts, lbs = all_pools[n]
        qc, acc, qcoords = run_al_benchmark(n, pts, lbs, max_queries=MAX_QUERIES)
        results[n] = (qc, acc, qcoords)

    # ══════════════════════════════════════════════════════════════════════════════
    # 6.  绘图 A: 准确率 vs 查询次数（Benchmark 主图）
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n【步骤 C】绘图")

    colors  = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    markers = ['o', 's', '^', 'D']

    fig1, ax1 = plt.subplots(figsize=(9, 5.5))

    for idx, n in enumerate(QUBIT_CONFIGS):
        qc, acc, _ = results[n]
        ax1.plot(qc, acc,
                 marker=markers[idx], color=colors[idx],
                 label=f'{n}-qubit (new oracle.py)',
                 markersize=5, lw=1.8, alpha=0.9)

    ax1.axhline(0.95, color='gray', ls='--', alpha=0.6, label='95% 参考线')

    ax1.set_title(
        "Active Learning Benchmark: Accuracy vs. SDP Oracle Queries\n"
        r"(Oracle: reusable Ohst-style oracle.py · GHZ-W-Noise Mixed State)",
        fontsize=11
    )
    ax1.set_xlabel("Number of SDP Oracle Queries (Labeled Data Size)", fontsize=11)
    ax1.set_ylabel("Accuracy on Hold-out Test Set", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.set_ylim(0.4, 1.02)
    ax1.set_xlim(left=0)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.2f}'))

    plt.tight_layout()
    fig1.savefig('ohst2024_al_accuracy_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("  准确率图已保存: ohst2024_al_accuracy_benchmark.png")

    # ══════════════════════════════════════════════════════════════════════════════
    # 7.  展示 SDP oracle 给出的 BSEP/GME 边界 + AL 查询点
    # ══════════════════════════════════════════════════════════════════════════════
    for n in QUBIT_CONFIGS:
        pts, lbs = all_pools[n]
        _, _, qcoords = results[n]

        fig2, ax2 = plt.subplots(figsize=(6.5, 6))

        # 散点（pool）
        ax2.scatter(pts[lbs==0, 0], pts[lbs==0, 1],
                    c='#AED6F1', s=25, alpha=0.6,
                    edgecolors='#2980B9', lw=0.4, label='Biseparable (SDP)')
        ax2.scatter(pts[lbs==1, 0], pts[lbs==1, 1],
                    c='#F1948A', s=25, alpha=0.6,
                    edgecolors='#C0392B', lw=0.4, label='GME (SDP)')

        # AL 查询点
        if len(qcoords) > 0:
            ax2.scatter(qcoords[:, 0], qcoords[:, 1],
                        c='gold', marker='*', s=200,
                        edgecolors='black', lw=0.8,
                        label='AL Query Points', zorder=10)

        # p1+p2=1 边界线
        xline = np.linspace(0, 1, 200)
        ax2.plot(xline, 1-xline, 'k--', lw=1.0, alpha=0.4, label='$p_1+p_2=1$')

        # 三顶点标注
        ax2.text(0.01, 0.97, 'Noise\n$I/d$', ha='left', va='top', fontsize=9, color='#555555')
        ax2.text(0.97, 0.01, 'GHZ', ha='right', va='bottom', fontsize=9, color='#1a5276')
        ax2.text(0.01, 0.97 - 0.97, 'W', ha='left', va='top', fontsize=9, color='#7b241c')
        ax2.annotate('GHZ', xy=(1,0), xytext=(0.85, 0.06),
                     fontsize=9, color='#1a5276',
                     arrowprops=dict(arrowstyle='->', color='#1a5276', lw=0.8))
        ax2.annotate('W', xy=(0,1), xytext=(0.07, 0.85),
                     fontsize=9, color='#7b241c',
                     arrowprops=dict(arrowstyle='->', color='#7b241c', lw=0.8))

        ax2.set_xlim(-0.02, 1.02)
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_xlabel('GHZ Fraction $p_1$', fontsize=12)
        ax2.set_ylabel('W Fraction $p_2$', fontsize=12)
        ax2.set_title(
            f'{n}-Qubit GME Phase Diagram\n'
            r'(Ohst $et\ al.$ 2024 BSEP-SDP Oracle, Fig.4a cross-section)',
            fontsize=11
        )
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_aspect('equal')

        fname = f'ohst2024_{n}qubit_phase_diagram.png'
        fig2.tight_layout()
        fig2.savefig(fname, dpi=200, bbox_inches='tight')
        plt.show()
        print(f"  相图已保存: {fname}")

    print("\n全部完成。")

if __name__ == "__main__":
    main()
