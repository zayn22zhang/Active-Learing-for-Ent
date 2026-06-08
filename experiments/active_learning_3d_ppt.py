import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from scipy.stats import entropy as scipy_entropy

def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

def make_ghz_mixed(p: float) -> np.ndarray:
    """p * |GHZ><GHZ| + (1-p) * I/8"""
    ghz = np.zeros(8, dtype=complex)
    ghz[0] = ghz[7] = 1 / np.sqrt(2)
    rho_ghz = np.outer(ghz, ghz.conj())
    return p * rho_ghz + (1 - p) * np.eye(8) / 8

def make_w_mixed(p: float) -> np.ndarray:
    """p * |W><W| + (1-p) * I/8"""
    w = np.zeros(8, dtype=complex)
    w[1] = w[2] = w[4] = 1 / np.sqrt(3) 
    rho_w = np.outer(w, w.conj())
    return p * rho_w + (1 - p) * np.eye(8) / 8

def make_product_state() -> np.ndarray:
    """随机三比特积态 rho_A ⊗ rho_B ⊗ rho_C"""
    def random_qubit():
        z = np.random.randn(2) + 1j * np.random.randn(2)
        z = _normalize(z)
        return np.outer(z, z.conj())
    rho = random_qubit()
    for _ in range(2):
        rho = np.kron(rho, random_qubit())
    return rho

def generate_3qubit_state() -> np.ndarray:
    """
    三类态各占约 1/3：
      0 → GHZ 混态
      1 → W 混态
      2 → 纯积态（不与 GHZ 混合，保证可分性清晰）
    """
    choice = np.random.randint(3)
    p = np.random.rand()
    if choice == 0:
        return make_ghz_mixed(p)
    elif choice == 1:
        return make_w_mixed(p)
    else:
        return make_product_state()   


def partial_transpose(rho: np.ndarray, subsystem: str) -> np.ndarray:
    """
    对 2⊗2⊗2 系统做指定子系统的部分转置。
    subsystem: 'A'(2|4), 'B'(4|2 in B|AC sense), 'C'(4|2)
    """
    if subsystem == 'A':
        t = rho.reshape(2, 4, 2, 4)
        t = t.transpose(2, 1, 0, 3)
    elif subsystem == 'B':
        t = rho.reshape(2, 2, 2, 2, 2, 2)   
        t = t.transpose(0, 4, 2, 3, 1, 5)  
        t = t.reshape(8, 8)
        return t
    elif subsystem == 'C':
        t = rho.reshape(2, 2, 2, 2, 2, 2)
        t = t.transpose(0, 1, 5, 3, 4, 2)
    else:
        raise ValueError(f"Unknown subsystem '{subsystem}'")
    return t.reshape(8, 8)

def is_entangled_ppt(rho: np.ndarray, tol: float = 1e-10) -> bool:
    for sub in ('A', 'C'):   
        rho_pt = partial_transpose(rho, sub)
        eigvals = np.linalg.eigvalsh(rho_pt)
        if np.any(eigvals < -tol):
            return True
    return False


def _random_pure_state_dm(dim: int) -> np.ndarray:
    z = np.random.randn(dim) + 1j * np.random.randn(dim)
    z /= np.linalg.norm(z)
    return np.outer(z, z.conj())

def oracle_adaptive_polytope(
    rho: np.ndarray,
    dim_A: int = 2,
    dim_B: int = 4,
    N: int = 10,
    tol: float = 1e-3,
) -> int:
    d = dim_A * dim_B
    sigma_A = [_random_pure_state_dm(dim_A) for _ in range(N)]

    t = cp.Variable(nonneg=True)
    tau_list = [cp.Variable((dim_B, dim_B), hermitian=True) for _ in range(N)]
    rhs = cp.sum([cp.kron(sigma_A[i], tau_list[i]) for i in range(N)])

    I_d = np.eye(d) / d
    lhs = t * rho + (1 - t) * I_d

    constraints = [
        t <= 1,
        lhs == rhs,
        *[tau >> 0 for tau in tau_list],
    ]

    prob = cp.Problem(cp.Maximize(t), constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-4)
    except cp.error.SolverError:
        return 1  

    if t.value is None:
        return 1
    
    return int(t.value < 1.0 - tol)



def build_dataset(N: int, seed: int = 42) -> tuple:
    np.random.seed(seed)
    rho_list = [generate_3qubit_state() for _ in range(N)]
    X = np.array([r.real.flatten() for r in rho_list])
    return X, rho_list



def entropy_uncertainty(model: MLPClassifier, X: np.ndarray) -> int:
    probs = model.predict_proba(X)
    entropies = scipy_entropy(probs.T)   # shape (n_samples,)
    return int(np.argmax(entropies))

def make_model() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        early_stopping=True,        # 停
        validation_fraction=0.15,
        n_iter_no_change=20,
        max_iter=500,
        random_state=42,
    )


def run_active_learning(
    pool_X: np.ndarray,
    pool_rhos: list,
    test_X: np.ndarray,
    test_y: np.ndarray,
    initial_size: int = 30,
    n_iterations: int = 15,
    seed: int = 0,
) -> tuple:
    rng = np.random.default_rng(seed)

    initial_idx = rng.choice(len(pool_X), initial_size, replace=False)
    mask = np.ones(len(pool_X), dtype=bool)
    mask[initial_idx] = False

    X_lab = pool_X[initial_idx].copy()
    y_lab = np.array([int(is_entangled_ppt(pool_rhos[i])) for i in initial_idx])

    X_unlab = pool_X[mask].copy()
    rhos_unlab = [pool_rhos[i] for i in np.where(mask)[0]]

    accuracy_curve = []
    query_points_2d_list = []   # 收集查询点（供可视化）

    print(f"{'轮':>3} | {'标注数':>6} | {'测试准确率':>10} | Oracle 标注")
    print("-" * 50)

    for i in range(n_iterations):
        model = make_model()
        model.fit(X_lab, y_lab)

        acc = accuracy_score(test_y, model.predict(test_X))
        accuracy_curve.append(acc)
        print(f"{i+1:3d} | {len(y_lab):6d} | {acc:10.2%}", end="")

        q_idx = entropy_uncertainty(model, X_unlab)
        rho_q = rhos_unlab[q_idx]

        y_q = oracle_adaptive_polytope(rho_q)
        print(f" | {'纠缠' if y_q else '可分'}")

        query_points_2d_list.append(X_unlab[q_idx])

        X_lab = np.vstack([X_lab, X_unlab[q_idx]])
        y_lab = np.hstack([y_lab, y_q])
        X_unlab = np.delete(X_unlab, q_idx, axis=0)
        del rhos_unlab[q_idx]

    final_model = make_model()
    final_model.fit(X_lab, y_lab)
    final_acc = accuracy_score(test_y, final_model.predict(test_X))
    accuracy_curve.append(final_acc)
    print(f"\n最终测试准确率: {final_acc:.2%}")

    return accuracy_curve, np.array(query_points_2d_list), final_model

def plot_results(
    pool_X: np.ndarray,
    pool_rhos: list,
    query_points: np.ndarray,
    accuracy_curve: list,
    initial_size: int,
):
    y_all = np.array([int(is_entangled_ppt(r)) for r in pool_rhos])

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(pool_X)
    q_2d = pca.transform(query_points)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("3-Qubit Entanglement Detection via Active Learning", fontsize=15, fontweight="bold")

    # —— 左图：PCA 边界与查询点 ——
    ax = axes[0]
    colors = {0: "cornflowerblue", 1: "crimson"}
    labels = {0: "Separable (PPT)", 1: "Entangled (PPT)"}
    for cls in (0, 1):
        idx = y_all == cls
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
                   c=colors[cls], s=18, alpha=0.5, label=labels[cls])
    ax.scatter(q_2d[:, 0], q_2d[:, 1],
               c="yellow", s=60, marker="o",
               edgecolors="black", linewidths=0.5,
               zorder=5, label="Oracle Queries (entropy)")
    ax.set_title("Entanglement Boundary (PCA)")
    ax.set_xlabel("PC-1")
    ax.set_ylabel("PC-2")
    ax.legend(fontsize=9)

    # —— 右图：准确率学习曲线 ——
    ax2 = axes[1]
    x_ticks = list(range(initial_size, initial_size + len(accuracy_curve)))
    ax2.plot(x_ticks, accuracy_curve, marker="o", color="steelblue", linewidth=2)
    ax2.axhline(accuracy_curve[-1], color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_title("Active Learning Accuracy Curve")
    ax2.set_xlabel("Number of Labeled Samples")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_ylim(0.02, 1.02)
    ax2.grid(True, alpha=0.3)
    ax2.annotate(f"Final: {accuracy_curve[-1]:.1%}",
                 xy=(x_ticks[-1], accuracy_curve[-1]),
                 xytext=(-60, -20), textcoords="offset points",
                 fontsize=10, color="steelblue",
                 arrowprops=dict(arrowstyle="->", color="steelblue"))

    plt.tight_layout()
    plt.savefig("entanglement_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("图表已保存到 entanglement_result.png")


if __name__ == "__main__":
    POOL_SIZE  = 800
    TEST_SIZE  = 150
    INIT_SIZE  = 30
    N_ITER     = 12

    print("生成数据集...")
    all_X, all_rhos = build_dataset(POOL_SIZE + TEST_SIZE, seed=42)

    # 测试集
    test_X  = all_X[:TEST_SIZE]
    test_y  = np.array([int(is_entangled_ppt(r)) for r in all_rhos[:TEST_SIZE]])
    pool_X  = all_X[TEST_SIZE:]
    pool_rhos = all_rhos[TEST_SIZE:]

    print(f"测试集纠缠比例: {test_y.mean():.1%}\n")

    print("开始主动学习（Oracle = 自适应多胞体 SDP）...")
    acc_curve, query_pts, _ = run_active_learning(
        pool_X, pool_rhos, test_X, test_y,
        initial_size=INIT_SIZE,
        n_iterations=N_ITER,
    )

    print("\n绘制结果...")
    plot_results(pool_X, pool_rhos, query_pts, acc_curve, INIT_SIZE)