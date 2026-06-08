import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from scipy.stats import entropy

# 2-Qubit 混合态（20% 纠缠比例）
def get_state(x, y):
    phi_p = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho_phi = np.outer(phi_p, phi_p.conj())
    psi_m = np.array([0, 1, -1, 0]) / np.sqrt(2)
    rho_psi = np.outer(psi_m, psi_m.conj())
    noise = np.eye(4) / 4
    
    return (0.78*x-0.18) * rho_phi + (0.78*y-0.18) * rho_psi + (1 - (0.78*(x+y)-0.36)) * noise

def is_entangled(rho):
    m = rho.reshape(2, 2, 2, 2)
    m_pt = np.transpose(m, (0, 3, 2, 1)).reshape(4, 4)
    return 1 if np.min(np.linalg.eigvalsh(m_pt)) < -1e-6 else 0

# 坐标池
res = 200
points = []
for x in np.linspace(0, 1, res):
    for y in np.linspace(0, 1, res):
        if x + y <= 1:
            points.append([x, y])
points = np.array(points)
labels = np.array([is_entangled(get_state(p[0], p[1])) for p in points])
print(f"纠缠态占比: {np.mean(labels):.1%}")

#最大熵采样 (Maximum Entropy Sampling)
init_idx = [0, np.where(labels == 1)[0][-1]] 
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=3000, alpha=1e-3, random_state=42)
query_history = []

for i in range(25):
    num_0 = np.sum(labels[init_idx] == 0)
    num_1 = np.sum(labels[init_idx] == 1)
    w0 = 1.0 / (num_0 if num_0 > 0 else 1)
    w1 = 1.0 / (num_1 if num_1 > 0 else 1)
    sample_weights = np.array([w0 if l == 0 else w1 for l in labels[init_idx]])
    
    model.fit(points[init_idx], labels[init_idx], sample_weight=sample_weights)
    
    probs_all = model.predict_proba(points)
    
    ents = entropy(probs_all.T)
    
    ents[init_idx] = -1.0 
    q_idx = np.argmax(ents)
    
    init_idx.append(q_idx)
    query_history.append(points[q_idx])

# 可视化
plt.figure(figsize=(8, 8))

plt.scatter(points[labels==0, 0], points[labels==0, 1], c='#4A90E2', s=15, alpha=0.3, label='Separable (80%)')
plt.scatter(points[labels==1, 0], points[labels==1, 1], c='#D0021B', s=15, alpha=0.3, label='Entangled (20%)')

# PPT 理论边界 (黑线)
plt.tricontour(points[:, 0], points[:, 1], labels, levels=[0.5], colors='black', linewidths=3)

# 最大熵采样的黄星
query_history = np.array(query_history)
plt.scatter(query_history[:, 0], query_history[:, 1], c='gold', marker='*', s=250, 
            edgecolors='black', linewidths=1.2, label='Max Entropy Queries', zorder=10)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Active Learning: Maximum Entropy Sampling (20% Ratio)', fontsize=14)
plt.legend()
plt.show()