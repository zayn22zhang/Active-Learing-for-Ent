# acquisition.py (新文件)

"""
主动学习采集策略：选择信息量最大的未标注样本
"""

import numpy as np
from typing import List, Tuple, Optional


def random_acquisition(model, unlabeled_pool, n_queries, **kwargs):
    """基线策略：随机采样"""
    indices = np.random.choice(len(unlabeled_pool), n_queries, replace=False)
    return indices, [1.0] * n_queries


def uncertainty_acquisition(model, unlabeled_pool, n_queries, n_mc_samples=20, **kwargs):
    """不确定性采样：选择模型最不确定的样本"""
    uncertainties = []
    for item in unlabeled_pool:
        _, std = model.predict_uncertainty(item['features'], n_mc_samples)
        uncertainties.append(std)
    
    indices = np.argsort(uncertainties)[-n_queries:]
    scores = [uncertainties[i] for i in indices]
    return indices, scores


def boundary_acquisition(model, unlabeled_pool, n_queries, 
                         threshold=0.99, n_mc_samples=20, 
                         beta=1.0, **kwargs):
    """
    边界聚焦采样：选择靠近分类边界且不确定性高的样本
    
    核心公式: score = uncertainty / (|χ - threshold| + ε) ^ β
    
    其中:
    - uncertainty: MC Dropout标准差
    - |χ - threshold|: 到边界的距离
    - β: 边界聚焦强度 (β越大越聚焦边界)
    """
    scores = []
    for item in unlabeled_pool:
        chi_mean, chi_std = model.predict_uncertainty(item['features'], n_mc_samples)
        boundary_dist = np.abs(chi_mean - threshold)
        
        # 信息量评分：不确定性高 + 靠近边界
        info_score = chi_std / (boundary_dist + 1e-6) ** beta
        scores.append(info_score)
    
    indices = np.argsort(scores)[-n_queries:]
    return indices, [scores[i] for i in indices]


def margin_acquisition(model, unlabeled_pool, n_queries, 
                       threshold=0.99, n_mc_samples=20, **kwargs):
    """
    边界裕度采样：选择预测值最接近阈值的样本
    
    这是边界聚焦的简化版本，只考虑预测值位置
    """
    boundary_dists = []
    for item in unlabeled_pool:
        chi_mean, _ = model.predict_uncertainty(item['features'], n_mc_samples)
        boundary_dists.append(np.abs(chi_mean - threshold))
    
    indices = np.argsort(boundary_dists)[:n_queries]
    return indices, [boundary_dists[i] for i in indices]


def hybrid_acquisition(model, unlabeled_pool, n_queries, 
                       threshold=0.99, n_mc_samples=20,
                       uncertainty_weight=0.5, **kwargs):
    """
    混合策略：平衡探索（高不确定性）和开发（边界附近）
    
    score = w * uncertainty_norm + (1-w) * (1 - boundary_dist_norm)
    """
    uncertainties = []
    boundary_dists = []
    
    for item in unlabeled_pool:
        chi_mean, chi_std = model.predict_uncertainty(item['features'], n_mc_samples)
        uncertainties.append(chi_std)
        boundary_dists.append(np.abs(chi_mean - threshold))
    
    # 归一化
    uncertainties = np.array(uncertainties)
    boundary_dists = np.array(boundary_dists)
    
    u_norm = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min() + 1e-6)
    b_norm = (boundary_dists - boundary_dists.min()) / (boundary_dists.max() - boundary_dists.min() + 1e-6)
    
    # 组合分数：高不确定性 + 低边界距离
    scores = uncertainty_weight * u_norm + (1 - uncertainty_weight) * (1 - b_norm)
    
    indices = np.argsort(scores)[-n_queries:]
    return indices, [scores[i] for i in indices]


# 采集策略注册表
ACQUISITION_STRATEGIES = {
    'random': random_acquisition,
    'uncertainty': uncertainty_acquisition,
    'boundary': boundary_acquisition,
    'margin': margin_acquisition,
    'hybrid': hybrid_acquisition,
}


def get_acquisition_strategy(name):
    """获取采集策略函数"""
    if name not in ACQUISITION_STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Choose from {list(ACQUISITION_STRATEGIES.keys())}")
    return ACQUISITION_STRATEGIES[name]