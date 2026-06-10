"""
AL4QED: Active Learning for Quantum Entanglement Detection
Tools for active learning and quantum entanglement certification.
"""

# 核心模块
from .oracle import AdaptivePolytopeOracle, OracleLabel
from .dataset import generate_dataset, extract_features_advanced, extract_features_simple, random_density_matrix
from .states import horodecki_3x3, werner_state, isotropic_state
from .adaptive import adaptive_polytope_bipartite, partial_transpose
from .network import ChiPredictor, SimpleChiPredictor

# 主动学习模块（如果存在）
try:
    from .active_learning import ActiveLearner
    from .acquisition import get_acquisition_strategy, ACQUISITION_STRATEGIES
except ImportError:
    # 如果文件还未创建，跳过
    pass

__all__ = [
    # Oracle
    "AdaptivePolytopeOracle",
    "OracleLabel",
    
    # Dataset
    "generate_dataset",
    "extract_features_advanced",
    "extract_features_simple",
    "random_density_matrix",
    
    # States
    "horodecki_3x3",
    "werner_state", 
    "isotropic_state",
    
    # Adaptive algorithm
    "adaptive_polytope_bipartite",
    "partial_transpose",
    
    # Neural networks
    "ChiPredictor",
    "SimpleChiPredictor",
    
    # Active learning
    "ActiveLearner",
    "get_acquisition_strategy",
    "ACQUISITION_STRATEGIES",
]