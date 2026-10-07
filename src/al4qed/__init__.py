"""AL4QED: numerical visibility regression and active learning."""
from .oracle import AdaptivePolytopeOracle, OracleLabel
from .dataset import generate_dataset, extract_features_advanced, extract_features_simple, random_density_matrix
from .states import horodecki_3x3, werner_state, isotropic_state
from .active_learning import ActiveLearner
from .acquisition import get_acquisition_strategy, ACQUISITION_STRATEGIES
# Neural models and SDP solvers are optional until explicitly selected.


def __getattr__(name):
    if name in ('ChiPredictor', 'SimpleChiPredictor'):
        from . import network
        return getattr(network, name)
    if name in ('adaptive_polytope_bipartite', 'partial_transpose'):
        from . import adaptive
        return getattr(adaptive, name)
    raise AttributeError(name)
