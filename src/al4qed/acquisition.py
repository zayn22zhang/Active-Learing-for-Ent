"""Batched acquisition. 'boundary' targets a numerical visibility level, not SEP."""
import numpy as np


def _count(pool, n):
    if int(n) != n or n < 0:
        raise ValueError('n_queries must be a nonnegative integer')
    return min(int(n), len(pool))


def _top(scores, count):
    if count == 0:
        return np.array([], dtype=int), []
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite acquisition score')
    indices = np.argsort(-scores, kind='stable')[:count]
    return indices, scores[indices].tolist()


def random_acquisition(learner, unlabeled_pool, n_queries, **kwargs):
    n = _count(unlabeled_pool, n_queries)
    indices = learner.rng.choice(len(unlabeled_pool), n, replace=False)
    return indices, np.ones(n).tolist()


def _pred(learner, pool, n_mc_samples):
    return learner.predict_uncertainty_batch(np.asarray([x['features'] for x in pool]), n_mc_samples)


def uncertainty_acquisition(learner, unlabeled_pool, n_queries, n_mc_samples=20, **kwargs):
    count = _count(unlabeled_pool, n_queries)
    if count == 0:
        return _top([], 0)
    _, std = _pred(learner, unlabeled_pool, n_mc_samples)
    return _top(std, count)


def boundary_acquisition(learner, unlabeled_pool, n_queries, threshold=0.99,
                         n_mc_samples=20, beta=1.0, **kwargs):
    count = _count(unlabeled_pool, n_queries)
    if count == 0:
        return _top([], 0)
    mean, std = _pred(learner, unlabeled_pool, n_mc_samples)
    # Bounded denominator avoids one near-threshold point dominating by 1e6.
    scores = std / (np.abs(mean-threshold)+0.02)**beta
    return _top(scores, count)


def margin_acquisition(learner, unlabeled_pool, n_queries, threshold=0.99, **kwargs):
    count = _count(unlabeled_pool, n_queries)
    if count == 0:
        return _top([], 0)
    mean = learner.predict_chi_batch(np.asarray([x['features'] for x in unlabeled_pool]))
    return _top(-np.abs(mean-threshold), count)


def hybrid_acquisition(learner, unlabeled_pool, n_queries, threshold=0.99,
                       n_mc_samples=20, uncertainty_weight=0.5, **kwargs):
    count = _count(unlabeled_pool, n_queries)
    if count == 0:
        return _top([], 0)
    if not 0 <= uncertainty_weight <= 1:
        raise ValueError('uncertainty_weight must be in [0,1]')
    mean, std = _pred(learner, unlabeled_pool, n_mc_samples)
    dist = np.abs(mean-threshold)
    u = (std-std.min())/(np.ptp(std)+1e-12)
    b = (dist-dist.min())/(np.ptp(dist)+1e-12)
    return _top(uncertainty_weight*u+(1-uncertainty_weight)*(1-b), count)


ACQUISITION_STRATEGIES = dict(random=random_acquisition, uncertainty=uncertainty_acquisition,
                            boundary=boundary_acquisition, margin=margin_acquisition,
                            hybrid=hybrid_acquisition)


def get_acquisition_strategy(name):
    if name not in ACQUISITION_STRATEGIES:
        raise ValueError(f'Unknown strategy {name}; choose {list(ACQUISITION_STRATEGIES)}')
    return ACQUISITION_STRATEGIES[name]
