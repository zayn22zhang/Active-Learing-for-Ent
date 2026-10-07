#!/usr/bin/env python3
"""Paired, budget-matched active learning of numerical visibility targets.

Example (no SDP/PyTorch required):
 python -m al4qed.train --backend ppt --dims 2 2 --model rf --mode compare
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
try:
    from .oracle import AdaptivePolytopeOracle
    from .dataset import sample_state, extract_features_advanced, FEATURE_VERSION
    from .active_learning import ActiveLearner
    from .acquisition import ACQUISITION_STRATEGIES
except ImportError:
    from oracle import AdaptivePolytopeOracle
    from dataset import sample_state, extract_features_advanced, FEATURE_VERSION
    from active_learning import ActiveLearner
    from acquisition import ACQUISITION_STRATEGIES


def get_model(model_name, input_dim=None, seed=42, **kwargs):
    if model_name == 'nn':
        import torch
        try:
            from .network import ChiPredictor
        except ImportError:
            from network import ChiPredictor
        torch.manual_seed(seed)
        return ChiPredictor(input_dim, **kwargs)
    if model_name == 'rf':
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1, **kwargs)
    if model_name == 'svm':
        return SVR(C=10, epsilon=0.01, **kwargs)
    if model_name == 'linear':
        return Ridge(alpha=1.0, **kwargs)
    if model_name == 'xgb':
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, max_depth=3, random_state=seed, n_jobs=1, **kwargs)
    raise ValueError(f'Unsupported model {model_name}')


def _json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def _oracle(args, cache):
    return AdaptivePolytopeOracle(N=args.oracle_N, max_iter=args.oracle_iters,
                                 backend=args.backend, external=args.external,
                                 backend_version=args.backend_version,
                                 cache_dir=cache, seed=args.oracle_seed)


def _label(oracle, states, dims):
    return [oracle.query(rho, *dims) for rho in states]


def run_experiment(args):
    """Pool labels stay unavailable until selected; test never selects/early-stops."""
    if args.initial < 2 or args.test < 2 or args.val < 2 or args.pool < 0:
        raise ValueError('initial/test/val >= 2 and pool >= 0 required')
    if args.cycles < 0 or args.queries < 1 or args.epochs < 1:
        raise ValueError('cycles >=0, queries/epochs >=1 required')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Supply unique seeds')
    strategies = args.strategies if args.mode == 'compare' else [args.strategy]
    if args.mode == 'supervised':
        strategies = ['random']
    strategies = list(dict.fromkeys(strategies))
    if args.model not in ('nn','rf') and any(s not in ('random','margin') for s in strategies):
        raise ValueError('Uncertainty strategies require --model nn or rf; other models support random/margin')
    output = Path(args.output)
    if (output/'config.json').exists():
        raise FileExistsError(f'{output} already contains a run. Choose a fresh --output; caches can be reused.')
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else output/'oracle_cache'
    config = dict(vars(args), feature_version=FEATURE_VERSION,
                  target='min(1, backend visibility); lower estimate unless exact backend',
                  protocol='frozen val/test, paired initial/pool, cold refit each cycle')
    config['versions'] = {}
    for dependency in ['numpy','scipy','scikit-learn','torch','cvxpy']:
        try:
            config['versions'][dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            config['versions'][dependency] = None
    config['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob('*.py')}
    _json(output/'config.json', config)
    dims = tuple(args.dims)
    # One independently generated, frozen test set across all runs.
    test_rng = np.random.default_rng(args.test_seed)
    test_states = [sample_state(dims, test_rng, args.distribution) for _ in range(args.test)]
    Xtest = np.asarray([extract_features_advanced(rho, dims) for rho in test_states])
    evaluation_oracle = _oracle(args, cache)
    test_records = _label(evaluation_oracle, test_states, dims)
    ytest = np.asarray([r['chi'] for r in test_records])
    np.savez_compressed(output/'test_set.npz', rho=test_states, X=Xtest, y=ytest,
                        labels=[r['label'].name for r in test_records])
    rows, run_accounting = [], []
    for seed in args.seeds:
        # Spawn separate streams: changing pool size cannot change initial/validation states.
        streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
        initial = [sample_state(dims, streams[0], args.distribution) for _ in range(args.initial)]
        validation = [sample_state(dims, streams[1], args.distribution) for _ in range(args.val)]
        pool = [sample_state(dims, streams[2], args.distribution) for _ in range(args.pool)]
        Xinitial = np.asarray([extract_features_advanced(r, dims) for r in initial])
        Xval = np.asarray([extract_features_advanced(r, dims) for r in validation])
        setup_oracle = _oracle(args, cache)
        initial_records = _label(setup_oracle, initial, dims)
        val_records = _label(setup_oracle, validation, dims)
        yinitial = np.asarray([r['chi'] for r in initial_records])
        yval = np.asarray([r['chi'] for r in val_records])
        np.savez_compressed(output/f'splits_seed{seed}.npz', initial=initial, validation=validation,
                            pool=np.asarray(pool), X_initial=Xinitial, y_initial=yinitial, X_val=Xval, y_val=yval)
        for strategy in strategies:
            start = time.perf_counter()
            oracle = _oracle(args, cache)
            model = get_model(args.model, input_dim=Xtest.shape[1], seed=seed)
            learner = ActiveLearner(oracle, model, device=args.device, seed=seed,
                                    n_epochs=args.epochs, patience=args.patience,
                                    batch_size=args.batch_size, validation_data=(Xval,yval))
            learner.add_labeled(Xinitial, yinitial)
            learner.add_to_pool(pool, [dims]*len(pool))
            folder = output/f'{args.model}_{strategy}_seed{seed}'
            folder.mkdir()
            predictions = None
            for cycle in range(args.cycles+1):
                learner.train_surrogate()
                predictions, mse, mae, r2 = learner.evaluate_on_test_set(Xtest, ytest)
                errors = np.abs(predictions-ytest)
                near = np.abs(ytest-args.threshold) <= 0.05
                row = dict(seed=seed, model=args.model, strategy=strategy, cycle=cycle,
                           train_queries=args.initial+learner.total_queries,
                           validation_queries=args.val, test_queries=args.test,
                           total_logical_queries=args.initial+learner.total_queries+args.val+args.test,
                           mse=float(mse), mae=float(mae), rmse=float(np.sqrt(mse)),
                           r2=float(r2) if np.isfinite(r2) else None,
                           near_level_mae=float(errors[near].mean()) if near.any() else None,
                           near_level_count=int(near.sum()),
                           queried_backend_calls=oracle.backend_calls, queried_cache_hits=oracle.cache_hits,
                           seconds=time.perf_counter()-start,
                           queried_oracle_seconds=oracle.elapsed_seconds,
                           model_trained_on=learner.trained_count)
                rows.append(row)
                print(f'seed={seed} {strategy:11s} labels={row["train_queries"]:4d} MAE={mae:.5f} RMSE={row["rmse"]:.5f}', flush=True)
                _json(folder/'query_history.json', learner.query_history)
                # Persist an auditable checkpoint after every completed training round.
                np.savez_compressed(folder/'labeled.npz', X=learner.X_labeled, y=learner.y_labeled)
                if learner.is_neural:
                    import torch
                    torch.save(dict(model=learner.model.state_dict(), input_dim=Xtest.shape[1],
                                    feature_version=FEATURE_VERSION, seed=seed), folder/'model.pt')
                    joblib.dump(learner.scaler, folder/'scaler.joblib')
                else:
                    joblib.dump(dict(model=learner.model, scaler=learner.scaler,
                                     feature_version=FEATURE_VERSION), folder/'model.joblib')
                with (output/'learning_curves.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                if cycle == args.cycles or not learner.pool:
                    break
                learner.query_uncertain(args.queries, strategy=strategy, n_mc_samples=args.mc_samples,
                                        threshold=args.threshold)
            np.savez_compressed(folder/'predictions.npz', target=ytest, prediction=predictions)
            _json(folder/'training_history.json', learner.training_history)
            run_accounting.append(dict(seed=seed, strategy=strategy, initial_and_val_calls=setup_oracle.backend_calls,
                                       active_calls=oracle.backend_calls, active_cache_hits=oracle.cache_hits))
    summary = []
    for strategy in strategies:
        budgets = sorted({r['train_queries'] for r in rows if r['strategy']==strategy})
        for budget in budgets:
            values = [r for r in rows if r['strategy']==strategy and r['train_queries']==budget]
            summary.append(dict(strategy=strategy, train_queries=budget, seeds=len(values),
                                mae_mean=float(np.mean([r['mae'] for r in values])),
                                mae_std=float(np.std([r['mae'] for r in values], ddof=1)) if len(values)>1 else 0.0,
                                rmse_mean=float(np.mean([r['rmse'] for r in values]))))
    _json(output/'summary.json', summary)
    _json(output/'accounting.json', dict(test_backend_calls=evaluation_oracle.backend_calls,
                                        test_cache_hits=evaluation_oracle.cache_hits,
                                        runs=run_accounting,
                                        note='Setup costs shared across strategies; do not sum repeated initial_and_val_calls. Logical budgets include cached labels.'))
    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7,4.5))
        for strategy in strategies:
            points = [r for r in summary if r['strategy']==strategy]
            x = np.array([r['train_queries'] for r in points])
            mean = np.array([r['mae_mean'] for r in points])
            std = np.array([r['mae_std'] for r in points])
            ax.plot(x, mean, marker='o', label=strategy)
            ax.fill_between(x, np.maximum(0,mean-std), mean+std, alpha=.15)
        ax.set(xlabel='Training labels queried (validation/test budget reported separately)',
               ylabel='Held-out MAE against backend target',
               title=f'{args.model.upper()} | {args.backend} | {dims[0]} x {dims[1]} | {len(args.seeds)} seeds')
        ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(output/'learning_curve.png', dpi=180); plt.close(fig)
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['active','compare','supervised'], default='compare')
    p.add_argument('--model', choices=['nn','rf','svm','linear','xgb'], default='nn')
    p.add_argument('--strategy', choices=list(ACQUISITION_STRATEGIES), default='uncertainty')
    p.add_argument('--strategies', choices=list(ACQUISITION_STRATEGIES), nargs='+', default=['random','uncertainty','boundary'])
    p.add_argument('--backend', choices=['local','external','ppt'], default='local')
    p.add_argument('--external', help='Verified adapter module:function, NOT an assumed upstream API')
    p.add_argument('--backend-version', help='Upstream commit and wrapper version; required for external')
    p.add_argument('--dims', type=int, nargs=2, default=[3,3])
    p.add_argument('--distribution', choices=['mixed','ginibre','horodecki'], default='mixed')
    for flag, default in [('initial',30),('pool',500),('cycles',5),('queries',15),('epochs',100),
                          ('test',200),('val',40),('patience',15),('batch-size',32),('mc-samples',20),
                          ('oracle-N',100),('oracle-iters',10),('oracle-seed',914),('test-seed',812739)]:
        p.add_argument('--'+flag, type=int, default=default)
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--threshold', type=float, default=.99, help='Acquisition level only, not a certification threshold')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', default='results/experiment')
    p.add_argument('--cache', help='Optional reusable oracle cache directory')
    p.add_argument('--no-plot', action='store_true')
    return p


def main():
    run_experiment(parser().parse_args())


if __name__ == '__main__':
    main()
