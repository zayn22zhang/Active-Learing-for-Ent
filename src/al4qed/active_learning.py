"""Reproducible regression learner, frozen validation and train-only preprocessing."""
import copy
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    from .dataset import extract_features_advanced
    from .acquisition import get_acquisition_strategy
except ImportError:
    from dataset import extract_features_advanced
    from acquisition import get_acquisition_strategy


class ActiveLearner:
    def __init__(self, oracle, model, input_dim=None, device='cpu', lr=1e-3,
                 batch_size=32, n_epochs=100, val_split=0.2, seed=42,
                 patience=15, validation_data=None):
        if batch_size < 1 or n_epochs < 1 or patience < 1 or not 0 <= val_split < 1:
            raise ValueError('Invalid training settings')
        self.oracle, self.model = oracle, model
        self.input_dim, self.device = input_dim, device
        self.lr, self.batch_size, self.n_epochs = lr, batch_size, n_epochs
        self.val_split, self.seed, self.patience = val_split, seed, patience
        self.rng = np.random.default_rng(seed)
        self.is_neural = hasattr(model, 'state_dict')
        if self.is_neural:
            self.model.to(device)
            self.initial_weights = copy.deepcopy(model.state_dict())
        else:
            self.template = clone(model)
        self.validation_data = validation_data
        self.validation_indices = None
        self.scaler = StandardScaler()
        self.X_labeled, self.y_labeled, self.pool = [], [], []
        self.query_history, self.training_history = [], []
        self.total_queries = 0
        self.pool_counter = 0
        self.is_scaler_fitted = False
        self.trained_count = 0

    def add_labeled(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if len(X) != len(y) or not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError('Invalid labels/features')
        if np.any((y < 0) | (y > 1)):
            raise ValueError('Visibility targets must be in [0,1]')
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        if X.shape[1] != self.input_dim:
            raise ValueError('Feature dimension mismatch')
        self.X_labeled.extend(X)
        self.y_labeled.extend(y.tolist())
        # Do not refit the scaler during prediction. Model and scaler form a pair.

    def add_to_pool(self, states, dims_list=None):
        dims_list = [(3,3)]*len(states) if dims_list is None else dims_list
        if len(states) != len(dims_list):
            raise ValueError('states and dims_list lengths differ')
        for rho, dims in zip(states, dims_list):
            features = extract_features_advanced(rho, dims)
            if self.input_dim is not None and len(features) != self.input_dim:
                raise ValueError('Cannot mix feature dimensions in one model')
            self.pool.append(dict(features=features, rho=rho, dA=dims[0], dB=dims[1],
                                  sample_id=f'pool-{self.pool_counter}'))
            self.pool_counter += 1

    def _prepare_data(self):
        X, y = np.asarray(self.X_labeled, dtype=np.float32), np.asarray(self.y_labeled, dtype=np.float32)
        if len(y) < 2:
            raise ValueError('Need at least two labeled samples')
        if self.validation_data is None:
            if self.validation_indices is None:
                n = max(1, int(len(y)*self.val_split)) if self.val_split else 0
                self.validation_indices = np.random.default_rng(self.seed).permutation(len(y))[:n]
            mask = np.ones(len(y), dtype=bool)
            mask[self.validation_indices] = False
            Xv, yv = X[~mask], y[~mask]
            X, y = X[mask], y[mask]
        else:
            Xv, yv = self.validation_data
            Xv, yv = np.asarray(Xv, dtype=np.float32), np.asarray(yv, dtype=np.float32)
        self.scaler.fit(X)
        self.is_scaler_fitted = True
        return self.scaler.transform(X), y, self.scaler.transform(Xv) if len(Xv) else None, yv

    def train_surrogate(self, verbose=False):
        X, y, Xv, yv = self._prepare_data()
        if not self.is_neural:
            self.model = clone(self.template)
            self.model.fit(X, y)
            log = dict(train_losses=[mean_squared_error(y, self.model.predict(X))], val_losses=[])
        else:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            # Cold restart avoids changing coordinates underneath warm-started weights.
            self.model.load_state_dict(self.initial_weights)
            generator = torch.Generator().manual_seed(self.seed)
            dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y))
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=generator)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
            best, best_loss, stale = None, float('inf'), 0
            log = dict(train_losses=[], val_losses=[])
            for epoch in range(self.n_epochs):
                self.model.train()
                total = 0.0
                for bx, by in loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(self.model(bx), by)
                    loss.backward()
                    optimizer.step()
                    total += loss.item()*len(by)
                log['train_losses'].append(total/len(y))
                self.model.eval()
                with torch.no_grad():
                    val_loss = (torch.nn.functional.mse_loss(
                        self.model(torch.as_tensor(Xv, dtype=torch.float32, device=self.device)),
                        torch.as_tensor(yv, dtype=torch.float32, device=self.device)).item()
                        if Xv is not None else total/len(y))
                log['val_losses'].append(val_loss)
                if val_loss < best_loss - 1e-8:
                    best_loss, stale = val_loss, 0
                    best = copy.deepcopy(self.model.state_dict())
                else:
                    stale += 1
                if stale >= self.patience and Xv is not None:
                    break
            self.model.load_state_dict(best)
            self.model.eval()
        self.trained_count = len(self.y_labeled)
        self.training_history.append(log)
        if verbose:
            print(f'Trained on {len(y)} samples; validation {len(yv)}; epochs {len(log["train_losses"])}')
        return log

    def _scaled(self, features):
        if not self.is_scaler_fitted or self.trained_count == 0:
            raise RuntimeError('Train the surrogate before predicting')
        return self.scaler.transform(np.atleast_2d(features)).astype(np.float32)

    def predict_chi_batch(self, features_batch, batch_size=1024):
        X = self._scaled(features_batch)
        if not self.is_neural:
            return np.clip(self.model.predict(X), 0, 1)
        import torch
        self.model.eval()
        with torch.no_grad():
            return np.concatenate([self.model(torch.as_tensor(X[i:i+batch_size], device=self.device)).cpu().numpy()
                                   for i in range(0, len(X), batch_size)])

    def predict_chi(self, features):
        return float(self.predict_chi_batch(features)[0])

    def predict_uncertainty_batch(self, features_batch, n_mc_samples=20, batch_size=512):
        if n_mc_samples < 2:
            raise ValueError('At least two uncertainty samples required')
        X = self._scaled(features_batch)
        if not self.is_neural:
            if not hasattr(self.model, 'estimators_'):
                raise ValueError('Uncertainty acquisition requires neural dropout or a random forest')
            predictions = np.stack([tree.predict(X) for tree in self.model.estimators_])
            return predictions.mean(0), predictions.std(0)
        import torch
        try:
            from .network import dropout_only
        except ImportError:
            from network import dropout_only
        means, stds = [], []
        # Isolate acquisition randomness from subsequent training and random selection.
        devices = [torch.device(self.device).index or 0] if str(self.device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices), dropout_only(self.model), torch.no_grad():
            torch.manual_seed(self.seed+len(self.training_history))
            for i in range(0, len(X), batch_size):
                bx = torch.as_tensor(X[i:i+batch_size], device=self.device)
                predictions = torch.stack([self.model(bx) for _ in range(n_mc_samples)])
                means.append(predictions.mean(0).cpu().numpy())
                stds.append(predictions.std(0, unbiased=False).cpu().numpy())
        return np.concatenate(means), np.concatenate(stds)

    def predict_uncertainty(self, features, n_mc_samples=20):
        mean, std = self.predict_uncertainty_batch(np.atleast_2d(features), n_mc_samples)
        return float(mean[0]), float(std[0])

    def query_uncertain(self, n_queries=5, n_mc_samples=20, strategy='uncertainty', **kwargs):
        if not self.pool:
            return []
        indices, scores = get_acquisition_strategy(strategy)(
            self, self.pool, n_queries, n_mc_samples=n_mc_samples, **kwargs)
        # Preserve score/index associations and acquisition ranking; delete by object ID.
        selected = [(self.pool[int(i)], float(s)) for i,s in zip(indices, scores)]
        results = []
        for item, score in selected:
            result = self.oracle.query(item['rho'], item['dA'], item['dB'])
            self.add_labeled(item['features'], result['chi'])
            self.total_queries += 1
            self.query_history.append(dict(cycle=len(self.training_history), sample_id=item['sample_id'],
                                           chi=result['chi'], strategy=strategy, score=score,
                                           cache_hit=result.get('cache_hit', False),
                                           oracle_seconds=result.get('seconds', 0)))
            self.pool = [p for p in self.pool if p is not item]
            results.append(result)
        return results

    def active_learning_cycle(self, n_cycles=10, queries_per_cycle=5, strategy='boundary', verbose=True, **kwargs):
        self.train_surrogate(verbose)
        for _ in range(n_cycles):
            if not self.pool:
                break
            self.query_uncertain(queries_per_cycle, strategy=strategy, **kwargs)
            self.train_surrogate(verbose)  # Includes the last queried batch.
        return self.X_labeled, self.y_labeled

    def evaluate_on_test_set(self, test_features, test_labels):
        pred = self.predict_chi_batch(test_features)
        return pred, mean_squared_error(test_labels, pred), mean_absolute_error(test_labels, pred), r2_score(test_labels, pred)

    def get_scaler(self):
        return self.scaler
