# Two-qutrit G/W experiment

## State definition (read before interpreting results)

This is a **bipartite 3 x 3** system, not three qubits or three qutrits.
We explicitly define

- G = (|00> + |11> + |22>)/sqrt(3), the maximally entangled two-qutrit state;
- W = (|01> + |10>)/sqrt(2), a rank-two entangled state embedded in 3 x 3;
- rho = p_g |G><G| + p_w |W><W| + (1-p_g-p_w) I_9/9.

Weights satisfy p_g,p_w >= 0 and p_g+p_w <= 1. This is an incoherent mixture, not a superposition. The name G/W is shorthand defined here; it does not transfer the multipartite GHZ/W entanglement-class distinction to two parties.

`--distribution ghz_w_3x3` draws the three weights from Dirichlet(1,1,1), uniformly in area on their simplex. It retains the entire mixed-state family without discarding PPT points. The mixture is only a two-parameter slice of the full state space, so success here would not establish general two-qutrit generalization or PPT-entanglement detection.

## Implementation

- `states.ghz_w_3x3`: constructs and checks mixture weights.
- `dataset.sample_state`: reproducible uniform-simplex sampling; rejects other dimensions.
- `train`: paired splits, frozen test set, saved state definition, test parameters and per-state target diagnostics; reports PPT/NPT subgroup errors, a training-mean baseline, and the cheap analytic PPT-upper prediction baseline. The latter is not a claim that PPT solves general two-qutrit separability.
- `active_learning`: logs the PPT eigenvalue, upper bound and target type alongside acquired samples.
- `ghzw_compare`: same budgets for RF and MLP, three paired seeds, random/uncertainty/boundary strategies, reusable target cache.
- `ghzw_diagnostics`: endpoint/interior numerical checks and a labelled diagnostic PPT grid. A PPT grid point is not automatically separable.

Current pilot defaults: 12 initial labels, two rounds of 6 queries (24 final training labels), 80 pool states, 12 validation states per seed, 40 independent test states shared across seeds, seeds 0/1/2, 80 MLP epochs with validation stopping. Oracle defaults: N=30, 3 adaptive iterations. This is a pilot budget, not a high-precision reference setting.

The maximum logical label demand per method/seed is 24+12+40=76. Physical backend calls across the study are reduced by sharing the test set, setup data and cached queried states. A cached call is not a newly timed oracle solve. No speedup claim is inferred from cached wall-clock comparisons.

## Reproduce

Install the project and optional dependencies as described in the main README; configure the pinned Ohst checkout first. From the repository root:

```bash
export OHST_REPO=/absolute/path/to/quantum-correlations
export OHST_SOLVER=SCS
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m al4qed.ghzw_compare \
  --output results/ghzw_official --backend external

python -m al4qed.ghzw_diagnostics --backend external \
  --vertices 60 --iterations 5 --output results/ghzw_diagnostics.json
```

The direct train interface supports larger experiments, e.g. `--distribution ghz_w_3x3 --dims 3 3`. The compare wrapper uses the same seed, splits, pool and budgets for both models. Existing completed run directories are not overwritten.

For a separately labelled local-backend diagnostic:

```bash
python -m al4qed.ghzw_compare --backend local --models rf \
  --output results/ghzw_local
```

`local` executes our CVXPY implementation; `external` executes the authors' Julia implementation. They are different target providers and their MAEs must not be compared as if measured against the same ground truth.

## Interpretation

The capped visibility of pure G is 1/4. The PPT upper bound at pure W is 2/11. The first is an analytic reference; the second is reported as an upper bound in the diagnostic. Interior targets are finite-budget inner-polytope estimates. The target/PPT gap bounds approximation tightness only when the exact mathematical bounds are valid; numerical tolerance must still be checked. A large gap does not establish PPT entanglement. Surrogate predictions do not inherit certification or lower-bound guarantees.

`boundary` still focuses on predicted target 0.99. If the empirical target range is far from that level, this strategy may be ineffective; the experiment should report this rather than tune the threshold on the test set.

See `GHZW_3X3_RESULTS.md` for the actual run outcomes and limitations.

Interrupted runs can reuse their validated cache with `--cache PATH` while writing to a **new** `--output`. This restarts model fitting deterministically; it does not pretend to resume an unrecorded optimizer state. Labelling progress is printed every ten states.

A physical interpretation check is essential: a mixture of a full-Schmidt-rank G state and an embedded rank-two W state need not contain a substantial region of PPT entanglement. This benchmark does not establish that such a region exists. In addition to active-versus-random comparisons, the report includes predicting the PPT upper bound directly; if this cheap baseline already outperforms the learned models against these numerical targets, the benefit of ML on this family has not been demonstrated.
