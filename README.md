# AL4QED

Adaptive learning experiments for quantum entanglement detection and Ohst-style
separability certification.

## Project layout

- `src/al4qed/` - reusable oracle and Ohst reproduction code.
- `experiments/` - standalone active-learning and benchmark scripts.
- `figures/` - generated plots and benchmark figures.
- `papers/` - reference papers used for the experiments.
- `patches/` - saved patch files from prior development.

## Main modules

- `al4qed.ohst_reproduction` reproduces the Ohst et al. hierarchy experiments,
  including BSEP, FBSEP, witness extraction, and adaptive refinement.
- `al4qed.oracle` exposes a reusable `EntanglementOracle` wrapper for two-qubit,
  three-qubit, and bipartite 2x4 classification tasks.

## Example

```python
from al4qed import EntanglementOracle
from al4qed.ohst_reproduction import ghz_state

oracle = EntanglementOracle(vertices=120, epsilon=5e-4)
result = oracle.classify_three_qubit(ghz_state())
print(result.as_dict())
```

## Environment

The numerical experiments depend on:

- `numpy`
- `scipy`
- `cvxpy`
- `matplotlib`
- `scikit-learn`
- `mosek`

Some scripts require a valid MOSEK license because the SDP solver is used for
high-accuracy certification.
