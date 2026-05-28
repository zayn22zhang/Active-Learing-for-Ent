# Active-Learning-for-Ent

Neural-network active learning for quantum entanglement detection, using a
reusable Ohst-style oracle.

## Main Files

- `oracle.py`: reusable entanglement oracle.
  - 2 qubits: exact PPT separability oracle.
  - 3 qubits: Ohst-style BSEP/GME hierarchy oracle.
  - 2 x 4: bipartite validation oracle.
- `ohst_reproduction.py`: frozen Ohst et al. 2024 hierarchy reproduction.
- `NN for Fei:DeepSemi-supervised`: Fei-style neural-network active learning
  script, now backed by `oracle.py`.
- `test.py`: same active-learning script with a `.py` extension for easier
  execution/import.

## Run

```bash
python test.py
```

The active-learning framework is unchanged in spirit: it samples GHZ-W-white-noise
mixed states, labels them with the oracle, trains an MLP classifier, and queries
new points by maximum entropy.

## Notes

The 3-qubit oracle is a strong numerical GME/BSEP oracle based on the finite
polytope hierarchy. Solver quality depends on MOSEK, vertex count, and epsilon.
