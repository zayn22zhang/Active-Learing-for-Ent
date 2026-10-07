"""Fetch the inspected upstream revision and required submodule (no API guesses)."""
import argparse
from pathlib import Path
import subprocess
try:
    from .ohst_bridge import UPSTREAM_COMMIT, KVANT_COMMIT
except ImportError:
    from ohst_bridge import UPSTREAM_COMMIT, KVANT_COMMIT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('destination',type=Path)
    p.add_argument('--install-julia-deps',action='store_true')
    args=p.parse_args()
    destination=args.destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError('Choose a new checkout directory')
    subprocess.run(['git','clone','https://gitlab.com/tqo/quantum-correlations.git',str(destination)],check=True)
    subprocess.run(['git','-C',str(destination),'checkout',UPSTREAM_COMMIT],check=True)
    # Upstream has an unrelated orphan gitlink at kvant; do not use --recursive.
    subprocess.run(['git','-C',str(destination),'submodule','update','--init','lib/kvant'],check=True)
    revision=subprocess.check_output(['git','-C',str(destination/'lib/kvant'),'rev-parse','HEAD'],text=True).strip()
    if revision!=KVANT_COMMIT:
        raise RuntimeError('Unexpected kvant version')
    if args.install_julia_deps:
        from juliacall import Main as jl
        jl.seval('using Pkg; Pkg.add(["Convex", "SCS", "Mosek", "MosekTools", "MathOptInterface", "Combinatorics", "Permutations"])')
    print(f'Checkout ready: {destination}')
    print('Set OHST_REPO to this directory before training.')


if __name__=='__main__':
    main()
