"""Run paired G/W two-qutrit pilots with shared audited oracle caches.

Defaults are a small pilot, not a statistically powered final experiment.
Override budgets through al4qed.train for a larger run.
"""
import argparse
from pathlib import Path
import os


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',required=True)
    cli.add_argument('--cache', help='Reuse validated targets after an interrupted run; never overwrite results')
    cli.add_argument('--backend',choices=['local','external'],default='external')
    cli.add_argument('--models',nargs='+',choices=['nn','rf'],default=['nn','rf'])
    cli.add_argument('--vertices',type=int,default=30)
    cli.add_argument('--iterations',type=int,default=3)
    cli.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    args=cli.parse_args()
    if len(args.models)!=len(set(args.models)):cli.error('models must be unique')
    root=Path(args.output)
    if any((root/model/'config.json').exists() for model in args.models):
        raise FileExistsError('Choose a fresh output root; existing results are not overwritten')
    # Initialize Julia before train/network imports torch (shared native runtime).
    if args.backend == 'external':
        from .ohst_bridge import _get_runtime
        _get_runtime()
    from .train import parser, run_experiment
    if 'nn' in args.models:
        import torch
        torch.set_num_threads(1)
    for model in args.models:
        argv=['--mode','compare','--model',model,'--backend',args.backend,
              '--dims','3','3','--distribution','ghz_w_3x3',
              '--initial','12','--pool','80','--cycles','2','--queries','6',
              '--test','40','--val','12','--epochs','80','--patience','20',
              '--oracle-N',str(args.vertices),'--oracle-iters',str(args.iterations),
              '--output',str(root/model),'--cache',str(Path(args.cache) if args.cache else root/'oracle_cache'),
              '--seeds',*[str(s) for s in args.seeds]]
        if args.backend=='external':
            from .ohst_bridge import BRIDGE_VERSION
            argv+=['--external','al4qed.ohst_bridge:query','--backend-version',
                   BRIDGE_VERSION+'-'+os.environ.get('OHST_SOLVER','SCS').upper()]
        run_experiment(parser().parse_args(argv))


if __name__=='__main__':main()
