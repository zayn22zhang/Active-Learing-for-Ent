"""Budget-matched comparison; shares the tested train.py experiment engine."""
try:
    from .train import parser, run_experiment
except ImportError:
    from train import parser, run_experiment

if __name__ == '__main__':
    args = parser().parse_args()
    args.mode = 'compare'
    run_experiment(args)
