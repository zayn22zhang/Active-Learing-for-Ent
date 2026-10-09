"""Export compact, auditable results from a completed G/W comparison."""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np


def export(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    report=[]
    for folder in sorted(source.iterdir()):
        if not (folder/'summary.json').exists(): continue
        model=folder.name
        dest=destination/model; dest.mkdir(exist_ok=True)
        for name in ['config.json','summary.json','accounting.json','learning_curves.csv',
                     'target_diagnostics.json','test_targets.json','test_parameters.json']:
            if (folder/name).exists(): shutil.copy2(folder/name,dest/name)
        summaries=json.loads((folder/'summary.json').read_text())
        records=json.loads((folder/'test_targets.json').read_text())
        upper=np.array([r['chi_upper_ppt'] for r in records]); targets=np.array([r['chi'] for r in records])
        baseline=float(np.abs(upper-targets).mean())
        curves=list(csv.DictReader((folder/'learning_curves.csv').open()))
        budget=max(int(r['train_queries']) for r in curves)
        final=[r for r in curves if int(r['train_queries'])==budget]
        paired=[]
        for r in final:
            if r['strategy']=='random':continue
            ref=next(v for v in final if v['seed']==r['seed'] and v['strategy']=='random')
            paired.append(dict(seed=int(r['seed']),strategy=r['strategy'],
                               mae_difference_vs_random=float(r['mae'])-float(ref['mae'])))
        (dest/'paired_differences.json').write_text(json.dumps(paired,indent=2))
        for r in summaries:
            if r['train_queries']==budget:report.append(dict(model=model,**r,ppt_upper_baseline_mae=baseline))
        for run in sorted(folder.glob('*_seed*')):
            if not run.is_dir():continue
            dst=dest/run.name;dst.mkdir(exist_ok=True)
            for name in ['query_history.json','training_history.json']:
                if (run/name).exists():shutil.copy2(run/name,dst/name)
            if (run/'predictions.npz').exists():
                data=np.load(run/'predictions.npz')
                with (dst/'predictions.csv').open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(['test_index','target','prediction'])
                    w.writerows((i,float(y),float(p)) for i,(y,p) in enumerate(zip(data['target'],data['prediction'])))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4.5))
        for strategy in dict.fromkeys(r['strategy'] for r in summaries):
            points=[r for r in summaries if r['strategy']==strategy]
            x=[r['train_queries'] for r in points]
            y=np.array([r['mae_mean'] for r in points]);std=np.array([r['mae_std'] for r in points])
            ax.plot(x,y,'o-',label=strategy)
            ax.fill_between(x,np.maximum(0,y-std),y+std,alpha=.15)
        ax.axhline(baseline,color='black',linestyle='--',label='PPT upper prediction (not exact target)')
        ax.set(xlabel='Training labels',ylabel='Test MAE against numerical target',title=f'Two-qutrit G/W mixture: {model.upper()}')
        ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(dest/'learning_curve.svg');plt.close(fig)
    if not report:raise ValueError('No completed runs found')
    (destination/'final_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source');p.add_argument('destination')
    a=p.parse_args();export(a.source,a.destination)


if __name__=='__main__':main()
