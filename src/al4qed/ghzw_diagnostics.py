"""Numerical audit of the explicitly defined two-qutrit G/W mixture.

PPT classification is necessary-only. This audit never assigns PPT states
an entangled label merely because the polytope lower estimate is below one.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .states import ghz_w_3x3
from .oracle import AdaptivePolytopeOracle, ppt_visibility


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--backend',choices=['local','external'],default='external')
    p.add_argument('--vertices',type=int,default=60)
    p.add_argument('--iterations',type=int,default=5)
    args=p.parse_args()
    dest=Path(args.output)
    if dest.exists(): raise FileExistsError(dest)
    extra={}
    if args.backend=='external':
        import os
        from .ohst_bridge import BRIDGE_VERSION
        extra=dict(external='al4qed.ohst_bridge:query',
                   backend_version=BRIDGE_VERSION+'-'+os.environ.get('OHST_SOLVER','SCS').upper())
    oracle=AdaptivePolytopeOracle(backend=args.backend,N=args.vertices,max_iter=args.iterations,**extra)
    records=[]
    for pg,pw in [(0,0),(1,0),(0,1),(.1,.1),(.3,.3),(.6,.2)]:
        r=oracle.query(ghz_w_3x3(pg,pw),3,3)
        r['label']=r['label'].name
        records.append(dict(p_g=pg,p_w=pw,**r))
        print(pg,pw,r['chi'],r['chi_upper_ppt'],flush=True)
    ppt=npt=0
    for pg in np.linspace(0,1,51):
        for pw in np.linspace(0,1-pg,51):
            _,m=ppt_visibility(ghz_w_3x3(float(pg),float(pw)),(3,3))
            if m>=-1e-8:ppt+=1
            else:npt+=1
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(dict(config=oracle.config(),records=records,
        grid=dict(ppt=ppt,npt=npt,note='Diagnostic grid is not uniform-area sampling; PPT does not imply SEP.')),
        indent=2,allow_nan=False))


if __name__=='__main__':main()
