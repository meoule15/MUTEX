"""Summarize reproduced failures and create contact sheets for inspection."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    args=ap.parse_args(); root=args.directory
    completed=json.loads((root/'complete.json').read_text())
    summaries=[]
    for name in completed['cases']:
        d=json.loads((root/f'{name}.json').read_text()); traces=d['traces']
        predicates=[]
        for i,p in enumerate(d['goal_predicates']):
            reached=[r['step'] for r in traces if r['predicates'][i]]
            predicates.append(dict(predicate=p,first_true=min(reached) if reached else None,
                                   last_true=max(reached) if reached else None,
                                   true_steps=len(reached),final=traces[-1]['predicates'][i]))
        objects=list(traces[0]['object_displacement'])
        summary=dict(name=name,case=d['case'],predicates=predicates,
            max_displacement={o:max(r['object_displacement'][o] for r in traces) for o in objects},
            grasp_steps={o:sum(r['grasp'][o] for r in traces) for o in objects},
            newly_achieved_other_goals=[int(t) for t,v in traces[0]['alternative_goals'].items()
                if not v and int(t)!=d['case']['task_id'] and any(r['alternative_goals'][t] for r in traces)],
            reproduced=d['reproduced'])
        summaries.append(summary)
        frames=np.load(root/f'{name}-frames.npz')
        indices=np.linspace(0,len(frames['steps'])-1,6,dtype=int)
        fig,axes=plt.subplots(1,6,figsize=(15,3))
        for ax,i in zip(axes,indices):
            ax.imshow(frames['frames'][i]); ax.set_title(f"Step {frames['steps'][i]}"); ax.axis('off')
        fig.suptitle(name + (' — success' if d['case']['success'] else ' — timeout'))
        fig.tight_layout(); fig.savefig(root/f'{name}.png',dpi=160); plt.close(fig)
    (root/'summary.json').write_text(json.dumps(summaries,indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__': main()
