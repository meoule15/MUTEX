"""Report paired task-completion rates; retain uncertainty about true sufficiency."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def interval(successes,n):
    z=1.959963984540054
    p=successes/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,center-half),min(1,center+half)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('--target',type=float,default=.9,help='Provisional observed completion target')
    args=ap.parse_args();root=args.directory
    if not 0<args.target<=1:ap.error('Target must be in (0,1]')
    assert (root/'complete.json').exists()
    manifest=json.loads((root/'manifest.json').read_text())
    results=json.loads((root/'episodes.json').read_text())
    summaries=[];paired=[];variants=[];initials=[]
    for task in manifest['tasks']:
        maps={m:{(r['initial_state_index'],r['spec_index']):r for r in results
                 if r['task_id']==task and r['modality']==m} for m in ['gl','vid']}
        assert maps['gl'].keys()==maps['vid'].keys()
        count=manifest['initial_states']*3
        assert len(maps['gl'])==count
        both=language=video=neither=0
        for key,l in maps['gl'].items():
            v=maps['vid'][key]
            assert l['initial_observation_sha256']==v['initial_observation_sha256']
            assert l['environment_seed']==v['environment_seed']
            for r in [l,v]:
                assert (1<=r['success_step']<=manifest['max_steps']) if r['success'] else r['success_step']==-1
            if l['success'] and v['success']:both+=1
            elif l['success']:language+=1
            elif v['success']:video+=1
            else:neither+=1
        paired.append(dict(task_id=task,both=both,language_only=language,video_only=video,neither=neither))
        for modality in ['gl','vid']:
            rows=list(maps[modality].values());wins=sum(r['success'] for r in rows)
            low,high=interval(wins,count)
            summaries.append(dict(task_id=task,task_name=manifest['task_names'][str(task)],modality=modality,
                successes=wins,episodes=count,completion_rate=wins/count,descriptive_wilson95_low=low,
                descriptive_wilson95_high=high,meets_observed_target=wins/count>=args.target,
                successful_mean_steps=float(np.mean([r['success_step'] for r in rows if r['success']])) if wins else None))
            for spec in [0,1,2]:
                selected=[r for r in rows if r['spec_index']==spec]
                variants.append(dict(task_id=task,modality=modality,spec_index=spec,
                                     successes=sum(r['success'] for r in selected),episodes=len(selected)))
            for state in sorted(set(r['initial_state_index'] for r in rows)):
                selected=[r for r in rows if r['initial_state_index']==state]
                initials.append(dict(task_id=task,modality=modality,initial_state_index=state,
                                     successes=sum(r['success'] for r in selected),episodes=len(selected)))
    with (root/'completion_rates.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    decisions=[]
    for task in manifest['tasks']:
        rows={r['modality']:r for r in summaries if r['task_id']==task}
        l=rows['gl']['meets_observed_target'];v=rows['vid']['meets_observed_target']
        status='both meet observed target' if l and v else 'language only meets observed target' if l else 'video only meets observed target' if v else 'neither meets observed target'
        decisions.append(dict(task_id=task,observed_grid_status=status,
                              reliability_established=False))
    report=dict(target=args.target,episodes=len(results),summaries=summaries,paired_outcomes=paired,
                variant_results=variants,initial_state_results=initials,grid_decisions=decisions,
                note='Meeting a point-estimate target on this grid does not certify true reliability or sufficiency.',
                limits=['Seen tasks and one frozen checkpoint.',
                        f"{manifest['initial_states']} initial states crossed with three held-out variants; one action-noise realization per case.",
                        f"Tasks tested: {manifest['tasks']}; task coverage is limited and is not a random sample of tasks.",
                        'Cases share initial states/variants; Wilson intervals are descriptive iid-binomial approximations.',
                        'A larger lift is not the sufficiency criterion; task completion is.',
                        'No claim that video is necessary if language fails on this small grid.'])
    (root/'report.json').write_text(json.dumps(report,indent=2))
    fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(manifest['tasks']))
    for modality,offset,label in [('gl',-.18,'Language'),('vid',.18,'Video')]:
        rows=[r for r in summaries if r['modality']==modality]
        rates=np.array([r['completion_rate'] for r in rows])
        errors=np.array([[r['completion_rate']-r['descriptive_wilson95_low'] for r in rows],
                         [r['descriptive_wilson95_high']-r['completion_rate'] for r in rows]])
        ax.bar(x+offset,rates,.36,yerr=errors,capsize=3,label=label)
    ax.axhline(args.target,color='gray',linestyle='--',label=f'{args.target:.0%} provisional target')
    ax.set_xticks(x);ax.set_xticklabels(manifest['tasks']);ax.set_xlabel('LIBERO task ID')
    ax.set_ylabel('Task completion fraction');ax.set_ylim(0,1.12);ax.legend()
    ax.set_title('Paired initial-state × specification grid; descriptive 95% intervals')
    fig.tight_layout();fig.savefig(root/'completion_rates.png',dpi=160);plt.close(fig)
    lines=['# Fixed-modality task-completion study','',
           f"Ran **{len(results)} episodes** at a **{manifest['max_steps']}-step limit**: {manifest['initial_states']} initial states × 3 specification variants × 2 modalities × {len(manifest['tasks'])} tasks.", '',
           'Language and video refer to the **goal specification**. Both conditions retain camera observations and robot state. Language and video were paired on initial observations, environment seeds, and common per-step action-sampling seeds. Neither policy switched specifications. No PMI diagnostics were used to control execution.', '',
           f"The provisional observed-completion target is **{args.target:.0%}**. This labels the tested grid only; it does not establish population reliability.", '',
           '| Task | Language | Video | Paired: language only / video only | Observed target status |',
           '| --- | ---: | ---: | ---: | --- |']
    if manifest.get('common_variant_action_noise'):
        lines[6:6]=['The environment seed and starting observations also match across specification variants. Action-sampling RNG is reset separately for each variant at every step, coupling mixture and Gaussian randomness across variants. Independent policy histories are retained.', '']
    for task in manifest['tasks']:
        rows={r['modality']:r for r in summaries if r['task_id']==task};p=next(p for p in paired if p['task_id']==task)
        status=next(d['observed_grid_status'] for d in decisions if d['task_id']==task)
        lines.append(f"| {task} | {rows['gl']['successes']}/{rows['gl']['episodes']} | {rows['vid']['successes']}/{rows['vid']['episodes']} | {p['language_only']} / {p['video_only']} | {status} |")
    lines += ['', '## Specification variants', '',
              f"Each cell lists successes for variants 0 / 1 / 2; each variant has {manifest['initial_states']} initial-state trials.", '',
              '| Task | Language | Video |', '| --- | --- | --- |']
    for task in manifest['tasks']:
        cells=[]
        for modality in ['gl','vid']:
            cells.append(' / '.join(str(r['successes']) for r in variants
                                  if r['task_id']==task and r['modality']==modality))
        lines.append(f'| {task} | {cells[0]} | {cells[1]} |')
    lines += ['', '## Task names','']+[f"- {t}: {manifest['task_names'][str(t)]}" for t in manifest['tasks']]
    lines += ['', '## Interpretation','',
              'Use completion rates per task to assess whether a modality may suffice. Inspect variant and initial-state breakdowns before attributing failures to modality: a result can depend on wording, the demonstration specification, or the initial arrangement. A point-estimate threshold is a screening result, not a reliability certificate.', '',
              '[Episode outcomes](episodes.json) · [Rates and intervals](completion_rates.csv) · [Plot](completion_rates.png) · [Full report and breakdowns](report.json)', '',
              '## Limits','']+['- '+v for v in report['limits']]
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'episodes':len(results),'target':args.target,'grid_decisions':decisions,'summaries':summaries},indent=2))


if __name__=='__main__':main()
