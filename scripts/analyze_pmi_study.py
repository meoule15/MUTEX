"""Validate pilot records and export summary tables and diagnostic figures."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    a = ap.parse_args(); root = a.directory
    results = json.loads((root/'results.json').read_text())
    assert json.loads((root/'complete.json').read_text())['rollouts'] == len(results) == 28
    manifest = json.loads((root/'manifest.json').read_text())
    frozen = json.loads((root/'frozen_settings.json').read_text())
    assert not set(manifest['validation_tasks']) & set(manifest['evaluation_tasks']+manifest['broader_tasks'])
    groups = defaultdict(list)
    traces = {}
    for result in results:
        rows = [json.loads(x) for x in (root/result['path']).open()]
        steps = defaultdict(list)
        for r in rows:
            assert all(np.isfinite(r[k]) for k in ['logp','reference_logp','lift','expected_lift','expected_lift_se','entropy'])
            assert abs(r['lift']-(r['logp']-r['reference_logp'])) < 1e-4
            assert r['expected_lift_se'] >= 0
            steps[r['step']].append(r)
        previous = 'vid' if result['mode']=='fixed_video' else 'gl'
        interval = rows[0]['selection_interval']
        for step, pair in steps.items():
            assert len(pair)==2 and {r['modality'] for r in pair}=={'gl','vid'}
            assert pair[0]['action']==pair[1]['action']
            assert pair[0]['acting_modality']==pair[1]['acting_modality']
            r = pair[0]
            if r['selection_switched']:
                assert r['selection_decision'] and (step-1)%interval==0
                assert r['acting_modality'] != previous
                if result['mode']=='expected_lift':
                    scores={x['modality']:x for x in pair}
                    new=scores[r['acting_modality']]; old=scores[previous]
                    z=r['selection_uncertainty']; margin=r['selection_margin']
                    assert new['expected_lift']-z*new['expected_lift_se'] > old['expected_lift']+z*old['expected_lift_se']+margin
            previous=r['acting_modality']
        assert len(steps)==result['steps']
        traces[result['path']] = rows
        if result['phase']!='validation':
            assert result['profile']==frozen['profile']
            groups[(result['phase'],result['mode'])].append(result)
    summary=[]
    for (phase, mode), rs in sorted(groups.items()):
        summary.append(dict(phase=phase, mode=mode, episodes=len(rs),
                            successes=sum(r['success'] for r in rs),
                            success_rate=np.mean([r['success'] for r in rs]).item(),
                            mean_steps=np.mean([r['steps'] for r in rs]).item(),
                            mean_switches=np.mean([r['switches'] for r in rs]).item(),
                            mean_elapsed_seconds=np.mean([r['elapsed_seconds'] for r in rs]).item(),
                            mean_seconds_per_step=np.mean([r['seconds_per_step'] for r in rs]).item()))
    with (root/'comparison.csv').open('w') as out:
        writer=csv.DictWriter(out, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    fig, axes=plt.subplots(1, 2, figsize=(12,4))
    modes=['fixed_language','fixed_video','random','expected_lift']
    for ax, phase in zip(axes,['evaluation','broader']):
        rs={r['mode']:r for r in summary if r['phase']==phase}
        ax.bar(modes,[rs[m]['success_rate'] for m in modes])
        ax.set_ylim(0,1.05); ax.set_title(phase+' pilot'); ax.set_ylabel('Success fraction')
        ax.tick_params(axis='x',rotation=20)
    fig.suptitle('Capped seen-task pilot; small samples, no significance claim')
    fig.tight_layout(); fig.savefig(root/'success_comparison.png',dpi=160); plt.close(fig)
    # Export one successful and one failed adaptive trace if available.
    adaptive=[r for r in results if r['mode']=='expected_lift' and r['phase']!='validation']
    for outcome in [1,0]:
        selected=next((r for r in adaptive if r['success']==outcome),None)
        if selected is None: continue
        rows=traces[selected['path']]
        fig,axes=plt.subplots(4,1,figsize=(11,9),sharex=True)
        for modality in ['gl','vid']:
            rs=[r for r in rows if r['modality']==modality]; x=[r['step'] for r in rs]
            for ax,field in zip(axes[:3],['lift','expected_lift','entropy']):
                ax.plot(x,[r[field] for r in rs],label=modality); ax.set_ylabel(field)
                ax.legend()
            axes[0].axhline(0,color='gray',linewidth=.5)
        rs=[r for r in rows if r['modality']=='gl']
        axes[3].step([r['step'] for r in rs],[int(r['acting_modality']=='vid') for r in rs],where='post')
        axes[3].set_yticks([0,1]); axes[3].set_yticklabels(['language','video']); axes[3].set_xlabel('Step')
        fig.suptitle(f"Task {selected['task_id']}, seed {selected['seed']}, success={outcome}; capped pilot")
        fig.tight_layout(); fig.savefig(root/f'trace-success{outcome}.png',dpi=160); plt.close(fig)
    references=[json.loads(x) for x in (root/'reference_comparison.jsonl').open()]
    matched=defaultdict(dict)
    for row in references:
        assert all(np.isfinite(row[k]) for k in ['conditioned_logp','reference_logp','lift'])
        matched[(row['task_id'],row['step'])][(row['modality'],row['reference'])]=row
    agreement=[]; task_scores=defaultdict(lambda:defaultdict(list))
    scatter=defaultdict(list)
    for (task,step), rs in matched.items():
        assert len(rs)==4
        assert all(r['action']==next(iter(rs.values()))['action'] for r in rs.values())
        for modality in ['gl','vid']:
            blank=rs[modality,'blank']; mix=rs[modality,'mixture']
            assert abs(blank['conditioned_logp']-mix['conditioned_logp'])<1e-5
            scatter[modality].append((blank['lift'],mix['lift']))
            for ref in ['blank','mixture']:
                task_scores[task][modality,ref].append(rs[modality,ref]['lift'])
        blank_diff=rs['gl','blank']['lift']-rs['vid','blank']['lift']
        mix_diff=rs['gl','mixture']['lift']-rs['vid','mixture']['lift']
        agreement.append(np.sign(blank_diff)==np.sign(mix_diff))
    per_task=[]
    for task, scores in task_scores.items():
        means={m+'_'+ref:float(np.mean(v)) for (m,ref),v in scores.items()}
        means.update(task_id=task, blank_prefers_language=means['gl_blank']>means['vid_blank'],
                     mixture_prefers_language=means['gl_mixture']>means['vid_mixture'])
        per_task.append(means)
    ref_summary=dict(matched_steps=len(matched), step_ranking_agreement=float(np.mean(agreement)),
                     task_ranking_agreement=float(np.mean([r['blank_prefers_language']==r['mixture_prefers_language'] for r in per_task])),
                     per_task=per_task, reference_pool=manifest['reference_tasks'])
    fig,ax=plt.subplots(figsize=(6,5))
    for modality,values in scatter.items():
        x,y=np.array(values).T; ax.scatter(x,y,s=5,alpha=.3,label=modality)
    ax.set_xlabel('Blank-reference action lift'); ax.set_ylabel('Mixture-reference action lift')
    ax.legend(); fig.tight_layout(); fig.savefig(root/'reference_comparison.png',dpi=160); plt.close(fig)
    combined=[]
    for mode in modes:
        rs=[r for r in results if r['phase']!='validation' and r['mode']==mode]
        combined.append(dict(mode=mode, successes=sum(r['success'] for r in rs), episodes=len(rs),
                             mean_switches=float(np.mean([r['switches'] for r in rs])),
                             mean_seconds_per_step=float(np.mean([r['seconds_per_step'] for r in rs]))))
    report=dict(validation=[r for r in results if r['phase']=='validation'], frozen=frozen,
                combined_comparison=combined,
                comparison=summary, references=ref_summary,
                limitations=['Seen training tasks only','300-step cap by default versus upstream 600',
                             'Two validation tasks and two joint tuning profiles',
                             'One initial state and one specification variant',
                             'Strategies consume action-sampling randomness differently; paired seeds do not ensure matched action noise',
                             'Reference pool contains only two tasks',
                             'Runtime includes diagnostics for all strategies; not production latency',
                             'No statistical evidence of generalization or superiority'])
    (root/'report.json').write_text(json.dumps(report,indent=2))
    lines=['PMI selection pilot', '', 'Frozen settings: '+json.dumps(frozen['settings']), '',
           'Evaluation and broader runs combined (validation excluded):']
    for row in combined:
        lines.append(f"{row['mode']}: {row['successes']:g}/{row['episodes']} successes; "
                     f"{row['mean_switches']:.2f} switches/episode; "
                     f"{row['mean_seconds_per_step']:.3f} seconds/step including diagnostics")
    lines += ['', f"Blank/mixture modality ranking agreement: {ref_summary['step_ranking_agreement']:.1%} of steps, "
               f"{ref_summary['task_ranking_agreement']:.1%} of task means.",
               '', 'Limits:'] + report['limitations']
    (root/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
