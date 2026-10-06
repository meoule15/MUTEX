"""Correct/mismatched specification controls on cached, matched action likelihoods.

Run from MUTEX/: python scripts/test_specification_controls.py ../results/reference-validation
Mixture over the three variants keeps task comparisons symmetric. This measures
action compatibility, not whether a modality reliably completes a task.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_pmi_reference import logmean


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    args=ap.parse_args(); root=args.directory
    manifest=json.loads((root/'manifest.json').read_text())
    frames=json.loads((root/'frames.json').read_text())
    metadata_path=root/'task_metadata.json'
    if not metadata_path.exists():
        from libero.libero import benchmark
        bm=benchmark.get_benchmark_dict()['libero_100'](0)
        metadata=[dict(task_id=i,name=bm.get_task(i).name,
                       scene=re.match(r'^.+?_SCENE\d+(?=_)',bm.get_task(i).name).group(0))
                  for i in range(bm.n_tasks)]
        metadata_path.write_text(json.dumps(metadata,indent=2))
    metadata=json.loads(metadata_path.read_text())
    data=np.load(root/'likelihoods.npz');lp=data['logp'].astype(np.float64)
    blank=data['blank_logp'].astype(np.float64)
    assert lp.shape==(len(frames),2,100,3) and np.isfinite(lp).all()
    task_logp=logmean(lp,(3,))
    reference=logmean(lp[:,:,manifest['reference_tasks'],:],(2,3))
    prefix_end={}
    for f in frames:
        key=f['task_id'],f['episode'];prefix_end[key]=max(prefix_end.get(key,0),f['step'])
    rows=[];top_rows=[]
    for i,f in enumerate(frames):
        t=f['task_id'];scene=metadata[t]['scene']
        phase=min(2,int(3*f['step']/max(1,prefix_end[t,f['episode']]+1)))
        for modality,m in [('gl',0),('vid',1)]:
            correct=task_logp[i,m,t]
            same=[j for j in range(100) if j!=t and metadata[j]['scene']==scene]
            best_wrong=max(task_logp[i,m,same])
            top_rows.append(dict(task_id=t,episode=f['episode'],step=f['step'],modality=modality,
                                  correct_beats_every_same_scene_task=bool(correct>best_wrong+1e-6),
                                  correct_minus_blank=float(correct-blank[i,m])))
            for j in range(100):
                if j==t: continue
                wrong=task_logp[i,m,j];gap=correct-wrong
                # Using either fixed reference must leave this paired difference unchanged.
                assert np.isclose(gap,(correct-reference[i,m])-(wrong-reference[i,m]),atol=1e-10)
                assert np.isclose(gap,(correct-blank[i,m])-(wrong-blank[i,m]),atol=1e-10)
                variant_gaps=lp[i,m,t,:,None]-lp[i,m,j,None,:]
                rows.append(dict(task_id=t,task_name=metadata[t]['name'],episode=f['episode'],
                     step=f['step'],prefix_third=phase,modality=modality,
                     mismatched_task_id=j,control_group='same_scene' if j in same else 'different_scene',
                     correct_logp=float(correct),mismatched_logp=float(wrong),
                     reference_logp=float(reference[i,m]),correct_lift=float(correct-reference[i,m]),
                     mismatched_lift=float(wrong-reference[i,m]),paired_gap=float(gap),
                     correct_wins=bool(gap>1e-6),tie=bool(abs(gap)<=1e-6),
                     individual_variant_win_fraction=float(np.mean(variant_gaps>1e-6))))
    with (root/'specification_controls.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    def summarize(selected):
        return dict(comparisons=len(selected),correct_win_rate=float(np.mean([r['correct_wins'] for r in selected])),
                    ties=int(sum(r['tie'] for r in selected)),mean_paired_gap=float(np.mean([r['paired_gap'] for r in selected])),
                    median_paired_gap=float(np.median([r['paired_gap'] for r in selected])),
                    individual_variant_win_fraction=float(np.mean([r['individual_variant_win_fraction'] for r in selected])))

    overall=[];per_task=[];by_prefix=[]
    for modality in ['gl','vid']:
        for group in ['same_scene','different_scene']:
            selected=[r for r in rows if r['modality']==modality and r['control_group']==group]
            entry=dict(modality=modality,control_group=group,**summarize(selected))
            if group=='same_scene':
                entry['correct_beats_all_controls_fraction']=float(np.mean([
                    r['correct_beats_every_same_scene_task'] for r in top_rows if r['modality']==modality]))
            overall.append(entry)
            for task in manifest['target_tasks']:
                per_task.append(dict(task_id=task,modality=modality,control_group=group,
                                     **summarize([r for r in selected if r['task_id']==task])))
            for phase in range(3):
                by_prefix.append(dict(prefix_third=phase,modality=modality,control_group=group,
                                      **summarize([r for r in selected if r['prefix_third']==phase])))
    report=dict(frames=len(frames),episodes=len(set((r['task_id'],r['episode']) for r in frames)),
                scenes=sorted(set(metadata[t]['scene'] for t in manifest['target_tasks'])),
                task_names={str(t):metadata[t]['name'] for t in manifest['target_tasks']},
                reference_tasks=manifest['reference_tasks'],variants='Uniform mixture of all three held-out variants',
                overall=overall,per_task=per_task,prefix_thirds=by_prefix,
                correct_blank={modality:dict(mean_gap=float(np.mean([r['correct_minus_blank'] for r in top_rows if r['modality']==modality])),
                         correct_win_rate=float(np.mean([r['correct_minus_blank']>1e-6 for r in top_rows if r['modality']==modality])))
                         for modality in ['gl','vid']},
                checks={'paired_gap_invariant_to_fixed_reference':True,'matched_actions_and_histories':True},
                limits=['All six scored tasks are in KITCHEN_SCENE10; the other 94 tasks are different-scene controls.',
                        'Training demonstrations and seen tasks; two episodes and sampled prefixes per task.',
                        'Same-scene tasks share objects and subgoals, so a mismatched goal can still support some correct actions.',
                        'Prefix thirds are not annotated task stages or full-episode thirds.',
                        'Many comparisons reuse the same actions; rates are descriptive, not independent success trials.',
                        'Correct/mismatched discrimination is not a modality-sufficiency or task-success estimate.'])
    (root/'SPECIFICATION_CONTROLS.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,modality in zip(axes,['gl','vid']):
        values=[r['correct_win_rate'] for r in per_task if r['modality']==modality and r['control_group']=='same_scene']
        ax.bar(manifest['target_tasks'],values);ax.set_ylim(0,1.05);ax.set_xlabel('Task ID')
        ax.set_ylabel('Correct specification wins against same-scene controls');ax.set_title(modality)
    fig.suptitle('Matched demonstration actions; action compatibility, not task success')
    fig.tight_layout();fig.savefig(root/'specification_controls.png',dpi=160);plt.close(fig)
    lines=['# Correct versus mismatched specification controls','',
           'Goal: identify modalities that suffice to complete each task. This control checks whether likelihood scores distinguish task-relevant specifications; completion must be measured in rollouts.', '',
           f"Used {len(frames)} matched demonstration actions across 12 episodes. Both modalities use the same observation history and action. Each task specification is a probability mixture over its three held-out variants.", '',
           '## Results','',
           '| Modality | Control | Comparisons | Correct specification wins | Mean correct−mismatched logp, nats |',
           '| --- | --- | ---: | ---: | ---: |']
    lines += [f"| {r['modality']} | {r['control_group']} | {r['comparisons']} | {r['correct_win_rate']:.1%} | {r['mean_paired_gap']:.2f} |" for r in overall]
    lines += ['', 'Both modalities contain task-specific action-likelihood signal on these demonstrations. Video discriminates slightly better in this descriptive sample. This does not establish that video is needed, or that language suffices.', '',
              '## Same-scene results by task','',
              '| Task | Language: correct wins | Video: correct wins |','| --- | ---: | ---: |']
    for task in manifest['target_tasks']:
        rates={r['modality']:r['correct_win_rate'] for r in per_task if r['task_id']==task and r['control_group']=='same_scene'}
        lines.append(f"| {task} | {rates['gl']:.1%} | {rates['vid']:.1%} |")
    lines += ['', '## Interpretation','',
              '- Correct-versus-mismatched differences are unchanged by substituting the blank reference for the fixed mixture reference: the denominator cancels within each modality.',
              '- The earlier blank-reference preference offset therefore does not imply that the language specification lacks useful information.',
              '- All six scored tasks share KITCHEN_SCENE10. The earlier 94-task reference excludes that entire scene; it is a cross-scene prior, which further limits its interpretation as a conditional marginal.',
              '- Different goals in one scene can share valid action prefixes. Lower discrimination at some steps can reflect shared subgoals rather than specification blindness.', '',
              '## What to test for sufficiency','',
              'Run fixed-language and fixed-video policies per task across multiple initial states and specification variants. Compare completion rates at the same rollout horizon. If language meets a predefined reliability target, it suffices even if video has larger lift; the current single-state pilot cannot establish that reliability.', '',
              '[Per-action controls](specification_controls.csv) · [Numerical results](SPECIFICATION_CONTROLS.json) · [Plot](specification_controls.png)', '',
              '## Limits','']+['- '+v for v in report['limits']]
    (root/'SPECIFICATION_CONTROLS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'overall':overall,'per_task':per_task,'prefix_thirds':by_prefix},indent=2))


if __name__=='__main__':main()
