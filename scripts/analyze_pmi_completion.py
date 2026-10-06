"""Evaluate frozen prospective PMI ranks against every paired branch outcome."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path)
    root=ap.parse_args().directory
    assert (root/'complete.json').exists()
    manifest=json.loads((root/'manifest.json').read_text());episodes=json.loads((root/'episodes.json').read_text())
    contexts=[];decisions=[]
    for file in sorted(root.glob('*-ranking.json')):
        rank=json.loads(file.read_text());task=rank['task_id'];state=rank['initial_state_index']
        outcomes={(r['modality'],r['spec_index']):r for r in episodes if r['task_id']==task and r['initial_state_index']==state}
        assert len(outcomes)==6 and rank['scored_before_rollouts']
        assert all(r['initial_observation_sha256']==rank['initial_observation_sha256'] for r in outcomes.values())
        for c in rank['candidates']:
            c['success']=outcomes[c['modality'],c['spec_index']]['success']
            c['success_step']=outcomes[c['modality'],c['spec_index']]['success_step']
        contexts.append(rank)
        for scope in ['all','video']:
            candidates=[c for c in rank['candidates'] if scope=='all' or c['modality']=='vid']
            uniform=sum(c['success'] for c in candidates)/len(candidates)
            for reference in ['blank','full_modality','full_shared']:
                ordered=sorted(candidates,key=lambda c:c['scores'][reference]['mean'],reverse=True)
                winner,runner=ordered[:2];gap=winner['scores'][reference]['mean']-runner['scores'][reference]['mean']
                combined=math.hypot(winner['scores'][reference]['se'],runner['scores'][reference]['se'])
                concordant=discordant=tied=0
                for successful in candidates:
                    if not successful['success']:continue
                    for failure in candidates:
                        if failure['success']:continue
                        diff=successful['scores'][reference]['mean']-failure['scores'][reference]['mean']
                        if abs(diff)<1e-8:tied+=1
                        elif diff>0:concordant+=1
                        else:discordant+=1
                decisions.append(dict(task_id=task,initial_state_index=state,scope=scope,reference=reference,
                    selected_modality=winner['modality'],selected_spec_index=winner['spec_index'],success=winner['success'],
                    uniform_success_probability=uniform,oracle_success=any(c['success'] for c in candidates),
                    informative=0<uniform<1,top_gap=gap,top_gap_combined_se=combined,clear_top_gap=gap>2*combined,
                    concordant_pairs=concordant,discordant_pairs=discordant,tied_pairs=tied))
    assert len(contexts)==len(manifest['tasks'])*manifest['initial_states']
    summaries=[]
    for scope in ['all','video']:
        for reference in ['blank','full_modality','full_shared']:
            rows=[d for d in decisions if d['scope']==scope and d['reference']==reference]
            clear=[r for r in rows if r['clear_top_gap']];info=[r for r in rows if r['informative']]
            good=sum(r['concordant_pairs'] for r in rows);bad=sum(r['discordant_pairs'] for r in rows);ties=sum(r['tied_pairs'] for r in rows)
            summaries.append(dict(scope=scope,reference=reference,contexts=len(rows),selected_successes=sum(r['success'] for r in rows),
                uniform_expected_successes=sum(r['uniform_success_probability'] for r in rows),oracle_successes=sum(r['oracle_success'] for r in rows),
                informative_contexts=len(info),informative_selected_successes=sum(r['success'] for r in info),
                clear_rankings=len(clear),clear_selected_successes=sum(r['success'] for r in clear),
                concordant_pairs=good,discordant_pairs=bad,tied_pairs=ties,
                pair_concordance=(good+.5*ties)/(good+bad+ties) if good+bad+ties else None))
    fixed={m:sum(next(c for c in r['candidates'] if c['modality']==m and c['spec_index']==0)['success'] for r in contexts) for m in ['gl','vid']}
    report=dict(protocol=manifest,contexts=contexts,decisions=decisions,summaries=summaries,fixed_variant0_successes=fixed,
                limits=['Four previously examined, seen training tasks; three initial states and one fresh action/environment seed.',
                        'Ranking observes only the initial observation, not future history or completion.',
                        '512 draws per candidate by default; sampling standard errors do not measure success uncertainty.',
                        'Uniform baseline is exact expected selection success from measured branches, not a sampled random-selector rollout.',
                        'Success/failure pairs share branches and contexts; concordance is descriptive, not independent-trial evidence.',
                        'References are zero-content intervention or uniform empirical mixtures, not an identified true conditional marginal.',
                        'This tests choosing one fixed specification before execution, not adaptive switching or alternate videos outside the three held-out candidates.'])
    (root/'selection_report.json').write_text(json.dumps(report,indent=2))
    with (root/'selection_decisions.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(decisions[0]));writer.writeheader();writer.writerows(decisions)
    lines=['# Prospective PMI ranking versus task completion','', '## 1. The experiment','',
        f"Ran {len(episodes)} full branches in {len(contexts)} matched contexts: four tasks × three initial states × six candidates (three language and three video variants), capped at {manifest['max_steps']} steps. Environment/action seed is {manifest['seed']}.", '',
        'All six candidates were scored at the identical initial observation before any branch was executed. Rankings use Monte Carlo expected action lift with 512 action samples per candidate. The chosen specification remains fixed throughout its rollout. Starting observations and action-noise streams match across candidates; histories evolve independently after actions diverge.', '',
        'Primary score: the existing modality-specific blank reference. Frozen secondary scores: a full 94-task same-modality mixture and a shared mixture of both modalities. Uniform task/variant weights are used; all KITCHEN_SCENE10 tasks are excluded from the empirical reference.', '',
        '## 2. Why it was performed','',
        'Test whether a high score predicts successful execution rather than merely a change in the action distribution. The video-only comparison directly tests selecting among three demonstrations; the six-candidate comparison also allows language selection.', '',
        '## 3. The results','',
        '| Candidate set | Reference | Selected successes | Uniform expected successes | Best-available oracle | Success/failure pair concordance | Clear top rankings |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for s in summaries:
        concordance=f"{s['pair_concordance']:.1%}" if s['pair_concordance'] is not None else 'No mixed outcomes'
        lines.append(f"| {s['scope']} | {s['reference']} | {s['selected_successes']}/{s['contexts']} | {s['uniform_expected_successes']:.2f}/{s['contexts']} | {s['oracle_successes']}/{s['contexts']} | {concordance} | {s['clear_rankings']}/{s['contexts']} |")
    lines+=['',f"Fixed variant 0: language {fixed['gl']}/{len(contexts)}; video {fixed['vid']}/{len(contexts)}.", '',
        'The oracle uses measured outcomes after execution and is only an upper bound. Uniform expected success averages each context’s branch outcomes. A clear top ranking means the top-two score gap exceeds two combined sampling standard errors; it is not a success-confidence guarantee.', '',
        '### Primary blank-reference choices by context','',
        '| Task | Initial state | Best of six: choice / outcome | Best video: choice / outcome | Available successes, all / video |',
        '| --- | ---: | --- | --- | --- |']
    for r in contexts:
        a=next(d for d in decisions if d['task_id']==r['task_id'] and d['initial_state_index']==r['initial_state_index'] and d['scope']=='all' and d['reference']=='blank')
        v=next(d for d in decisions if d['task_id']==r['task_id'] and d['initial_state_index']==r['initial_state_index'] and d['scope']=='video' and d['reference']=='blank')
        result=lambda d:f"{d['selected_modality']}{d['selected_spec_index']} / {'success' if d['success'] else 'failure'}"
        lines.append(f"| {r['task_id']} | {r['initial_state_index']} | {result(a)} | {result(v)} | {sum(c['success'] for c in r['candidates'])}/6 / {sum(c['success'] for c in r['candidates'] if c['modality']=='vid')}/3 |")
    lines+=['', '## 4. What the results imply','',
        'Compare selected success with the uniform expectation and fixed baselines, especially where some candidates succeed and others fail. An incorrect choice in a context with a successful alternative is a direct counterexample to guaranteed selection. Clear score gaps can still accompany poor choices: sampling confidence is not completion confidence. The small diagnostic grid cannot establish population reliability or rule out other PMI constructions.', '',
        '[Complete rankings and outcomes](selection_report.json) · [Per-context decisions](selection_decisions.csv) · [Episode outcomes](episodes.json) · [Frozen protocol](manifest.json)', '',
        '### Limits','']+['- '+s for s in report['limits']]
    (root/'SELECTION_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__': main()
