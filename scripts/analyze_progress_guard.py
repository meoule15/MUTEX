"""Analyze matched fresh-seed progress-guard ablations, retaining oracle limits."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path)
    root=ap.parse_args().directory;assert (root/'complete.json').exists()
    manifest=json.loads((root/'manifest.json').read_text());rows=json.loads((root/'episodes.json').read_text())
    modes=manifest['strategies'];summaries=[];pairs=[];verified=0
    for mode in modes:
        selected=[r for r in rows if r['strategy']==mode]
        summaries.append(dict(strategy=mode,successes=sum(r['success'] for r in selected),episodes=len(selected),
            switches=sum(r['progress_switches'] for r in selected),
            successful_mean_steps=float(np.mean([r['success_step'] for r in selected if r['success']])) if any(r['success'] for r in selected) else None))
    for task in manifest['tasks']:
        batch=[r for r in rows if r['task_id']==task];trace=np.load(root/f'task{task}-actions.npz')
        assert np.isfinite(trace['actions']).all()
        assert trace['success_step'].tolist()==[r['success_step'] for r in batch]
        for state in sorted({r['initial_state_index'] for r in batch}):
            context=[r for r in batch if r['initial_state_index']==state];assert len(context)==4
            assert len({r['initial_observation_sha256'] for r in context})==1
            assert len({r['environment_seed'] for r in context})==1
            pmi=next(r for r in context if r['strategy']=='guarded_pmi');guard=next(r for r in context if r['strategy']=='guarded_progress')
            assert pmi['initial_choice']==guard['initial_choice']
            switch=next((e for e in guard['events'] if 'fallback_to' in e),None)
            effective=switch['effective_step'] if switch else len(trace['actions'])+1
            a=batch.index(pmi);b=batch.index(guard)
            # After a branch succeeds, its actions are zeroed. Compare active prefixes.
            length=min(effective-1,pmi['success_step'] if pmi['success'] else len(trace['actions']),
                       guard['success_step'] if guard['success'] else len(trace['actions']))
            error=float(np.max(np.abs(trace['actions'][:length,a]-trace['actions'][:length,b]))) if length else 0.
            assert error<1e-4,'Unmatched pre-intervention actions'
            if not switch:
                assert pmi['success']==guard['success'] and pmi['success_step']==guard['success_step']
            for r in context:
                assert any(e['joint_complete'] for e in r['events'])==r['success']
                assert all(e['joint_complete']==all(e['current_predicates']) for e in r['events'])
            pairs.append(dict(task_id=task,initial_state_index=state,abstention_success=pmi['success'],progress_success=guard['success'],
                initial_choice=guard['initial_choice'],trigger_step=switch['step'] if switch else None,
                trigger_reason=switch['reason'] if switch else None,fallback=switch['fallback_to'] if switch else None,
                guard_success_step=guard['success_step'],maximum_pre_intervention_action_error=error))
            verified+=1
    improvements=sum(p['progress_success'] and not p['abstention_success'] for p in pairs)
    harms=sum(p['abstention_success'] and not p['progress_success'] for p in pairs)
    report=dict(protocol=manifest,summaries=summaries,paired_cases=pairs,improvements=improvements,harms=harms,
        verification=dict(contexts=verified,identical_starts=True,matched_pre_intervention_actions=True,joint_completion_verified=True),
        limitations=['LIBERO predicates are privileged benchmark information, not a learned deployable progress signal.',
                     'One seed on previously studied tasks and limited initial states; exploratory, correlated cases.',
                     'Fallback is one fixed opposite-modality variant0 intervention, not a calibrated success predictor.',
                     'This cohort tests the abstention-only and progress extensions against fixed baselines; it does not rerun the old forced-ranking selector on this seed.'])
    (root/'report.json').write_text(json.dumps(report,indent=2))
    lines=['# Abstention and progress fallback validation','', '## 1. The experiment','',
        f"Ran {len(rows)} episodes: {len(manifest['tasks'])} tasks × {manifest['initial_states']} initial states × four strategies, with a 600-step limit and seed {manifest['seed']}.", '',
        'The guarded strategies initially use the same six-candidate blank-PMI ranking. Unresolved top-two gaps retain video variant 0. The progress strategy permits one opposite-modality variant-0 fallback after five consecutive lost-subgoal observations or 240 steps without new predicate progress. Rules were frozen before fresh-seed outcomes.', '',
        '**Progress uses privileged LIBERO goal predicates. It is a benchmark-assisted experiment, not a learned completion predictor or production-ready controller.** Goal predicates are kept outside the policy inputs.', '',
        '## 2. Why it was performed','',
        'Implement explicit score abstention and test whether detection of regression/stagnation helps execution. The abstention-only control isolates the progress intervention from initial specification selection.', '',
        '## 3. The results','',
        '| Strategy | Successes | Progress switches |', '| --- | ---: | ---: |']
    lines += [f"| {s['strategy']} | {s['successes']}/{s['episodes']} | {s['switches']} |" for s in summaries]
    lines += ['',f"Compared with abstention-only selection, progress fallback recovered **{improvements}** failures and introduced **{harms}** failures.", '',
              '### Per-task counts','', '| Task | Fixed language | Fixed video | Abstention only | With progress fallback |', '| --- | ---: | ---: | ---: | ---: |']
    for task in manifest['tasks']:
        values=[sum(r['success'] for r in rows if r['task_id']==task and r['strategy']==mode) for mode in modes]
        lines.append(f"| {task} | "+' | '.join(f"{v}/{manifest['initial_states']}" for v in values)+' |')
    lines+=['','### Paired progress interventions','','| Task | State | Initial choice | Trigger | Fallback | Abstention / progress outcome |','| --- | ---: | --- | --- | --- | --- |']
    for p in pairs:
        trigger=f"{p['trigger_reason']} at {p['trigger_step']}" if p['trigger_step'] else 'None'
        lines.append(f"| {p['task_id']} | {p['initial_state_index']} | {p['initial_choice']} | {trigger} | {p['fallback'] or 'None'} | {int(p['abstention_success'])} / {int(p['progress_success'])} |")
    lines+=['', 'All matched-start, pre-intervention action, saved-outcome, and joint-predicate completion checks passed.', '',
        '## 4. What the results imply','',
        'Use the paired recoveries and harms to judge this intervention. A net gain on this small cohort is exploratory and does not validate general reliability. A harmful fallback shows that a plateau is not proof that the current specification cannot finish. The simulator predicate adapter must be replaced by a separately validated observable/learned signal for deployment.', '',
        '[Detailed outcomes and events](episodes.json) · [Verification and paired comparisons](report.json) · [Frozen protocol](manifest.json)', '',
        '### Limits','']+['- '+s for s in report['limitations']]
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summaries=summaries,improvements=improvements,harms=harms,verification=report['verification']),indent=2))


if __name__=='__main__':main()
