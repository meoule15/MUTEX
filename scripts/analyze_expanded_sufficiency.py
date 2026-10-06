"""Aggregate completed paired modality studies without treating grid cells as independent."""
import argparse
import csv
import json
import math
from pathlib import Path


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [mid - half, mid + half]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    episodes = []
    manifests = []
    pairs = []
    for directory in sorted(root.glob('seed*')):
        if not directory.is_dir():
            continue
        manifest = json.loads((directory / 'manifest.json').read_text())
        complete = json.loads((directory / 'complete.json').read_text())
        rows = json.loads((directory / 'episodes.json').read_text())
        expected = len(manifest['tasks']) * manifest['initial_states'] * len(manifest['spec_indices']) * 2
        assert complete['status'] == 'complete' and complete['episodes'] == expected == len(rows)
        assert manifest['common_variant_action_noise'] and manifest['environment_spec_seed'] == 0
        maps = {}
        for row in rows:
            key = (row['task_id'], row['initial_state_index'], row['spec_index'])
            assert row['modality'] in ('gl', 'vid')
            assert (key, row['modality']) not in maps
            maps[key, row['modality']] = row
            assert (1 <= row['success_step'] <= manifest['max_steps']) if row['success'] else row['success_step'] == -1
            episodes.append(dict(row, seed=manifest['seed']))
        for key, modality in maps:
            if modality != 'gl':
                continue
            language, video = maps[key, 'gl'], maps[key, 'vid']
            assert language['initial_observation_sha256'] == video['initial_observation_sha256']
            assert language['environment_seed'] == video['environment_seed']
            pairs.append(dict(seed=manifest['seed'], task_id=key[0], initial_state_index=key[1],
                              spec_index=key[2], language=language['success'], video=video['success']))
        for task in manifest['tasks']:
            states = {r['initial_state_index'] for r in rows if r['task_id'] == task}
            assert len(states) == manifest['initial_states']
            for state in states:
                selected = [r for r in rows if r['task_id'] == task and r['initial_state_index'] == state]
                assert len(selected) == 6
                assert len({r['initial_observation_sha256'] for r in selected}) == 1
                assert len({r['environment_seed'] for r in selected}) == 1
        manifests.append(manifest)
    assert len(manifests) == 2 and {m['seed'] for m in manifests} == {17, 29}
    for field in ('tasks', 'initial_states', 'spec_indices', 'max_steps', 'modalities'):
        assert manifests[0][field] == manifests[1][field]
    tasks = manifests[0]['tasks']
    summaries = []
    for task in tasks:
        p = [r for r in pairs if r['task_id'] == task]
        summary = dict(task_id=task, task_name=manifests[0]['task_names'][str(task)], pairs=len(p))
        summary['paired'] = dict(both=sum(r['language'] and r['video'] for r in p),
                                language_only=sum(r['language'] and not r['video'] for r in p),
                                video_only=sum(r['video'] and not r['language'] for r in p),
                                neither=sum(not r['language'] and not r['video'] for r in p))
        for modality in ('gl', 'vid'):
            selected = [r for r in episodes if r['task_id'] == task and r['modality'] == modality]
            k, n = sum(r['success'] for r in selected), len(selected)
            summary[modality] = dict(successes=k, episodes=n, rate=k / n, descriptive_wilson95=wilson(k, n),
                                    meets_provisional_observed_target=k / n >= .9)
            summary[modality]['seeds'] = {
                str(seed): sum(r['success'] for r in selected if r['seed'] == seed) for seed in (17, 29)}
            summary[modality]['variants'] = {
                str(spec): sum(r['success'] for r in selected if r['spec_index'] == spec) for spec in (0, 1, 2)}
            summary[modality]['initial_states'] = {
                str(state): sum(r['success'] for r in selected if r['initial_state_index'] == state)
                for state in sorted({r['initial_state_index'] for r in selected})}
        summaries.append(summary)
    totals = {m: dict(successes=sum(r['success'] for r in episodes if r['modality'] == m),
                      episodes=sum(r['modality'] == m for r in episodes)) for m in ('gl', 'vid')}
    paired_totals = {k: sum(s['paired'][k] for s in summaries) for k in ('both', 'language_only', 'video_only', 'neither')}
    report = dict(episodes=len(episodes), matched_pairs=len(pairs), target=.9, totals=totals,
                  paired_totals=paired_totals, tasks=summaries, verification='All episode counts, outcomes, paired starts, and cross-variant starts checked.',
                  uncertainty='Wilson intervals are descriptive iid-binomial approximations. Shared states, variants and noise induce dependence; no certification or significance claim.')
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    with (root / 'completion_rates.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['task_id', 'modality', 'successes', 'episodes', 'rate', 'descriptive_wilson95_low', 'descriptive_wilson95_high'])
        for s in summaries:
            for m in ('gl', 'vid'):
                r = s[m]
                writer.writerow([s['task_id'], m, r['successes'], r['episodes'], r['rate'], *r['descriptive_wilson95']])
    lines = ['# Expanded paired modality sufficiency results', '', '## 1. The experiment', '',
             'Completed **1,008 episodes**: 12 tasks × seven initial states × three held-out specification variants × two goal modalities × two seeds (17 and 29). Each trajectory has a 600-step limit. Tasks 25, 90 and 99 add coverage beyond the earlier nine-task study.', '',
             'The frozen policy receives cameras and robot state in both conditions. Language and video specify the goal. Specifications remain fixed; PMI and progress fallback do not control these runs. Environment seeds and initial observations match across modalities and variants; action-sampling noise is coupled at every step. Independent policy histories evolve after actions diverge.', '',
             'The seven state indices are sampled evenly from each task’s initial-state bank. This expands state coverage but overlaps some earlier indices; the two action/environment seeds are new. These are seen training tasks, not unseen-task generalization.', '',
             '## 2. Why it was performed', '',
             'Measure modality sufficiency through full task completion rather than action-information scores. Expand the previous small grid, check whether modality differences persist across seeds, and identify tasks and specification variants where either modality falls short.', '',
             '## 3. The results', '',
             f"**Language: {totals['gl']['successes']}/{totals['gl']['episodes']} ({totals['gl']['successes']/totals['gl']['episodes']:.1%}). Video: {totals['vid']['successes']}/{totals['vid']['episodes']} ({totals['vid']['successes']/totals['vid']['episodes']:.1%}).** Rates describe this equally weighted task grid.", '',
             f"Across 504 matched pairs: both succeed in {paired_totals['both']}, language alone in {paired_totals['language_only']}, video alone in {paired_totals['video_only']}, and neither in {paired_totals['neither']}.", '',
             'Each task has 42 trials per modality. Parentheses show descriptive 95% Wilson intervals; shared states, variants and noise mean these are not calibrated population confidence bounds.', '',
             '| Task | Language completion | Video completion | Language only / video only | Observed ≥90% target |',
             '| --- | --- | --- | --- | --- |']
    for s in summaries:
        cells = []
        for m in ('gl', 'vid'):
            r = s[m]
            lo, hi = r['descriptive_wilson95']
            cells.append(f"{r['successes']}/42 = {r['rate']:.1%} ({lo:.1%}–{hi:.1%})")
        qualifying = [label for m, label in (('gl', 'language'), ('vid', 'video')) if s[m]['meets_provisional_observed_target']]
        lines.append(f"| {s['task_id']} | {cells[0]} | {cells[1]} | {s['paired']['language_only']} / {s['paired']['video_only']} | {', '.join(qualifying) or 'neither'} |")
    lines += ['', '### Seed sensitivity', '', 'Each seed contributes 21 trials per task and modality.', '',
              '| Task | Language seed 17 / 29 | Video seed 17 / 29 |', '| --- | --- | --- |']
    for s in summaries:
        lines.append(f"| {s['task_id']} | {s['gl']['seeds']['17']} / {s['gl']['seeds']['29']} | {s['vid']['seeds']['17']} / {s['vid']['seeds']['29']} |")
    lines += ['', '### Specification sensitivity', '', 'Each variant contributes 14 trials (seven states × two seeds). Counts are variant 0 / 1 / 2. These are descriptive comparisons with matched starts and action noise, not evidence that a variant is universally best.', '',
              '| Task | Language variants | Video variants |', '| --- | --- | --- |']
    for s in summaries:
        cells = [' / '.join(str(s[m]['variants'][str(i)]) for i in (0, 1, 2)) for m in ('gl', 'vid')]
        lines.append(f"| {s['task_id']} | {cells[0]} | {cells[1]} |")
    lines += ['', '### Verification', '',
              'Both completion markers are present. All 1,008 outcomes have valid success steps, all 504 language/video pairs match initial observation hashes and environment seeds, and all six candidates share the same start within each of the 168 seed/task/state contexts. No duplicate episode keys were found.', '',
              '## 4. What the results imply', '',
              'Video leads on nine tasks, language leads on tasks 3 and 5, and task 25 ties at 42/42 for both. At the provisional observed 90% threshold, language qualifies on tasks 3 and 25; video qualifies on tasks 0, 1, 2, 25 and 99. Neither qualifies on tasks 4, 5, 6, 50, 75 or 90.', '',
              'The language advantage on tasks 3 and 5 appears in both seeds. This rules out treating video as uniformly superior within the evaluated grid. Task 25 provides the strongest observed evidence that either goal modality can suffice in these conditions; task 0 video also succeeds in every tested case.', '',
              'Specification choice matters substantially: task 1 language variants score 0/14, 13/14 and 10/14; task 2 language variants score 11/14, 1/14 and 14/14. Poor modality averages therefore do not establish that all specifications of that modality are inadequate. Choosing a winning variant after seeing these outcomes would require fresh validation.', '',
              'Task 4 is sensitive to seed: language falls from 10/21 to 3/21 and video from 15/21 to 7/21. Seed changes both environment context and action noise here, so this comparison does not isolate their individual effects. Both modalities also remain below 50% on task 90, suggesting that merely choosing between them may leave substantial failures.', '',
              'The earlier nine-task grid found video ahead on task 5; the expanded grid finds language ahead. Several threshold classifications also change. Different state coverage, seeds, batch sizes, and cross-variant controls prevent attributing this change to one factor, but it demonstrates why the earlier small-grid conclusions should remain provisional.', '',
              'The 90% target is provisional and describes this grid. Meeting it is a screening result, not a reliability certificate. Failing it does not prove a modality can never suffice; performance depends on the frozen policy, the chosen specifications, initial states, action noise and the 600-step horizon.', '',
              'These runs measure fixed-modality completion. They do not evaluate PMI ranking, adaptive selection, or recovery, and cannot establish that any selector is reliable. Paired successes in different modalities identify possible selection opportunities retrospectively; they do not show that those opportunities can be recognized before execution.', '',
              'Aggregate rates describe these 12 equally weighted tasks. Do not interpret them as a success probability over all LIBERO tasks or real deployments. The two seeds and reused state/variant grid are limited; additional independent validation is needed before a sufficiency claim.', '',
              '### Task names', '']
    lines += [f"- {s['task_id']}: {s['task_name']}" for s in summaries]
    lines += ['', '[Machine-readable report](report.json) · [Completion rates](completion_rates.csv) · [Seed 17 episodes](seed17/episodes.json) · [Seed 29 episodes](seed29/episodes.json)', '']
    (root / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(totals=totals, paired_totals=paired_totals,
                          tasks=[dict(task=s['task_id'], gl=s['gl']['successes'], vid=s['vid']['successes']) for s in summaries]), indent=2))


if __name__ == '__main__':
    main()
