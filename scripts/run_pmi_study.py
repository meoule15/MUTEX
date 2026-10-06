"""Controlled CPU pilot: tune on tasks 0/1, evaluate on disjoint tasks 2--5.

Run from MUTEX: python scripts/run_pmi_study.py --output ../results/pmi-pilot
This is a capped pilot on training tasks, not a full LIBERO benchmark.
"""
import argparse
from collections import Counter
import json
import pickle
from pathlib import Path
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
from easydict import EasyDict
from transformers.utils import logging
import robomimic.utils.obs_utils as ObsUtils
from libero.libero import benchmark
from mutex.models.policy import BCMutexPolicy
from mutex.utils import torch_load_model, control_seed
from mutex.embed_utils import get_visual_specifications_all
from mutex.task_spec_provider import TaskSpecProvider
from mutex.pmi import ActionLiftScorer
from mutex.metric import evaluate_one_task_success
from mutex.logprob_recorder import LogProbRecorder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--dataset', type=Path, default=Path('/data/mehul/pmi_robot/datasets'))
    ap.add_argument('--max-steps', type=int, default=300)
    ap.add_argument('--offline-steps', type=int, default=100)
    ap.add_argument('--resume', action='store_true', help='Reuse completed rollout records')
    a = ap.parse_args()
    if a.output.exists() and not a.resume:
        ap.error('Choose a new output directory')
    if a.max_steps < 1 or a.offline_steps < 1:
        ap.error('Step limits must be positive')
    a.output.mkdir(parents=True, exist_ok=a.resume)
    torch.set_num_threads(4)
    logging.set_verbosity_error()
    cfg = EasyDict(json.load(open('mutex_pretrained/config.json')))
    cfg.device = 'cpu'; cfg.num_gpus = 1; cfg.folder = str(a.dataset.resolve())
    cfg.policy.task_spec_modalities = 'gl_vid'; cfg.train.use_augmentation = False
    cfg.recalculate_ts_embs = False
    cfg.bddl_folder = str(Path('LIBERO/libero/libero/bddl_files').resolve())
    cfg.init_states_folder = str(Path('LIBERO/libero/libero/init_files').resolve())
    cfg.eval.use_mp = False; cfg.eval.num_procs = 1; cfg.eval.n_eval = 1
    cfg.eval.max_steps = a.max_steps; cfg.record_pmi = True
    for k in ['add_mim', 'add_mgm', 'add_mrm', 'add_mfm', 'add_maim', 'add_magm']:
        cfg.policy[k] = False
    for k in ['inst', 'gl', 'ai', 'ag', 'img']:
        cfg.policy.projection_layer.network_kwargs[k+'_transform_kwargs'].network_kwargs.add_cross_modal_layer = True
    control_seed(0)
    policy = BCMutexPolicy(cfg, cfg.shape_meta)
    policy.eval()
    checkpoint = Path('mutex_pretrained/models/mutex_weights.pth')
    state, _, _ = torch_load_model(str(checkpoint), device='cpu')
    policy.load_state_dict(state, strict=True)
    del state
    bm = benchmark.get_benchmark_dict()['libero_100'](0)
    with (a.dataset/'libero_100/task_spec/gl_openai_clip-vit-large-patch14_ts_mode_eval_emb.pt').open('rb') as f:
        embeddings, _ = pickle.load(f)
    bm.set_gl_embs(embeddings)
    bm.set_visual_task_specifications(get_visual_specifications_all(
        'LIBERO_100', [bm.get_task(i).name for i in range(bm.n_tasks)],
        [bm.get_task_demonstration(i) for i in range(bm.n_tasks)], cfg, mode='eval'))
    provider = TaskSpecProvider(bm, policy, 'cpu', cfg.policy.num_task_frames)
    ObsUtils.initialize_obs_utils_with_obs_specs({'obs': {
        'rgb': ['agentview_rgb', 'eye_in_hand_rgb'],
        'low_dim': ['gripper_states', 'joint_states']}})
    settings = [dict(selection_interval=10, selection_margin=0.5, selection_uncertainty=2., pmi_samples=32),
                dict(selection_interval=20, selection_margin=2., selection_uncertainty=2., pmi_samples=64)]
    manifest = dict(validation_tasks=[0, 1], evaluation_tasks=[2, 3], broader_tasks=[4, 5],
                    reference_tasks=[6, 7], tuning_profiles=settings, checkpoint=str(checkpoint.resolve()),
                    dataset=str(a.dataset.resolve()), max_steps=a.max_steps, offline_steps=a.offline_steps,
                    n_eval=1, initial_state_index=0, spec_index=0,
                    note='All tasks were seen in checkpoint training. Different seeds vary action sampling, not initial-state index.',
                    task_names={str(i): bm.get_task(i).name for i in range(8)})
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2))

    class Algo:
        def __init__(self): self.policy = policy
        def eval(self): policy.eval()
        def reset(self): policy.reset()
    algo = Algo()
    results_path = a.output/'results.json'
    results = json.loads(results_path.read_text()) if a.resume and results_path.exists() else []
    for result in results:
        if isinstance(result['success'], list):
            result['success'] = result['success'][0]

    def rollout(phase, task_id, seed, mode, setting, profile):
        if any((r['phase'],r['task_id'],r['seed'],r['mode'],r['profile']) ==
               (phase,task_id,seed,mode,profile) for r in results):
            return
        cfg.update(setting); cfg.seed = seed
        cfg.spec_selection = mode if mode in ('random', 'expected_lift') else 'fixed'
        keys = ['vid', 'gl'] if mode == 'fixed_video' else ['gl', 'vid']
        control_seed(seed)
        path = a.output/f'{phase}-task{task_id}-seed{seed}-{mode}-{profile}.jsonl'
        recorder = LogProbRecorder(out_path=str(path), store_actions=True)
        recorder.set_context(task_id=task_id, task_name=bm.get_task(task_id).name, seed=seed,
                             phase=phase, mode=mode, profile=profile)
        started = time.perf_counter()
        try:
            success, _ = evaluate_one_task_success(cfg, algo, bm.get_task(task_id), None, task_id,
                        spec_provider=provider, recorder=recorder, spec_modalities=keys)
        finally:
            recorder.close()
        elapsed = time.perf_counter()-started
        rows = [json.loads(x) for x in path.open()]
        acted = [r for r in rows if r['modality'] == keys[0]]
        result = dict(phase=phase, task_id=task_id, seed=seed, mode=mode, profile=profile,
                      success=success, steps=len(acted), switches=sum(r['selection_switched'] for r in acted),
                      elapsed_seconds=elapsed, seconds_per_step=elapsed/len(acted),
                      acting_counts=dict(Counter(r['acting_modality'] for r in acted)), path=path.name)
        results.append(result)
        (a.output/'results.json').write_text(json.dumps(results, indent=2))
        print('COMPLETED', json.dumps(result), flush=True)

    for profile, setting in enumerate(settings):
        for task_id in [0, 1]:
            rollout('validation', task_id, 0, 'expected_lift', setting, profile)
    # Predeclared tie-break: success first, then fewer switches, then fewer steps,
    # then earlier profile. Do not use evaluation tasks to select settings.
    def rank(profile):
        rows = [r for r in results if r['profile']==profile]
        return (sum(r['success'] for r in rows), -sum(r['switches'] for r in rows),
                -sum(r['steps'] for r in rows), -profile)
    chosen = max(range(len(settings)), key=rank)
    frozen = dict(profile=chosen, settings=settings[chosen],
                  selection_rule='maximize validation success, then minimize switches, steps, profile index')
    (a.output/'frozen_settings.json').write_text(json.dumps(frozen, indent=2))
    print('FROZEN', json.dumps(frozen), flush=True)
    for phase, tasks, seeds in [('evaluation', [2, 3], [0]), ('broader', [4, 5], [0, 1])]:
        for task_id in tasks:
            for seed in seeds:
                for mode in ['fixed_language', 'fixed_video', 'random', 'expected_lift']:
                    rollout(phase, task_id, seed, mode, settings[chosen], chosen)

    # Matched recorded actions: evaluate both null constructions on the same
    # demonstration prefix. Reference task pool is fixed and disjoint.
    scorer = ActionLiftScorer(policy)
    with (a.output/'reference_comparison.jsonl').open('x') as out:
        for task_id in range(6):
            scorer.reset()
            candidates, refs, ref_keys = {}, {}, {}
            for modality in ['gl', 'vid']:
                full = provider.get(task_id, modality, 0).unsqueeze(0)
                refs[modality+':blank'] = provider.get_blank(task_id, modality, 0).unsqueeze(0)
                for other in [6, 7]:
                    refs[modality+':'+str(other)] = provider.get(other, modality, 0).unsqueeze(0)
                for reference in ['blank', 'mixture']:
                    key = modality+':'+reference
                    candidates[key] = full
                    ref_keys[key] = [modality+':blank'] if reference=='blank' else [modality+':6', modality+':7']
            with h5py.File(a.dataset/bm.get_task_demonstration(task_id), 'r') as f:
                ep = f['data/demo_0']
                for step in range(min(a.offline_steps, len(ep['actions']))):
                    obs = {key: ObsUtils.process_obs(torch.from_numpy(ep['obs'][key][step]),
                             obs_key=key).float().unsqueeze(0) for key in cfg.shape_meta.all_obs_keys}
                    action = torch.from_numpy(ep['actions'][step]).float().unsqueeze(0)
                    scores = scorer.score_step({'obs': obs}, action, candidates, refs, ref_keys)
                    for key, score in scores.items():
                        modality, reference = key.split(':')
                        row = dict(task_id=task_id, step=step, modality=modality, reference=reference,
                                   action=action[0].tolist(), **{k:v.item() for k,v in score.items()})
                        out.write(json.dumps(row, allow_nan=False)+'\n')
            print('REFERENCE_COMPLETED', task_id, flush=True)
    (a.output/'complete.json').write_text(json.dumps({'status':'complete', 'rollouts':len(results)}))


if __name__ == '__main__':
    main()
