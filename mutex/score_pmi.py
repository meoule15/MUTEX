"""Offline language/video lift on recorded LIBERO actions.

Run from MUTEX/: python -m mutex.score_pmi --help
"""
import argparse
import json
import pickle
from pathlib import Path

import h5py
import torch
from transformers.utils import logging as hf_logging
from easydict import EasyDict
import robomimic.utils.obs_utils as ObsUtils

from mutex.models.policy import BCMutexPolicy
from mutex.pmi import ActionLiftScorer, blank_specification
from mutex.utils import torch_load_model
from mutex.task_spec_provider import TaskSpecProvider
from mutex.embed_utils import get_visual_specifications_all
from libero.libero import benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True,
                        help='Parent directory containing libero_100/')
    parser.add_argument('--experiment', type=Path, default=Path('mutex_pretrained'))
    parser.add_argument('--checkpoint', default='mutex_weights.pth')
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--spec-index', type=int, default=0)
    parser.add_argument('--episode', default='demo_0')
    parser.add_argument('--modalities', nargs='+', choices=['gl', 'vid'], default=['gl'])
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Score an episode prefix; history starts at step zero')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reference', choices=['blank', 'mixture'], default='blank')
    parser.add_argument('--reference-task-ids', type=int, nargs='+', default=None)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error('--max-steps must be positive')
    if args.output.exists() or args.output.with_suffix('.summary.json').exists():
        parser.error('Output already exists; choose a new output path')
    if args.reference == 'mixture' and not args.reference_task_ids:
        parser.error('Mixture requires explicit --reference-task-ids')
    if args.reference == 'blank' and args.reference_task_ids:
        parser.error('--reference-task-ids is only used with mixture')
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    hf_logging.set_verbosity_error()
    cfg = EasyDict(json.loads((args.experiment / 'config.json').read_text()))
    cfg.device = args.device
    cfg.num_gpus = 1
    cfg.policy.task_spec_modalities = '_'.join(dict.fromkeys(args.modalities))
    cfg.folder = str(args.dataset.resolve())
    cfg.recalculate_ts_embs = False
    cfg.train.use_augmentation = False
    for key in ['add_mim', 'add_mgm', 'add_mrm', 'add_mfm', 'add_maim', 'add_magm']:
        cfg.policy[key] = False
    for key in ['inst', 'gl', 'ai', 'ag', 'img']:
        cfg.policy.projection_layer.network_kwargs[key + '_transform_kwargs'].network_kwargs.add_cross_modal_layer = True
    policy = BCMutexPolicy(cfg, cfg.shape_meta).to(args.device)
    policy.eval()
    checkpoint = args.experiment / 'models' / args.checkpoint
    state, _, _ = torch_load_model(str(checkpoint), device=args.device)
    policy.load_state_dict(state, strict=True)
    bm = benchmark.get_benchmark_dict()['libero_100'](0)
    if not 0 <= args.task_id < bm.n_tasks:
        parser.error('--task-id out of range')
    demo_path = args.dataset / bm.get_task_demonstration(args.task_id)
    cache = args.dataset / 'libero_100/task_spec/gl_openai_clip-vit-large-patch14_ts_mode_eval_emb.pt'
    with cache.open('rb') as source:
        embeddings, _ = pickle.load(source)
    if not 0 <= args.spec_index < embeddings.shape[1]:
        parser.error('--spec-index out of range')

    bm.set_gl_embs(embeddings)
    if 'vid' in args.modalities:
        tasks = [bm.get_task(i).name for i in range(bm.n_tasks)]
        demos = [bm.get_task_demonstration(i) for i in range(bm.n_tasks)]
        # Require the cache: never silently recompute or write to the dataset.
        visual_cache = args.dataset / 'libero_100/task_spec' / (
            'visual_' + cfg.tokenizer.replace('/', '_') + '_emb.pt')
        if not visual_cache.exists():
            parser.error('Missing visual cache: ' + str(visual_cache))
        bm.set_visual_task_specifications(get_visual_specifications_all(
            'LIBERO_100', tasks, demos, cfg, mode='eval'))
    provider = TaskSpecProvider(bm, policy, args.device, cfg.policy.num_task_frames)
    candidates, references, reference_keys = {}, {}, {}
    ids = args.reference_task_ids
    if ids and (len(set(ids)) != len(ids) or args.task_id in ids or
                any(not 0 <= i < bm.n_tasks for i in ids)):
        parser.error('Reference tasks must be unique, valid and exclude the scored task')
    for modality in dict.fromkeys(args.modalities):
        candidates[modality] = provider.get(args.task_id, modality, args.spec_index).unsqueeze(0)
        reference_keys[modality] = []
        if args.reference == 'blank':
            key = modality + ':blank'
            references[key] = provider.get_blank(args.task_id, modality, args.spec_index).unsqueeze(0)
            reference_keys[modality].append(key)
        else:
            for task_id in ids:
                key = modality + ':' + str(task_id)
                references[key] = provider.get(task_id, modality, args.spec_index).unsqueeze(0)
                reference_keys[modality].append(key)
    full = next(iter(candidates.values()))

    ObsUtils.initialize_obs_utils_with_obs_specs({'obs': {
        'rgb': ['agentview_rgb', 'eye_in_hand_rgb'],
        'low_dim': ['gripper_states', 'joint_states']}})
    scorer = ActionLiftScorer(policy)
    scorer.reset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lifts = {key: [] for key in candidates}
    metadata = {'checkpoint': str(checkpoint.resolve()), 'dataset': str(args.dataset.resolve()),
                'task_id': args.task_id, 'episode': args.episode,
                'spec_index': args.spec_index, 'action_source': 'demonstration',
                'reference': args.reference, 'reference_task_ids': args.reference_task_ids,
                'seed': args.seed, 'max_history': policy.max_seq_len}
    with h5py.File(demo_path, 'r') as h5, args.output.open('x') as output:
        episode = h5['data'][args.episode]
        for step in range(min(args.max_steps, len(episode['actions']))):
            obs = {}
            for key in cfg.shape_meta.all_obs_keys:
                value = torch.from_numpy(episode['obs'][key][step])
                obs[key] = ObsUtils.process_obs(value, obs_key=key).float().unsqueeze(0).to(args.device)
            action = torch.from_numpy(episode['actions'][step]).float().unsqueeze(0).to(args.device)
            results = scorer.score_step({'obs': obs, 'task_emb': full}, action,
                                        candidates, references, reference_keys)
            for modality, result in results.items():
                row = dict(metadata, modality=modality, step=step,
                           action=action[0].cpu().tolist(),
                           **{k: v.item() for k, v in result.items()})
                output.write(json.dumps(row, allow_nan=False) + '\n')
                lifts[modality].append(row['lift'])
    summary = dict(metadata, modalities={
        key: {'steps': len(values),
              'mean_lift': sum(values)/len(values) if values else None,
              'summed_lift': sum(values),
              'negative_fraction': sum(x < 0 for x in values)/len(values) if values else None}
        for key, values in lifts.items()})
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
