"""Paired fixed-modality task-completion grid on LIBERO-100.

Run from MUTEX/: python scripts/evaluate_modality_sufficiency.py --output ../results/modality-sufficiency
Five initial states x three held-out variants x two modalities x nine tasks.
Vector environments run simulations concurrently; each episode has independent
policy history. Confidence intervals are descriptive, not a generalization claim.
"""
import argparse
import hashlib
import functools
import random
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from easydict import EasyDict
from transformers.utils import logging
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
import robomimic.utils.obs_utils as ObsUtils
from mutex.models.policy import BCMutexPolicy
from mutex.embed_utils import get_visual_specifications_all
from mutex.task_spec_provider import TaskSpecProvider
from mutex.metric import raw_obs_to_tensor_obs
from mutex.utils import torch_load_model, control_seed


def make_seeded_env(env_args, seed):
    # LIBERO.env.seed resets only NumPy; seed Python before scene construction too.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return OffScreenRenderEnv(**env_args)


def observation_hash(obs):
    h=hashlib.sha256()
    for key in ['agentview_image','robot0_eye_in_hand_image','robot0_joint_pos','robot0_gripper_qpos']:
        h.update(key.encode());h.update(np.asarray(obs[key]).tobytes())
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--dataset',type=Path,default=Path('/data/mehul/pmi_robot/datasets'))
    ap.add_argument('--tasks',type=int,nargs='+',default=[0,1,2,3,4,5,6,50,75])
    ap.add_argument('--initial-states',type=int,default=5)
    ap.add_argument('--max-steps',type=int,default=600)
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--environment-spec-seed',type=int,choices=[0,1,2],default=None,
                    help='Use one environment seed across all variants to isolate specification changes')
    ap.add_argument('--common-variant-action-noise',action='store_true',
                    help='Reset action-sampling RNG for each variant of the same initial state')
    ap.add_argument('--rank-pmi-samples',type=int,default=0,
                    help='Score all six candidates prospectively at each initial state')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.output.exists() and not args.resume: ap.error('Choose a new output directory')
    if min(args.initial_states,args.max_steps,args.threads)<1: ap.error('Counts must be positive')
    if len(set(args.tasks))!=len(args.tasks) or any(not 0<=t<100 for t in args.tasks): ap.error('Invalid task IDs')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    mp.set_start_method('spawn',force=True)
    torch.set_num_threads(args.threads);logging.set_verbosity_error();control_seed(args.seed)
    cfg=EasyDict(json.load(open('mutex_pretrained/config.json')))
    cfg.device=args.device;cfg.num_gpus=1;cfg.folder=str(args.dataset.resolve())
    cfg.policy.task_spec_modalities='gl_vid';cfg.train.use_augmentation=False
    cfg.recalculate_ts_embs=False
    for k in ['add_mim','add_mgm','add_mrm','add_mfm','add_maim','add_magm']:cfg.policy[k]=False
    for k in ['inst','gl','ai','ag','img']:
        cfg.policy.projection_layer.network_kwargs[k+'_transform_kwargs'].network_kwargs.add_cross_modal_layer=True
    policy=BCMutexPolicy(cfg,cfg.shape_meta);policy.eval()
    checkpoint=Path('mutex_pretrained/models/mutex_weights.pth')
    state,_,_=torch_load_model(str(checkpoint),device='cpu');policy.load_state_dict(state,strict=True);del state
    policy.to(args.device)
    bm=benchmark.get_benchmark_dict()['libero_100'](0)
    with (args.dataset/'libero_100/task_spec/gl_openai_clip-vit-large-patch14_ts_mode_eval_emb.pt').open('rb') as f:
        embeddings,_=pickle.load(f)
    assert embeddings.shape[:2]==(100,3)
    bm.set_gl_embs(embeddings)
    bm.set_visual_task_specifications(get_visual_specifications_all(
        'LIBERO_100',[bm.get_task(i).name for i in range(100)],
        [bm.get_task_demonstration(i) for i in range(100)],cfg,mode='eval'))
    provider=TaskSpecProvider(bm,policy,args.device,cfg.policy.num_task_frames)
    provider.prepare(args.tasks,['gl','vid'])
    reference_banks=None
    if args.rank_pmi_samples:
        if args.rank_pmi_samples<2 or args.environment_spec_seed is None or not args.common_variant_action_noise:
            ap.error('Ranking requires >=2 samples, identical environment seeds, and common action noise')
        reference_tasks=list(range(6,100))
        if any(t not in range(6) for t in args.tasks): ap.error('Ranking reference excludes scene10; target tasks must be 0–5')
        provider.prepare(reference_tasks,['gl','vid'])
        reference_banks={m:torch.stack([provider.get(t,m,s) for t in reference_tasks for s in range(3)]) for m in ['gl','vid']}
    ObsUtils.initialize_obs_utils_with_obs_specs({'obs':{'rgb':['agentview_rgb','eye_in_hand_rgb'],
                                                     'low_dim':['gripper_states','joint_states']}})
    manifest=dict(tasks=args.tasks,initial_states=args.initial_states,spec_indices=[0,1,2],
                  max_steps=args.max_steps,seed=args.seed,modalities=['gl','vid'],
                  device=args.device,fresh_seeded_environments_per_modality=True,
                  pairing='Same initial-state indices, specification indices, environment seeds and per-step Torch seeds; fixed batch size/order.',
                  task_names={str(t):bm.get_task(t).name for t in args.tasks},
                  conditions='Seen training tasks; fixed policy; no specification switching or likelihood diagnostics during control.')
    if args.environment_spec_seed is not None:
        manifest['environment_spec_seed']=args.environment_spec_seed
    if args.common_variant_action_noise:
        manifest['common_variant_action_noise']=True
    if args.rank_pmi_samples:
        manifest['prospective_ranking']=dict(samples=args.rank_pmi_samples,reference_tasks=reference_tasks,
            primary='blank',secondary=['full_modality','full_shared'],selection='highest mean score, first candidate breaks ties',
            decision='before first action; hold chosen specification for complete rollout',history='one initial observation',
            scopes=['all six candidates','video-only three demonstrations'],outcome='any task success within max_steps',
            baselines=['uniform candidate expected success','fixed modality variant 0','best available candidate oracle'],
            no_outcome_tuning=True)
    path=args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest: ap.error('Resume settings differ from saved manifest')
    path.write_text(json.dumps(manifest,indent=2))
    results_path=args.output/'episodes.json'
    results=json.loads(results_path.read_text()) if results_path.exists() else []
    for task_id in args.tasks:
        task=bm.get_task(task_id)
        initial_path=Path('LIBERO/libero/libero/init_files')/task.problem_folder/task.init_states_file
        initial=torch.load(initial_path)
        if len(initial)<args.initial_states: raise ValueError('Not enough distinct initial states')
        ids=np.linspace(0,len(initial)-1,args.initial_states,dtype=int).tolist()
        cases=list(itertools.product(ids,range(3)));count=len(cases)
        missing=[m for m in ['gl','vid'] if not any(r['task_id']==task_id and r['modality']==m for r in results)]
        if not missing: continue
        env_args=dict(bddl_file_name=str(Path('LIBERO/libero/libero/bddl_files')/task.problem_folder/task.bddl_file),
                      camera_heights=cfg.data.img_h,camera_widths=cfg.data.img_w)
        for modality in missing:
            env=None
            try:
                started=time.perf_counter()
                env_seeds=[args.seed*100000+task_id*1000+state_id*3+
                           (spec if args.environment_spec_seed is None else args.environment_spec_seed)
                           for state_id,spec in cases]
                env=SubprocVectorEnv([functools.partial(make_seeded_env,env_args,seed) for seed in env_seeds])
                env.seed(env_seeds);env.reset()
                obs=env.set_init_state(np.stack([initial[state_id] for state_id,_ in cases]))
                for _ in range(5):obs,_,_,_=env.step(np.zeros((count,7)))
                already=[bool(x) for x in env.check_success()]
                if any(already):raise ValueError('An initial state already satisfies the task')
                hashes=[observation_hash(o) for o in obs]
                if args.environment_spec_seed is not None:
                    for state_id in ids:
                        assert len({hashes[i] for i,c in enumerate(cases) if c[0]==state_id})==1, 'Unmatched cross-variant starts'
                np.savez_compressed(args.output/f'task{task_id}-{modality}-initial_obs.npz',
                    **{key:np.stack([o[key] for o in obs]) for key in
                       ['agentview_image','robot0_eye_in_hand_image','robot0_joint_pos','robot0_gripper_qpos']})
                paired={ (r['initial_state_index'],r['spec_index']):r for r in results if r['task_id']==task_id}
                for index,case in enumerate(cases):
                    if case in paired:assert hashes[index]==paired[case]['initial_observation_sha256'],'Unmatched initial observations'
                if args.rank_pmi_samples and modality=='gl':
                    from rank_completion_candidates import rank_candidates
                    for state_id in ids:
                        score_path=args.output/f'task{task_id}-state{state_id}-ranking.json'
                        if not score_path.exists():
                            index=cases.index((state_id,0))
                            ranking=rank_candidates(policy,obs[index],cfg,provider,task_id,state_id,
                                                    reference_banks,args.rank_pmi_samples,args.seed)
                            ranking['initial_observation_sha256']=hashes[index]
                            score_path.write_text(json.dumps(ranking,indent=2))
                            print('RANKED',task_id,state_id,flush=True)
                task_emb=torch.stack([provider.get(task_id,modality,spec) for _,spec in cases])
                policy.reset();success=np.zeros(count,dtype=bool);first=np.full(count,-1,dtype=int)
                trace=[]
                for step in range(1,args.max_steps+1):
                    data=raw_obs_to_tensor_obs(obs,task_emb,cfg)
                    # Reset the common per-step stream: ending another case cannot
                    # shift the action noise of still-active episodes.
                    devices=[torch.device(args.device).index or 0] if args.device.startswith('cuda') else []
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(args.seed*10000000+task_id*10000+step)
                        if args.common_variant_action_noise:
                            with torch.no_grad():
                                latents=policy._encode_obs_step(data)
                                dist=policy._action_dist_from_latents(latents,data['task_emb'])
                                base=dist.component_distribution.base_dist
                                draws=[]
                                for i,(state_id,_) in enumerate(cases):
                                    torch.manual_seed(args.seed*10000000+task_id*10000+step+state_id*100000)
                                    single=torch.distributions.MixtureSameFamily(
                                        torch.distributions.Categorical(logits=dist.mixture_distribution.logits[i]),
                                        torch.distributions.Independent(torch.distributions.Normal(base.loc[i],base.scale[i]),1))
                                    draws.append(single.sample())
                                actions=torch.stack(draws).cpu().numpy()
                        else:
                            actions=policy.get_action(data)
                    assert actions.shape==(count,7) and np.isfinite(actions).all()
                    actions[success]=0
                    obs,_,done,_=env.step(actions)
                    new=np.asarray(done,dtype=bool)&~success
                    first[new]=step;success|=new
                    trace.append(actions.copy())
                    if success.all():break
                elapsed=time.perf_counter()-started
                np.savez_compressed(args.output/f'task{task_id}-{modality}-actions.npz',actions=np.stack(trace),
                                    cases=np.array(cases),success=success,success_step=first)
                for i,(state_id,spec) in enumerate(cases):
                    results.append(dict(task_id=task_id,task_name=task.name,modality=modality,
                        initial_state_index=state_id,spec_index=spec,environment_seed=env_seeds[i],
                        success=bool(success[i]),success_step=int(first[i]),
                        episode_steps=int(first[i]) if success[i] else args.max_steps,
                        initial_observation_sha256=hashes[i],group_elapsed_seconds=elapsed,
                        group_simulation_steps=step))
                results_path.write_text(json.dumps(results,indent=2))
                print('COMPLETED',task_id,modality,'successes',int(success.sum()),'/',count,
                      'steps',step,'seconds',round(elapsed,1),flush=True)
            finally:
                if env is not None:env.close()
    expected=len(args.tasks)*args.initial_states*3*2
    assert len(results)==expected
    for task in args.tasks:
        language={(r['initial_state_index'],r['spec_index']):r for r in results if r['task_id']==task and r['modality']=='gl'}
        video={(r['initial_state_index'],r['spec_index']):r for r in results if r['task_id']==task and r['modality']=='vid'}
        assert language.keys()==video.keys()
        assert all(language[k]['initial_observation_sha256']==video[k]['initial_observation_sha256'] for k in language)
    (args.output/'complete.json').write_text(json.dumps(dict(status='complete',episodes=expected,initial_observations_matched=True)))


if __name__=='__main__':main()
