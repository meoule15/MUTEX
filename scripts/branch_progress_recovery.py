"""Exact saved-prefix regression intervention; post-hoc mechanism test only."""
import argparse
import functools
import json
from pathlib import Path
import pickle
import numpy as np
import torch
import evaluate_modality_sufficiency as setup
from evaluate_progress_guard import make_env
from mutex.progress import GoalProgressMonitor


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--source',type=Path,default=Path('../results/pmi-prospective-completion'))
    args=ap.parse_args()
    if args.output.exists():ap.error('Choose a new output directory')
    args.output.mkdir(parents=True);setup.mp.set_start_method('spawn',force=True)
    torch.set_num_threads(4);setup.logging.set_verbosity_error();setup.control_seed(7)
    cfg=setup.EasyDict(json.load(open('mutex_pretrained/config.json')))
    cfg.device='cuda:0';cfg.num_gpus=1;cfg.folder='/data/mehul/pmi_robot/datasets'
    cfg.policy.task_spec_modalities='gl_vid';cfg.train.use_augmentation=False;cfg.recalculate_ts_embs=False
    for k in ['add_mim','add_mgm','add_mrm','add_mfm','add_maim','add_magm']:cfg.policy[k]=False
    for k in ['inst','gl','ai','ag','img']:
        cfg.policy.projection_layer.network_kwargs[k+'_transform_kwargs'].network_kwargs.add_cross_modal_layer=True
    policy=setup.BCMutexPolicy(cfg,cfg.shape_meta);policy.eval()
    weights,_,_=setup.torch_load_model('mutex_pretrained/models/mutex_weights.pth',device='cpu')
    policy.load_state_dict(weights,strict=True);del weights;policy.to(cfg.device)
    bm=setup.benchmark.get_benchmark_dict()['libero_100'](0)
    with open(cfg.folder+'/libero_100/task_spec/gl_openai_clip-vit-large-patch14_ts_mode_eval_emb.pt','rb') as f:emb,_=pickle.load(f)
    bm.set_gl_embs(emb);bm.set_visual_task_specifications(setup.get_visual_specifications_all('LIBERO_100',
        [bm.get_task(i).name for i in range(100)],[bm.get_task_demonstration(i) for i in range(100)],cfg,mode='eval'))
    provider=setup.TaskSpecProvider(bm,policy,cfg.device,cfg.policy.num_task_frames);provider.prepare([5],['gl','vid'])
    setup.ObsUtils.initialize_obs_utils_with_obs_specs({'obs':{'rgb':['agentview_rgb','eye_in_hand_rgb'],'low_dim':['gripper_states','joint_states']}})
    source_rows=json.loads((args.source/'episodes.json').read_text())
    case=next(r for r in source_rows if r['task_id']==5 and r['initial_state_index']==0 and r['modality']=='vid' and r['spec_index']==0)
    trace=np.load(args.source/'task5-vid-actions.npz');column=trace['cases'].tolist().index([0,0]);saved=trace['actions'][:,column]
    info=bm.get_task(5);base=Path('LIBERO/libero/libero')
    env_args=dict(bddl_file_name=str(base/'bddl_files'/info.problem_folder/info.bddl_file),camera_heights=cfg.data.img_h,camera_widths=cfg.data.img_w)
    env=setup.SubprocVectorEnv([functools.partial(make_env,env_args,case['environment_seed']) for _ in range(2)])
    try:
        env.seed([case['environment_seed']]*2);env.reset();initial=torch.load(base/'init_files'/info.problem_folder/info.init_states_file)
        obs=env.set_init_state(np.stack([initial[0]]*2))
        for _ in range(5):obs,_,_,_=env.step(np.zeros((2,7)))
        assert all(setup.observation_hash(o)==case['initial_observation_sha256'] for o in obs)
        progress=env.get_env_attr('current_goal_progress');monitors=[GoalProgressMonitor(len(p)) for p in progress]
        events=[[monitors[i].update(0,p)] for i,p in enumerate(progress)]
        policy.reset();success=[False,False];first=[-1,-1];intervention=None;actions_saved=[]
        for step in range(1,601):
            data=setup.raw_obs_to_tensor_obs(obs,provider.get(5,'gl',0).unsqueeze(0).repeat(2,1,1),cfg)
            with torch.no_grad():x=policy._encode_obs_step(data)
            actions=np.stack([saved[step-1]]*2).copy()
            if intervention is not None:
                with torch.no_grad():dist=policy._action_dist_from_latents(x[1:2],provider.get(5,'gl',0).unsqueeze(0))
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(7*10000000+5*10000+step)
                    actions[1]=dist.sample()[0].cpu().numpy()
            for i in range(2):
                if success[i]:actions[i]=0
            obs,_,done,_=env.step(actions);actions_saved.append(actions.copy())
            progress=env.get_env_attr('current_goal_progress')
            for i,p in enumerate(progress):
                if success[i]:continue
                e=monitors[i].update(step,p);events[i].append(e)
                assert e['joint_complete']==bool(done[i])
                if done[i]:success[i]=True;first[i]=step
            if intervention is None:
                assert setup.observation_hash(obs[0])==setup.observation_hash(obs[1])
                if events[1][-1]['reassess']:
                    assert events[1][-1]['reason']=='regression' and step==220
                    intervention=dict(step=step,effective_step=step+1,from_spec='vid0',to_spec='gl0',reason='regression',
                        branch_observation_sha256=setup.observation_hash(obs[1]))
            if all(success):break
        assert first[0]==case['success_step'] and intervention is not None
        result=dict(source_case=case,source_prefix_reproduced=True,intervention=intervention,
            branches=[dict(strategy='saved_video_continuation',success=success[0],success_step=first[0]),
                      dict(strategy='regression_fallback_language0',success=success[1],success_step=first[1])],
            events=events,limitations='Post-hoc exact-prefix intervention. Baseline continues saved actions; altered branch uses live policy with original per-step action RNG. Not independent reliability validation.')
        (args.output/'report.json').write_text(json.dumps(result,indent=2));np.savez_compressed(args.output/'actions.npz',actions=np.stack(actions_saved))
        print(json.dumps(dict(intervention=intervention,branches=result['branches']),indent=2),flush=True)
    finally:env.close()


if __name__=='__main__':main()
