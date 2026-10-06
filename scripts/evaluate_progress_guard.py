"""Fresh paired benchmark validation of abstention plus oracle progress fallback."""
import argparse
import functools
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import random
import time
import numpy as np
import torch
import evaluate_modality_sufficiency as setup
from mutex.pmi import expected_lift,rank_with_abstention
from mutex.progress import GoalProgressMonitor


class ProgressEnv(setup.OffScreenRenderEnv):
    @property
    def goal_predicates(self):return self.env.parsed_problem['goal_state']
    @property
    def current_goal_progress(self):return [bool(self.env._eval_predicate(p)) for p in self.goal_predicates]


def make_env(args,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    return ProgressEnv(**args)


@torch.no_grad()
def initial_ranking(policy,obs,cfg,provider,task,state,seed):
    devices=[torch.device(cfg.device).index or 0] if cfg.device.startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        policy.reset();data=setup.raw_obs_to_tensor_obs([obs],provider.get(task,'gl',0).unsqueeze(0),cfg)
        x=policy._encode_obs_step(data);scores={}
        for i,(modality,spec) in enumerate(itertools.product(['gl','vid'],range(3))):
            dist=policy._action_dist_from_latents(x,provider.get(task,modality,spec).unsqueeze(0))
            blank=policy._action_dist_from_latents(x,provider.get_blank(task,modality,spec).unsqueeze(0))
            torch.manual_seed(seed*10000000+task*10000+state*100+i+500000)
            scores[modality+str(spec)]=expected_lift(dist,[blank],512)
        decision=rank_with_abstention(scores,uncertainty=2.,margin=0.)[0]
        policy.reset()
        return dict(decision=decision,scores={k:{f:float(v[f][0]) for f in ['expected_lift','expected_lift_se']} for k,v in scores.items()})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--seed',type=int,default=11)
    ap.add_argument('--tasks',type=int,nargs='+',default=[1,2,4,5])
    ap.add_argument('--initial-states',type=int,default=3)
    args=ap.parse_args()
    if args.output.exists():ap.error('Choose a new output directory')
    if args.initial_states<1 or len(set(args.tasks))!=len(args.tasks) or any(not 0<=t<100 for t in args.tasks):
        ap.error('Invalid task/state configuration')
    args.output.mkdir(parents=True);mp.set_start_method('spawn',force=True)
    torch.set_num_threads(4);setup.logging.set_verbosity_error();setup.control_seed(args.seed)
    cfg=setup.EasyDict(json.load(open('mutex_pretrained/config.json')))
    cfg.device=args.device;cfg.num_gpus=1;cfg.folder='/data/mehul/pmi_robot/datasets'
    cfg.policy.task_spec_modalities='gl_vid';cfg.train.use_augmentation=False;cfg.recalculate_ts_embs=False
    for k in ['add_mim','add_mgm','add_mrm','add_mfm','add_maim','add_magm']:cfg.policy[k]=False
    for k in ['inst','gl','ai','ag','img']:
        cfg.policy.projection_layer.network_kwargs[k+'_transform_kwargs'].network_kwargs.add_cross_modal_layer=True
    policy=setup.BCMutexPolicy(cfg,cfg.shape_meta);policy.eval()
    weights,_,_=setup.torch_load_model('mutex_pretrained/models/mutex_weights.pth',device='cpu')
    policy.load_state_dict(weights,strict=True);del weights;policy.to(args.device)
    bm=setup.benchmark.get_benchmark_dict()['libero_100'](0)
    with open(cfg.folder+'/libero_100/task_spec/gl_openai_clip-vit-large-patch14_ts_mode_eval_emb.pt','rb') as f:emb,_=pickle.load(f)
    bm.set_gl_embs(emb);bm.set_visual_task_specifications(setup.get_visual_specifications_all('LIBERO_100',
        [bm.get_task(i).name for i in range(100)],[bm.get_task_demonstration(i) for i in range(100)],cfg,mode='eval'))
    provider=setup.TaskSpecProvider(bm,policy,args.device,cfg.policy.num_task_frames);provider.prepare(args.tasks,['gl','vid'])
    setup.ObsUtils.initialize_obs_utils_with_obs_specs({'obs':{'rgb':['agentview_rgb','eye_in_hand_rgb'],'low_dim':['gripper_states','joint_states']}})
    modes=['fixed_language','fixed_video','guarded_pmi','guarded_progress'];tasks=args.tasks
    manifest=dict(tasks=tasks,initial_states=args.initial_states,strategies=modes,seed=args.seed,max_steps=600,pmi_samples=512,
        initial_selection='highest blank lift only if top-two gap > two combined sampling SE; otherwise vid0',
        progress_source='LIBERO privileged goal predicates; benchmark-only, not policy inputs or learned prediction',
        stall_steps=240,regression_patience=5,cooldown_steps=60,max_progress_switches=1,
        fallback='opposite modality variant0, fixed before validation outcomes',
        pairing='identical initial observations/environment seeds; common per-step action sampling seeds; independent histories',
        status='experimental; one fresh seed on previously examined tasks; no production reliability claim')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2));rows=[]
    base=Path('LIBERO/libero/libero')
    for task in tasks:
        info=bm.get_task(task);initial=torch.load(base/'init_files'/info.problem_folder/info.init_states_file)
        if len(initial)<args.initial_states:raise ValueError('Not enough distinct initial states')
        ids=np.linspace(0,len(initial)-1,args.initial_states,dtype=int).tolist();cases=list(itertools.product(ids,modes));count=len(cases)
        seeds=[args.seed*100000+task*1000+s*3 for s,_ in cases]
        env_args=dict(bddl_file_name=str(base/'bddl_files'/info.problem_folder/info.bddl_file),camera_heights=cfg.data.img_h,camera_widths=cfg.data.img_w)
        env=setup.SubprocVectorEnv([functools.partial(make_env,env_args,seed) for seed in seeds])
        try:
            started=time.perf_counter();env.seed(seeds);env.reset();obs=env.set_init_state(np.stack([initial[s] for s,_ in cases]))
            for _ in range(5):obs,_,_,_=env.step(np.zeros((count,7)))
            hashes=[setup.observation_hash(o) for o in obs]
            for s in ids:assert len({hashes[i] for i,c in enumerate(cases) if c[0]==s})==1
            np.savez_compressed(args.output/f'task{task}-initial_obs.npz',**{k:np.stack([o[k] for o in obs]) for k in ['agentview_image','robot0_eye_in_hand_image','robot0_joint_pos','robot0_gripper_qpos']})
            names=env.get_env_attr('goal_predicates');progress=env.get_env_attr('current_goal_progress')
            assert not any(all(p) for p in progress)
            monitors=[GoalProgressMonitor(len(p)) for p in progress]
            events=[[monitors[i].update(0,p)] for i,p in enumerate(progress)]
            current=[];rankings={};initial_choices=[]
            for i,(s,mode) in enumerate(cases):
                if mode.startswith('guarded_'):
                    if str(s) not in rankings:
                        ranking=initial_ranking(policy,obs[i],cfg,provider,task,s,args.seed)
                        ranking['initial_observation_sha256']=hashes[i];rankings[str(s)]=ranking
                    ranking=rankings[str(s)]
                    current.append(ranking['decision']['choice'] or 'vid0')
                else:current.append('gl0' if mode=='fixed_language' else 'vid0')
            initial_choices=list(current)
            (args.output/f'task{task}-initial_rankings.json').write_text(json.dumps(rankings,indent=2))
            policy.reset();success=np.zeros(count,dtype=bool);first=np.full(count,-1,dtype=int);switches=np.zeros(count,dtype=int)
            action_trace=[];spec_trace=[]
            for step in range(1,601):
                spec_trace.append(list(current));data=setup.raw_obs_to_tensor_obs(obs,provider.get(task,'gl',0).unsqueeze(0).repeat(count,1,1),cfg)
                with torch.no_grad():
                    x=policy._encode_obs_step(data);individual={}
                    for modality in ['gl','vid']:
                        indices=[i for i,k in enumerate(current) if k.startswith(modality)]
                        if not indices:continue
                        embeds=torch.stack([provider.get(task,modality,int(current[i][-1])) for i in indices])
                        dist=policy._action_dist_from_latents(x[indices],embeds);component=dist.component_distribution.base_dist
                        for j,i in enumerate(indices):
                            individual[i]=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=dist.mixture_distribution.logits[j]),
                                torch.distributions.Independent(torch.distributions.Normal(component.loc[j],component.scale[j]),1))
                    devices=[torch.device(args.device).index or 0] if args.device.startswith('cuda') else []
                    with torch.random.fork_rng(devices=devices):
                        actions=[]
                        for i,(s,_) in enumerate(cases):
                            torch.manual_seed(args.seed*10000000+task*10000+step+s*100000)
                            actions.append(individual[i].sample())
                        actions=torch.stack(actions).cpu().numpy()
                assert actions.shape==(count,7) and np.isfinite(actions).all();actions[success]=0
                obs,_,done,_=env.step(actions);action_trace.append(actions.copy())
                progress=env.get_env_attr('current_goal_progress')
                for i,p in enumerate(progress):
                    if success[i]:continue
                    event=monitors[i].update(step,p)
                    assert event['joint_complete']==bool(done[i])
                    if event['gained'] or event['reassess'] or p!=events[i][-1]['current_predicates'] or event['joint_complete']:events[i].append(event)
                    if done[i]:success[i]=True;first[i]=step
                    elif step<600 and cases[i][1]=='guarded_progress' and event['reassess'] and switches[i]==0:
                        old=current[i];current[i]='gl0' if old.startswith('vid') else 'vid0';switches[i]+=1
                        event['fallback_from']=old;event['fallback_to']=current[i];event['effective_step']=step+1
                if success.all():break
            np.savez_compressed(args.output/f'task{task}-actions.npz',actions=np.stack(action_trace),specifications=np.array(spec_trace),success=success,success_step=first)
            for i,(s,mode) in enumerate(cases):
                rows.append(dict(task_id=task,task_name=info.name,initial_state_index=s,strategy=mode,
                    success=bool(success[i]),success_step=int(first[i]),initial_choice=initial_choices[i],final_choice=current[i],
                    progress_switches=int(switches[i]),environment_seed=seeds[i],initial_observation_sha256=hashes[i],goal_predicates=names[i],events=events[i]))
            (args.output/'episodes.json').write_text(json.dumps(rows,indent=2))
            print('COMPLETED',task,{m:int(sum(r['success'] for r in rows if r['task_id']==task and r['strategy']==m)) for m in modes},'seconds',round(time.perf_counter()-started,1),flush=True)
        finally:env.close()
    expected=len(tasks)*args.initial_states*len(modes)
    assert len(rows)==expected
    (args.output/'complete.json').write_text(json.dumps(dict(episodes=expected,matched_initial_observations=True)))


if __name__=='__main__':main()
