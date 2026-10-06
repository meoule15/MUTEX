"""Repeat a failed prospective ranking and separate conditioned/reference terms."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import torch
import evaluate_modality_sufficiency as setup
from rank_completion_candidates import rank_candidates


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--task',type=int,default=4);ap.add_argument('--state',type=int,default=49)
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);setup.logging.set_verbosity_error();setup.control_seed(7)
    cfg=setup.EasyDict(json.load(open('mutex_pretrained/config.json')))
    cfg.device=args.device;cfg.num_gpus=1;cfg.folder='/data/mehul/pmi_robot/datasets'
    cfg.policy.task_spec_modalities='gl_vid';cfg.train.use_augmentation=False;cfg.recalculate_ts_embs=False
    for k in ['add_mim','add_mgm','add_mrm','add_mfm','add_maim','add_magm']:cfg.policy[k]=False
    for k in ['inst','gl','ai','ag','img']:
        cfg.policy.projection_layer.network_kwargs[k+'_transform_kwargs'].network_kwargs.add_cross_modal_layer=True
    policy=setup.BCMutexPolicy(cfg,cfg.shape_meta);policy.eval()
    state,_,_=setup.torch_load_model('mutex_pretrained/models/mutex_weights.pth',device='cpu')
    policy.load_state_dict(state,strict=True);del state;policy.to(args.device)
    bm=setup.benchmark.get_benchmark_dict()['libero_100'](0)
    with open(cfg.folder+'/libero_100/task_spec/gl_openai_clip-vit-large-patch14_ts_mode_eval_emb.pt','rb') as f:emb,_=pickle.load(f)
    bm.set_gl_embs(emb)
    bm.set_visual_task_specifications(setup.get_visual_specifications_all('LIBERO_100',
        [bm.get_task(i).name for i in range(100)],[bm.get_task_demonstration(i) for i in range(100)],cfg,mode='eval'))
    provider=setup.TaskSpecProvider(bm,policy,args.device,cfg.policy.num_task_frames)
    provider.prepare([args.task]+list(range(6,100)),['gl','vid'])
    banks={m:torch.stack([provider.get(t,m,s) for t in range(6,100) for s in range(3)]) for m in ['gl','vid']}
    setup.ObsUtils.initialize_obs_utils_with_obs_specs({'obs':{'rgb':['agentview_rgb','eye_in_hand_rgb'],'low_dim':['gripper_states','joint_states']}})
    cases=np.load(args.directory/f'task{args.task}-gl-actions.npz')['cases'].tolist()
    index=cases.index([args.state,0]);data=np.load(args.directory/f'task{args.task}-gl-initial_obs.npz')
    obs={k:data[k][index] for k in data.files}
    original=json.loads((args.directory/f'task{args.task}-state{args.state}-ranking.json').read_text())
    assert setup.observation_hash(obs)==original['initial_observation_sha256']
    runs=[]
    for samples,seed in [(512,7)]+[(n,s) for n in [512,4096] for s in [13,29,47]]:
        result=rank_candidates(policy,obs,cfg,provider,args.task,args.state,banks,samples,seed)
        result['scoring_seed']=seed;runs.append(result)
        if seed==7:
            for a,b in zip(result['candidates'],original['candidates']):
                for ref in a['scores']:assert abs(a['scores'][ref]['mean']-b['scores'][ref]['mean'])<1e-4
        (args.output/'repeated_rankings.json').write_text(json.dumps(runs,indent=2))
        print('DIAGNOSED',samples,seed,flush=True)
    (args.output/'score_complete.json').write_text(json.dumps(dict(original_reproduced=True,runs=len(runs))))


if __name__=='__main__':main()
