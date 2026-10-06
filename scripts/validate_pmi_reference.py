"""Audit empirical reference mixtures on matched LIBERO demonstration actions.

Run from MUTEX/: python scripts/validate_pmi_reference.py --output ../results/reference-validation
Collect likelihoods once, then compare nested independent pools without replaying
the policy. Results validate an empirical prior, not a true unconditional policy.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
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
from mutex.pmi import batched_action_log_probs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--dataset', type=Path, default=Path('/data/mehul/pmi_robot/datasets'))
    ap.add_argument('--prefix-steps', type=int, default=100)
    ap.add_argument('--sample-steps', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--mc-samples', type=int, default=0,
                    help='Also evaluate expected lift with this many samples per candidate')
    args=ap.parse_args()
    if args.output.exists(): ap.error('Choose a new output directory')
    if min(args.prefix_steps,args.sample_steps,args.batch_size)<1: ap.error('Counts must be positive')
    if args.mc_samples not in (0,) and args.mc_samples<2: ap.error('MC samples must be zero or >=2')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4); logging.set_verbosity_error(); control_seed(0)
    cfg=EasyDict(json.load(open('mutex_pretrained/config.json')))
    cfg.device='cpu'; cfg.num_gpus=1; cfg.folder=str(args.dataset.resolve())
    cfg.policy.task_spec_modalities='gl_vid'; cfg.train.use_augmentation=False
    cfg.recalculate_ts_embs=False
    for key in ['add_mim','add_mgm','add_mrm','add_mfm','add_maim','add_magm']: cfg.policy[key]=False
    for key in ['inst','gl','ai','ag','img']:
        cfg.policy.projection_layer.network_kwargs[key+'_transform_kwargs'].network_kwargs.add_cross_modal_layer=True
    policy=BCMutexPolicy(cfg,cfg.shape_meta); policy.eval()
    checkpoint=Path('mutex_pretrained/models/mutex_weights.pth')
    state,_,_=torch_load_model(str(checkpoint),device='cpu')
    policy.load_state_dict(state,strict=True); del state
    bm=benchmark.get_benchmark_dict()['libero_100'](0)
    with (args.dataset/'libero_100/task_spec/gl_openai_clip-vit-large-patch14_ts_mode_eval_emb.pt').open('rb') as f:
        embeddings,_=pickle.load(f)
    assert embeddings.shape[:2]==(100,3)
    bm.set_gl_embs(embeddings)
    bm.set_visual_task_specifications(get_visual_specifications_all(
        'LIBERO_100',[bm.get_task(i).name for i in range(bm.n_tasks)],
        [bm.get_task_demonstration(i) for i in range(bm.n_tasks)],cfg,mode='eval'))
    provider=TaskSpecProvider(bm,policy,'cpu',cfg.policy.num_task_frames)
    provider.prepare(range(100),['gl','vid'])
    banks={}
    for modality in ['gl','vid']:
        entries=[provider.get(task,modality,spec) for task in range(100) for spec in range(3)]
        entries += [provider.get_blank(task,modality,0) for task in range(6)]
        banks[modality]=torch.stack(entries)
    ObsUtils.initialize_obs_utils_with_obs_specs({'obs': {
        'rgb':['agentview_rgb','eye_in_hand_rgb'],'low_dim':['gripper_states','joint_states']}})
    eligible=list(range(6,100))
    pools={str(seed):np.random.default_rng(seed).permutation(eligible).tolist() for seed in [0,1,2]}
    manifest=dict(target_tasks=list(range(6)),reference_tasks=eligible,modalities=['gl','vid'],
                  variants=[0,1,2],episodes=['demo_0','demo_1'],prefix_steps=args.prefix_steps,
                  sample_steps=args.sample_steps,pool_sizes=[2,8,16,32,64,94],pool_permutations=pools,
                  mc_samples=args.mc_samples,
                  prior='uniform tasks, uniform held-out specification variants, separate modalities',
                  stability_gate={'mean_absolute_reference_logp_error_nats':0.5,
                                  'step_ranking_agreement':0.9,'task_ranking_agreement':5/6},
                  note='Seen training tasks and demonstration prefixes. The 94-task mixture is an empirical anchor, not ground truth.')
    h=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    manifest['checkpoint_sha256']=h.hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    frames=[]; matrices=[]; blanks=[]; batch_checks=[]
    mc_matrices=[]; mc_conditioned=[]; mc_blanks=[]
    for task in range(6):
        with h5py.File(args.dataset/bm.get_task_demonstration(task),'r') as f:
            for episode in ['demo_0','demo_1']:
                policy.reset(); ep=f['data'][episode]
                length=min(args.prefix_steps,len(ep['actions']))
                selected=set(np.linspace(0,length-1,min(args.sample_steps,length),dtype=int).tolist())
                for step in range(length):
                    obs={key:ObsUtils.process_obs(torch.from_numpy(ep['obs'][key][step]),obs_key=key).float().unsqueeze(0)
                         for key in cfg.shape_meta.all_obs_keys}
                    with torch.no_grad(): latents=policy._encode_obs_step({'obs':obs})
                    if step not in selected: continue
                    action=torch.from_numpy(ep['actions'][step]).float().unsqueeze(0)
                    scored_actions=action
                    if args.mc_samples:
                        samples=[]; own=[]
                        with torch.random.fork_rng(devices=[]), torch.no_grad():
                            torch.manual_seed(task*10000+int(episode[-1])*1000+step)
                            for modality in ['gl','vid']:
                                dist=policy._action_dist_from_latents(latents,banks[modality][task*3:task*3+1])
                                sample=dist.sample((args.mc_samples,))
                                samples.append(sample); own.append(dist.log_prob(sample).squeeze(-1).numpy())
                        scored_actions=torch.cat([action.unsqueeze(0)]+samples,dim=0)
                        mc_conditioned.append(np.stack(own))
                    modalities=[]; nulls=[]
                    mc_modalities=[]; mc_nulls=[]
                    history=len(policy.latent_queue)
                    for modality in ['gl','vid']:
                        if len(batch_checks)<2:
                            one=batched_action_log_probs(policy,latents,banks[modality][:4],action,1)
                            batch=batched_action_log_probs(policy,latents,banks[modality][:4],action,4)
                            torch.testing.assert_close(one,batch,rtol=1e-5,atol=1e-3)
                            batch_checks.append(dict(modality=modality,max_absolute_difference=(one-batch).abs().max().item()))
                        values=batched_action_log_probs(policy,latents,banks[modality],scored_actions,args.batch_size)
                        if args.mc_samples:
                            mc_modalities.append(values[1:,:300].reshape(2,args.mc_samples,100,3).numpy())
                            mc_nulls.append(values[1:,300+task].reshape(2,args.mc_samples).numpy())
                            values=values[0]
                        modalities.append(values[:300].reshape(100,3).numpy())
                        nulls.append(values[300+task].item())
                    assert len(policy.latent_queue)==history
                    frames.append(dict(task_id=task,episode=episode,step=step,action=action[0].tolist()))
                    matrices.append(np.stack(modalities)); blanks.append(nulls)
                    if args.mc_samples:
                        mc_matrices.append(np.stack(mc_modalities,axis=2))
                        mc_blanks.append(np.stack(mc_nulls,axis=2))
                print('COMPLETED',task,episode,'scored steps',len(selected),flush=True)
        # Save completed task data for inspection/reanalysis even if interrupted.
        extra={}
        if args.mc_samples:
            extra=dict(mc_logp=np.stack(mc_matrices),mc_conditioned=np.stack(mc_conditioned),mc_blank=np.stack(mc_blanks))
        np.savez_compressed(args.output/'likelihoods.npz',logp=np.stack(matrices),blank_logp=np.array(blanks),**extra)
        (args.output/'frames.json').write_text(json.dumps(frames))
    (args.output/'checks.json').write_text(json.dumps(dict(batch_checks=batch_checks,frames=len(frames),
                     finite=bool(np.isfinite(matrices).all()),history_preserved=True),indent=2))
    (args.output/'complete.json').write_text(json.dumps({'status':'complete','frames':len(frames)}))


if __name__=='__main__': main()
