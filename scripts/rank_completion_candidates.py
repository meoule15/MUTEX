"""Prospective initial-state expected-lift rankings, without rollout outcomes."""
import math
import torch
from mutex.metric import raw_obs_to_tensor_obs
from mutex.pmi import batched_action_log_probs, log_mean_probability


@torch.no_grad()
def rank_candidates(policy, obs, cfg, provider, task, state_id, banks, samples, seed):
    devices=[torch.device(cfg.device).index or 0] if cfg.device.startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        policy.reset()
        data=raw_obs_to_tensor_obs([obs],provider.get(task,'gl',0).unsqueeze(0),cfg)
        latents=policy._encode_obs_step(data)
        assert len(policy.latent_queue)==1
        candidates=[]; draws=[]; conditioned=[]; blanks=[]
        for modality in ['gl','vid']:
            for spec in range(3):
                emb=provider.get(task,modality,spec).unsqueeze(0)
                dist=policy._action_dist_from_latents(latents,emb)
                torch.manual_seed(seed*10000000+task*10000+state_id*100+len(candidates)+500000)
                action=dist.sample((samples,))
                blank=policy._action_dist_from_latents(latents,provider.get_blank(task,modality,spec).unsqueeze(0))
                candidates.append(dict(modality=modality,spec_index=spec))
                draws.append(action);conditioned.append(dist.log_prob(action)[:,0]);blanks.append(blank.log_prob(action)[:,0])
        actions=torch.cat(draws,dim=0)
        refs={m:log_mean_probability(batched_action_log_probs(policy,latents,bank,actions,batch_size=16),dim=1)
              for m,bank in banks.items()}
        shared=torch.logaddexp(refs['gl'],refs['vid'])-math.log(2)
        rows=[]
        for i,candidate in enumerate(candidates):
            part=slice(i*samples,(i+1)*samples)
            estimates={}
            for key,reference in [('blank',blanks[i]),('full_modality',refs[candidate['modality']][part]),('full_shared',shared[part])]:
                lift=conditioned[i]-reference
                assert torch.isfinite(lift).all()
                estimates[key]=dict(mean=float(lift.mean()),se=float(lift.std(unbiased=True)/math.sqrt(samples)))
            rows.append(dict(**candidate,scores=estimates,
                conditioned_logp_mean=float(conditioned[i].mean()),entropy_estimate=float(-conditioned[i].mean()),
                reference_logp_means=dict(blank=float(blanks[i].mean()),full_modality=float(refs[candidate['modality']][part].mean()),full_shared=float(shared[part].mean())),
                sampled_action_mean=draws[i][:,0].mean(0).cpu().tolist(),
                sampled_action_std=draws[i][:,0].std(0).cpu().tolist()))
        assert len(policy.latent_queue)==1
        policy.reset()
        return dict(task_id=task,initial_state_index=state_id,samples_per_candidate=samples,
                    candidates=rows,scored_before_rollouts=True,history_observations=1)
