"""Matched-action specification lift for a frozen policy.

Blank-reference lift is a likelihood ratio, not automatically a true PMI.
"""
from collections import OrderedDict
import math
import random
import torch


CONTENT_KEYS = {'gl_emb', 'inst_emb', 'vid_spec', 'img_spec',
                'ai_task_spec', 'ag_task_spec'}


def log_mean_probability(log_probs, dim=0):
    """Uniform probability mixture, evaluated stably in log space."""
    if log_probs.shape[dim] == 0 or not torch.isfinite(log_probs).all():
        raise ValueError('Expected a nonempty set of finite log probabilities')
    return torch.logsumexp(log_probs, dim=dim) - math.log(log_probs.shape[dim])


@torch.no_grad()
def batched_action_log_probs(policy, latents, embeddings, action, batch_size=32):
    """Score one history/action against an embedding bank without advancing history.

The supplied latents represent a single observation history. Chunking only batches
independent specification branches, so this requires a policy in evaluation mode.
"""
    if policy.training:
        raise ValueError('Batched reference scoring requires evaluation mode')
    if (batch_size < 1 or latents.shape[0] != 1 or action.ndim not in (2, 3) or
            action.shape[-2] != 1 or not len(embeddings)):
        raise ValueError('Expected one history/action and a nonempty embedding bank')
    values = []
    for start in range(0, len(embeddings), batch_size):
        emb = embeddings[start:start + batch_size]
        x = latents.repeat(len(emb), *([1] * (latents.ndim - 1)))
        dist = policy._action_dist_from_latents(x, emb)
        values.append(dist.log_prob(action.expand(*action.shape[:-2], len(emb), -1)))
    result = torch.cat(values, dim=-1)
    if result.shape != action.shape[:-2] + (len(embeddings),) or not torch.isfinite(result).all():
        raise ValueError('Expected one finite log probability per specification')
    return result


def blank_specification(data):
    """Remove cached content before encoding; preserve masks and token layout.

Learned projection biases, temporal tokens and modality tokens remain. This is
a zero-content intervention, not a claim that the policy was trained on nulls.
Tokenized text must first be encoded to gl_emb/inst_emb.
"""
    if any(k.endswith('_tokens') for k in data):
        raise ValueError('Blank references require cached content embeddings')
    if not CONTENT_KEYS.intersection(data):
        raise ValueError('No specification content found')
    return {k: torch.zeros_like(v) if k in CONTENT_KEYS else v
            for k, v in data.items()}


def action_lift(conditioned, references, action):
    """Score one action under a conditioned distribution and uniform mixture.

references must be a nonempty sequence of distributions. Averaging probabilities
in log space is deliberately different from averaging log probabilities.
"""
    if not references:
        raise ValueError('At least one reference distribution is required')
    full = conditioned.log_prob(action)
    ref = log_mean_probability(torch.stack([d.log_prob(action) for d in references]))
    if not (torch.isfinite(full).all() and torch.isfinite(ref).all()):
        raise ValueError('Nonfinite action log probability')
    return {'conditioned_logp': full, 'reference_logp': ref, 'lift': full - ref}


class ActionLiftScorer:
    """Advance history once and evaluate every candidate on the same action."""
    def __init__(self, policy):
        self.policy = policy

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def score_step(self, data, action, candidates, references, reference_keys=None):
        if not candidates or not references:
            raise ValueError('Candidates and references must be nonempty')
        if reference_keys is None:
            reference_keys = {k: list(references) for k in candidates}
        if set(reference_keys) != set(candidates) or any(
                not keys or any(k not in references for k in keys)
                for keys in reference_keys.values()):
            raise ValueError('Each candidate must name valid, nonempty references')
        embeddings = OrderedDict()
        for key, emb in candidates.items():
            embeddings['candidate:' + key] = emb
        for key, emb in references.items():
            embeddings['reference:' + key] = emb
        dists = self.policy.get_action_dists(data, embeddings)
        return OrderedDict((k, action_lift(dists['candidate:' + k],
                           [dists['reference:' + r] for r in reference_keys[k]], action))
                           for k in candidates)


def expected_lift(conditioned, references, num_samples=32):
    """Monte Carlo KL to the reference and differential entropy, per batch item.

Estimates can be negative from sampling noise. Never clamp them. Callers should
isolate the scoring RNG so diagnostics do not change executed actions.
"""
    if num_samples < 2:
        raise ValueError('At least two samples are required for standard errors')
    samples = conditioned.sample((num_samples,))
    scores = action_lift(conditioned, references, samples)
    lift = scores['lift']
    return {'expected_lift': lift.mean(0),
            'expected_lift_se': lift.std(0, unbiased=True) / math.sqrt(num_samples),
            'entropy': -scores['conditioned_logp'].mean(0),
            'num_samples': num_samples}


def rank_with_abstention(scores, uncertainty=2.0, margin=0.0):
    """Per-environment score ranking; None means the top-two gap is unresolved.

This uncertainty concerns Monte Carlo scores, never completion probability.
"""
    if not scores or not math.isfinite(uncertainty) or uncertainty<0 or not math.isfinite(margin) or margin<0:
        raise ValueError('Invalid ranking configuration')
    keys=list(scores);shape=scores[keys[0]]['expected_lift'].shape
    if len(shape)!=1:raise ValueError('Expected per-environment score vectors')
    for result in scores.values():
        for field in ['expected_lift','expected_lift_se']:
            if result[field].shape!=shape or not torch.isfinite(result[field]).all():
                raise ValueError('Expected finite, equally shaped score vectors')
        if (result['expected_lift_se']<0).any():raise ValueError('Negative sampling error')
    results=[]
    for i in range(shape[0]):
        ordered=sorted(keys,key=lambda k:scores[k]['expected_lift'][i].item(),reverse=True)
        top=ordered[0];runner=ordered[1] if len(ordered)>1 else None
        gap=(scores[top]['expected_lift'][i]-scores[runner]['expected_lift'][i]).item() if runner else None
        error=math.hypot(scores[top]['expected_lift_se'][i].item(),scores[runner]['expected_lift_se'][i].item()) if runner else None
        resolved=runner is None or gap>margin+uncertainty*error
        results.append(dict(choice=top if resolved else None,top_candidate=top,runner_up=runner,
                            gap=gap,combined_se=error,status='resolved' if resolved else 'abstained'))
    return results


class SpecificationSelector:
    """Experimental expected-lift selection with conservative switching.

Each candidate has its own blank reference. Scores measure sensitivity to that
reference, not task success. Tune interval, margin and uncertainty multiplier on
separate validation tasks before drawing performance conclusions.
"""
    def __init__(self, keys, batch_size, interval=10, margin=0.5, uncertainty=2.0,initial_key=None):
        if (not keys or len(set(keys)) != len(keys) or batch_size < 1 or
                interval < 1 or not math.isfinite(margin) or margin < 0 or
                not math.isfinite(uncertainty) or uncertainty < 0):
            raise ValueError('Invalid selector configuration')
        self.keys = list(keys)
        if initial_key is not None and initial_key not in self.keys:raise ValueError('Unknown initial specification')
        self.current = [initial_key or self.keys[0]] * batch_size
        self.last_decisions=[]
        self.interval, self.margin, self.uncertainty = interval, margin, uncertainty

    def due(self, step):
        return step >= 1 and (step - 1) % self.interval == 0

    def update(self, step, scores):
        if not self.due(step):
            return [False] * len(self.current)
        if set(scores) != set(self.keys):
            raise ValueError('Scores must cover every candidate')
        for result in scores.values():
            for field in ['expected_lift', 'expected_lift_se']:
                value = result[field]
                if value.shape != (len(self.current),) or not torch.isfinite(value).all():
                    raise ValueError('Expected finite per-environment scores')
            if (result['expected_lift_se'] < 0).any():
                raise ValueError('Standard errors must be nonnegative')
        self.last_decisions=rank_with_abstention(scores,self.uncertainty,self.margin)
        switched = []
        for i, current in enumerate(self.current):
            lower = {key: (scores[key]['expected_lift'][i] -
                          self.uncertainty * scores[key]['expected_lift_se'][i]).item()
                     for key in self.keys}
            best = self.last_decisions[i]['choice']
            upper = (scores[current]['expected_lift'][i] +
                     self.uncertainty * scores[current]['expected_lift_se'][i]).item()
            change = best is not None and best != current and lower[best] > upper + self.margin
            if change:
                self.current[i] = best
            self.last_decisions[i]['switched']=change
            self.last_decisions[i]['acting_candidate']=self.current[i]
            switched.append(change)
        return switched


class RandomSpecificationSelector(SpecificationSelector):
    """Uniform baseline at the same decision interval, with an independent RNG."""
    def __init__(self, keys, batch_size, interval=10, seed=0):
        super().__init__(keys, batch_size, interval)
        self.rng = random.Random(seed)

    def update(self, step, scores=None):
        if not self.due(step):
            return [False] * len(self.current)
        previous = self.current[:]
        self.current = [self.rng.choice(self.keys) for _ in previous]
        return [a != b for a, b in zip(previous, self.current)]
