"""Runtime delivery of task-specification embeddings.

MUTEX bakes the specification modality into cfg.policy.task_spec_modalities at
construction time, and the rollout path (BCMutexPolicy.get_action) consumes a
task embedding that was computed once before the episode. This module makes the
modality a runtime choice instead, so a caller can select or switch
specifications per step.

Two delivery modes, both encoding from the same constructor so their inputs are
byte-identical:

  'bank'    precompute every (task, modality) embedding up front and index it.
  'reembed' rebuild on demand from the raw specifications the benchmark holds.

Deliberately standalone rather than attached to the benchmark (which lives in the
LIBERO submodule and stores task embeddings as one whole tensor with no
per-modality slot) or to the policy (an nn.Module loaded with strict=True, where
extra tensor state invites state_dict and device surprises).
"""

import torch

from mutex.models.policy.bc_mutex_policy import resolve_modalities
from mutex.utils import sample_frames


def build_spec_data_dict(benchmark, task_id, modalities, device,
                         num_task_frames, spec_index=None):
    """Assemble the raw specification tensors get_task_embs expects.

    Extracted verbatim from eval.bm_set_task_embs so the bank and re-embed paths
    share one constructor.

    spec_index: when given, keep only that specification variant (dimension 0 of
    each stacked tensor); when None, keep all of them.
    """
    modalities = resolve_modalities(modalities, [])
    data_dict = {}

    visual_spec = benchmark.get_visual_task_specification(task_id)
    if 'vid' in modalities:
        vid_spec_list, vid_spec_mask_list = [], []
        for idx in range(len(visual_spec['vid_task_spec'])):
            vid_task_spec = visual_spec['vid_task_spec'][idx]
            vid_task_spec_mask = visual_spec['vid_task_spec_mask'][idx]
            frame_idx = sample_frames(
                    num_frames=min(num_task_frames - 1, vid_task_spec.shape[0] - 1),
                    vlen=vid_task_spec.shape[0] - 1,
                    sample='uniform') ## uniform sampling for evaluation
            frame_idx.append(vid_task_spec.shape[0] - 1)

            vid_spec_list.append(vid_task_spec[frame_idx].to(device))
            vid_spec_mask_list.append(vid_task_spec_mask[frame_idx].to(device))
        data_dict['vid_spec'] = torch.stack(vid_spec_list, dim=0)  # [num_eval_ts,num_frames,E]
        data_dict['vid_spec_mask'] = torch.stack(vid_spec_mask_list, dim=0)

    if 'img' in modalities:
        data_dict['img_spec'] = torch.stack(visual_spec['img_task_spec'], dim=0).to(device)
        data_dict['img_spec_mask'] = None

    if 'inst' in modalities:
        data_dict['inst_emb'] = benchmark.get_inst_emb(task_id).to(device)
        data_dict['inst_emb_mask'] = torch.ones(
                data_dict['inst_emb'].shape[:-1]).to(device)

    if 'gl' in modalities:
        ## adding time dimension
        data_dict['gl_emb'] = benchmark.get_gl_emb(task_id).unsqueeze(dim=1).to(device)

    if 'ai' in modalities:
        ai_task_spec = benchmark.get_ai_task_spec(task_id)
        data_dict['ai_task_spec'] = ai_task_spec['ai_task_spec'].to(device)
        data_dict['ai_task_spec_mask'] = ai_task_spec['ai_task_spec_mask'].to(device)

    if 'ag' in modalities:
        ag_task_spec = benchmark.get_ag_task_spec(task_id)
        data_dict['ag_task_spec'] = ag_task_spec['ag_task_spec'].to(device)
        data_dict['ag_task_spec_mask'] = ag_task_spec['ag_task_spec_mask'].to(device)

    if spec_index is not None:
        data_dict = {
            k: (v[spec_index:spec_index + 1] if torch.is_tensor(v) else v)
            for k, v in data_dict.items()
        }
    return data_dict


class TaskSpecProvider:
    """Serves task-specification embeddings by (task, modality, spec index)."""

    BANK = 'bank'
    REEMBED = 'reembed'

    def __init__(self, benchmark, policy, device, num_task_frames,
                 mode=BANK, cache_reembed=True):
        if mode not in (self.BANK, self.REEMBED):
            raise ValueError(f"mode must be '{self.BANK}' or '{self.REEMBED}', got {mode!r}")
        self.benchmark = benchmark
        self.policy = policy
        self.device = device
        self.num_task_frames = num_task_frames
        self.mode = mode
        self.cache_reembed = cache_reembed
        self._store = {}   # (task_id, key) -> [num_eval_ts, T, E] on CPU
        self._cache = {}   # (task_id, key, spec_index) -> [T, E] on CPU

    # -- construction ----------------------------------------------------

    def prepare(self, task_ids, modality_keys):
        """Precompute embeddings for every (task, modality). No-op in reembed mode."""
        if self.mode != self.BANK:
            return self
        for task_id in task_ids:
            for key in modality_keys:
                self._store[(task_id, key)] = self._encode(task_id, key, None).cpu()
        return self

    def _encode(self, task_id, modality_key, spec_index):
        data_dict = build_spec_data_dict(
                benchmark=self.benchmark,
                task_id=task_id,
                modalities=modality_key,
                device=self.device,
                num_task_frames=self.num_task_frames,
                spec_index=spec_index)
        with torch.no_grad():
            emb, *_ = self.policy.get_task_embs(data_dict, modalities=modality_key)
        return emb

    # -- access ----------------------------------------------------------

    def get(self, task_id, modality_key, spec_index):
        """Return one task embedding, shaped [T, E], on the provider's device."""
        if self.mode == self.BANK:
            key = (task_id, modality_key)
            if key not in self._store:
                self._store[key] = self._encode(task_id, modality_key, None).cpu()
            return self._store[key][spec_index].to(self.device)

        cache_key = (task_id, modality_key, spec_index)
        if self.cache_reembed and cache_key in self._cache:
            return self._cache[cache_key].to(self.device)
        ## Encode only the requested variant: cheaper, and valid because the
        ## per-modality transform modules have no batch-coupled operations.
        emb = self._encode(task_id, modality_key, spec_index)[0]
        if self.cache_reembed:
            self._cache[cache_key] = emb.cpu()
        return emb.to(self.device)

    def get_many(self, task_id, modality_keys, spec_index):
        """Ordered mapping of modality key -> [T, E], for get_action_dists."""
        from collections import OrderedDict
        return OrderedDict(
                (key, self.get(task_id, key, spec_index)) for key in modality_keys)

    def num_specs(self, task_id, modality_key):
        key = (task_id, modality_key)
        if key not in self._store:
            self._store[key] = self._encode(task_id, modality_key, None).cpu()
        return self._store[key].shape[0]
