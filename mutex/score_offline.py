"""Offline log-likelihood scoring over demonstration windows.

Infrastructure only: it runs the frozen policy over demonstration data once per
requested specification and writes the raw per-step log-likelihoods through a
LogProbRecorder. It computes no score and subtracts nothing -- assembling a
pointwise mutual information from the resulting rows is a separate step.

Mirrors eval.py's configuration handling: the model's own cfg comes from the JSON
written at training time, with eval-time fields copied on top.

    python mutex/score_offline.py \
        benchmark_name=LIBERO_100 \
        folder=<dataset-path> \
        experiment_dir=mutex_pretrained \
        model_name=mutex_weights.pth \
        eval_modality_set=gl,inst,img,vid \
        record_logprobs=True
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import time

import hydra
import torch
from easydict import EasyDict
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import robomimic.utils.tensor_utils as TensorUtils
from libero.libero import benchmark as bm

from mutex.algos import Multitask
from mutex.logprob_recorder import LogProbRecorder
from mutex.utils import control_seed, safe_device, torch_load_model


@hydra.main(config_path="../configs/eval", config_name="eval_only", version_base=None)
def main(eval_cfg):
    with open(os.path.join(eval_cfg.experiment_dir, "config.json"), "r") as f:
        cfg = EasyDict(json.load(f))

    cfg.folder = eval_cfg.folder
    cfg.benchmark_name = eval_cfg.benchmark_name
    cfg.device = eval_cfg.device
    cfg.num_gpus = 1
    ## Both scoring passes must see identical pixels, so no augmentation.
    cfg.train.use_augmentation = False
    ## Masked-modeling heads are irrelevant to scoring and add loss terms.
    for head in ("add_mim", "add_mgm", "add_mrm", "add_mfm", "add_maim", "add_magm", "add_mlm"):
        if head in cfg.policy:
            cfg.policy[head] = False

    from mutex.eval import parse_modality_sets
    modality_sets = parse_modality_sets(
            eval_cfg.eval_modality_set or cfg.policy.task_spec_modalities)
    canonical = cfg.policy.task_spec_modalities.split('_')
    requested = {m for s in modality_sets for m in s.split('_')}
    assert requested <= set(canonical), \
            f"unknown modalities {requested - set(canonical)}; policy provides {canonical}"
    ## Superset available at runtime; the per-call argument selects from it.
    cfg.policy.task_spec_modalities = '_'.join(m for m in canonical if m in requested)

    control_seed(cfg.seed)
    benchmark = bm.get_benchmark_dict()[cfg.benchmark_name.lower()]()

    raise SystemExit(
        "score_offline: dataset construction is intentionally left to the caller.\n"
        "Build the MLMTaskDataset list exactly as main_masked_modeling.py does, then call\n"
        "score_datasets(cfg, algo, datasets, modality_sets, recorder, spec_indices).\n"
        "Datasets and pretrained weights are not present in this checkout yet.")


def score_datasets(cfg, algo, datasets, modality_sets, recorder,
                   spec_indices=(0,), batch_size=16, num_workers=0):
    """Run the frozen policy over each dataset once per (specification, variant).

    datasets: one MLMTaskDataset per task, indexed by task id.
    modality_sets: e.g. ['gl', 'inst', 'vid'] -- each scored in its own pass.
    spec_indices: which specification variants to pin, via set_task_spec_id.
    """
    algo.eval()
    algo.policy.set_logprob_recorder(recorder)
    try:
        for task_id, dataset in enumerate(datasets):
            dataset.deterministic_frames = True   ## match eval.py's uniform frame sampling
            for spec_index in spec_indices:
                dataset.set_task_spec_id(spec_index)
                loader = DataLoader(dataset, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=False)
                for batch_index, data in enumerate(loader):
                    data = TensorUtils.map_tensor(
                            data, lambda x: safe_device(x, device=cfg.device))
                    for modality in modality_sets:
                        recorder.set_context(
                                task_id=task_id,
                                spec_index=spec_index,
                                batch_offset=batch_index * batch_size)
                        with torch.no_grad():
                            algo.policy(data, modalities=modality)
            dataset.set_task_spec_id(None)
    finally:
        algo.policy.set_logprob_recorder(None)
        recorder.flush()
    return recorder


if __name__ == "__main__":
    main()
