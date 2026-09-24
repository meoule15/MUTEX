import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
import json
import numpy as np
import pandas as pd
import pprint
import time
import torch
import cv2
from easydict import EasyDict
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from libero.libero import benchmark as bm
from libero.libero.envs import OffScreenRenderEnv
from mutex.utils import sample_frames
from mutex.algos import Multitask
from mutex.lf_datasets import get_dataset
from mutex.metric import evaluate_multitask_training_success
from mutex.utils import control_seed, safe_device, torch_load_model, \
                        make_dir, confidence_interval
from mutex.embed_utils import get_visual_specifications_all, \
                        get_task_embs, get_audio_specification
from mutex.logprob_recorder import LogProbRecorder, WandbLogProbSink
from mutex.task_spec_provider import TaskSpecProvider, build_spec_data_dict

class EvalLogger:
    def __init__(self, log_keys: list):
        self.log_keys = log_keys
        self._dict = {}
        for key in self.log_keys:
            self._dict.update({key: []})

    def add_kv(self, key, value):
        assert key in self.log_keys, f"Tried to log {key} but logger is initialized with keys {self.log_keys}"
        self._dict[key].append(value)

    def save(self, filename):
        df = pd.DataFrame(self._dict)
        df.to_csv(filename, index=False)

def summary2video(task_ind, task_i, result_summary, eval_cfg, cfg):
    #initiate evaluation envs
    record_h, record_w = 512, 512
    env_args = {
        "bddl_file_name": os.path.join(cfg.bddl_folder, task_i.problem_folder, task_i.bddl_file),
        "camera_heights": record_h,
        "camera_widths": record_w,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(cfg.seed)
    for traj_key in result_summary[task_ind]["sim_states"].keys():
        print(f"Task index {task_ind}, eval_traj {traj_key}, length {len(result_summary[task_ind]['sim_states'][traj_key])}")

        sim_state_traj = result_summary[task_ind]['sim_states'][traj_key]
        imgs = []
        for sim_state in sim_state_traj:
            obs = env.regenerate_obs_from_state(sim_state)
            img = obs["agentview_image"][::-1,:,:]
            imgs.append(img)

        make_dir(os.path.join(eval_cfg.experiment_dir, "videos"))
        video_path = os.path.join(eval_cfg.experiment_dir, "videos", f"summary_taskind{task_ind}_no{traj_key}.avi")
        print(f"---- Saving video: {len(imgs)}\n", video_path)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (record_h, record_w))
        # Write frames to the VideoWriter object
        for frame in imgs:
            out.write(frame)
        # Release the VideoWriter object
        out.release()

    env.close()
    import gc; gc.collect()
    return

def parse_modality_sets(value):
    """Normalise the eval_modality_set option to a list of specification keys.

    Hydra reads a bare comma-separated override as a list, so accept both:
        eval_modality_set=[gl,gl_inst,vid]     -> list
        eval_modality_set='"gl,gl_inst,vid"'   -> string
    """
    if value is None:
        return None
    if isinstance(value, str):
        items = value.split(',')
    else:
        items = list(value)
    return [str(s).strip() for s in items if str(s).strip()]


def bm_set_task_embs(algo, benchmark, eval_spec_modalities, task_range, device):
    new_task_embs = []
    for i in task_range:
        ## Body extracted to mutex.task_spec_provider.build_spec_data_dict so the
        ## runtime provider encodes from byte-identical inputs.
        data_dict = build_spec_data_dict(
                benchmark=benchmark,
                task_id=i,
                modalities=eval_spec_modalities,
                device=device,
                num_task_frames=algo.cfg.policy.num_task_frames)

        emb, *temp = algo.policy.get_task_embs(data_dict, modalities=eval_spec_modalities)
        new_task_embs.append(emb)

    new_task_embs = torch.stack(new_task_embs, dim=0)  # [num_tasks, num_eval_ts, T, E]
    benchmark.set_task_embs(new_task_embs)
    return new_task_embs

@hydra.main(config_path="../configs/eval", config_name="eval_only", version_base=None)
def main(eval_cfg):
    with open(os.path.join(eval_cfg.experiment_dir, "config.json"), "r") as f:
        cfg = json.load(f)

    ## preprocessing
    cfg = EasyDict(cfg)
    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    # control seed
    control_seed(cfg.seed)

    # prepare multitask learning. Overriding from eval_cfg because train and test could be in different machines
    cfg.num_gpus = 1
    cfg.eval.use_mp = eval_cfg.use_mp
    cfg.recalculate_ts_embs = False
    cfg.device = eval_cfg.device
    cfg.folder = to_absolute_path(eval_cfg.folder)
    cfg.bddl_folder = to_absolute_path(eval_cfg.bddl_folder)
    cfg.init_states_folder = to_absolute_path(eval_cfg.init_states_folder)
    cfg.eval.num_workers = eval_cfg.num_workers
    cfg.eval.n_eval = eval_cfg.n_eval
    cfg.experiment_dir = eval_cfg.experiment_dir
    cfg.eval.num_procs = eval_cfg.num_workers
    train_benchmark_name = cfg.benchmark_name
    cfg.benchmark_name = eval_cfg.benchmark_name if eval_cfg.benchmark_name is not None else cfg.benchmark_name
    cfg.pretrain_model_path = [os.path.join(cfg.experiment_dir, 'models', eval_cfg.model_name)]
    if ('cmm_' in eval_cfg.model_name) or ('mutex' in eval_cfg.model_name):
        cfg.policy.add_mim = False
        cfg.policy.add_mgm = False
        cfg.policy.add_mrm = False
        cfg.policy.add_mfm = False
        cfg.policy.add_maim = False
        cfg.policy.add_magm = False

        cfg.policy.projection_layer.network_kwargs.inst_transform_kwargs.network_kwargs.add_cross_modal_layer = True
        cfg.policy.projection_layer.network_kwargs.gl_transform_kwargs.network_kwargs.add_cross_modal_layer = True
        cfg.policy.projection_layer.network_kwargs.ai_transform_kwargs.network_kwargs.add_cross_modal_layer = True
        cfg.policy.projection_layer.network_kwargs.ag_transform_kwargs.network_kwargs.add_cross_modal_layer = True
        cfg.policy.projection_layer.network_kwargs.img_transform_kwargs.network_kwargs.add_cross_modal_layer = True

    ## Runtime specification selection / recording. Defaults leave every path below
    ## exactly as published.
    cfg.eval_modality_set = getattr(eval_cfg, 'eval_modality_set', None)
    cfg.modality_delivery = getattr(eval_cfg, 'modality_delivery', 'bank')
    cfg.record_logprobs = getattr(eval_cfg, 'record_logprobs', False)
    cfg.record_actions = getattr(eval_cfg, 'record_actions', False)
    cfg.record_wandb = getattr(eval_cfg, 'record_wandb', False)
    cfg.wandb_eval_project = getattr(eval_cfg, 'wandb_project', 'mutex-pmi')
    cfg.wandb_eval_mode = getattr(eval_cfg, 'wandb_mode', 'online')
    cfg.wandb_trace_episodes = getattr(eval_cfg, 'wandb_trace_episodes', 2)
    cfg.record_logprobs_dir = getattr(eval_cfg, 'record_logprobs_dir', None) \
            or os.path.join(eval_cfg.experiment_dir, "logprobs")

    if cfg.eval_modality_set is not None:
        ## The candidate sets are selected per call at runtime, so the policy is
        ## constructed with their union: task_spec_modalities becomes the superset
        ## available, not the selection. Canonical ordering keeps cache filenames stable.
        ## Accepts either Hydra list form  eval_modality_set=[gl,gl_inst,vid]
        ## or quoted string form          eval_modality_set='"gl,gl_inst,vid"'
        cfg.eval_modality_sets = parse_modality_sets(cfg.eval_modality_set)
        canonical = cfg.policy.task_spec_modalities.split('_')
        requested = {m for s in cfg.eval_modality_sets for m in s.split('_')}
        assert requested <= set(canonical), \
                f"unknown modalities {requested - set(canonical)}; policy provides {canonical}"
        cfg.policy.task_spec_modalities = '_'.join(m for m in canonical if m in requested)
        cfg.eval_run_tag = 'set-' + '+'.join(cfg.eval_modality_sets)
    else:
        cfg.eval_modality_sets = None
        if eval_cfg.eval_spec_modalities is not None:
            assert all([task_spec in cfg.policy.task_spec_modalities.split('_') for task_spec in eval_cfg.eval_spec_modalities.split('_')])
            cfg.policy.task_spec_modalities = eval_cfg.eval_spec_modalities
        cfg.eval_run_tag = cfg.policy.task_spec_modalities

    prefix = cfg.eval_run_tag + '_'
    if eval_cfg.task_id == -1:
        cfg.eval_csv_filename = os.path.join(eval_cfg.experiment_dir, \
                f"{cfg.benchmark_name}_{prefix}eval_data_ts_{eval_cfg.ts_mode}_{eval_cfg.model_name}.csv")
    else:
        make_dir(os.path.join(eval_cfg.experiment_dir, "logs"))
        cfg.eval_csv_filename = os.path.join(eval_cfg.experiment_dir, \
                f"logs/{cfg.benchmark_name}_{prefix}eval_data_ts_{eval_cfg.ts_mode}_{eval_cfg.model_name}_task_{eval_cfg.task_id:03d}.csv")

    benchmark_dict = bm.get_benchmark_dict()
    benchmark = benchmark_dict[cfg.benchmark_name.lower()]()
    n_manip_tasks = benchmark.n_tasks

    # prepare datasets from the benchmark
    task_list, task_demo_path_list = [], []
    descriptions = []
    shape_meta = None
    dataset_inst = []
    dataset_gl = []

    # currently we assume tasks from same benchmark have the same shape_meta
    data_path_0 = os.path.join(cfg.folder, benchmark.get_task_demonstration(0))
    if os.path.exists(data_path_0):
        _, shape_meta = get_dataset(
                n_demos=1,
                dataset_path=data_path_0,
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=True,
                seq_len = cfg.data.seq_len,
                frame_stack = cfg.data.frame_stack
        )
    else:
        print(f"[WARNING]: Dataset not found here {data_path_0} \n Using default shape_meta information")
        train_benchmark = benchmark_dict[train_benchmark_name.lower()]()
        _, shape_meta = get_dataset(
                n_demos=1,
                dataset_path=os.path.join(cfg.folder, train_benchmark.get_task_demonstration(0)),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=True,
                seq_len = cfg.data.seq_len,
                frame_stack = cfg.data.frame_stack
        )

    assert 'task_spec_modalities' in cfg.policy
    task_list = []
    for i in range(n_manip_tasks):
        task_spec_modalities = cfg.policy.task_spec_modalities.split('_')
        task_name = benchmark.get_task(i).name

        # add language to the vision dataset, hence we call vl_dataset
        task_name = benchmark.get_task(i).name
        task_description = benchmark.get_task(i).language
        instructions = benchmark.get_task(i).instructions
        goal_language = benchmark.get_task(i).goal_language

        task_list.append(task_name)
        task_demo_path_list.append(benchmark.get_task_demonstration(i))
        descriptions.append(task_description)
        dataset_inst.append(instructions)
        dataset_gl.append(goal_language)

    inst_embs, inst_tokens = None, None
    gl_embs, gl_tokens = None, None
    ag_task_specs, ai_task_specs = None, None
    task_visual_specifications = [None]*n_manip_tasks
    if ('img' in task_spec_modalities) or ('vid' in task_spec_modalities):
        task_visual_specifications = get_visual_specifications_all(
                                cfg=cfg,
                                task_list=task_list,
                                benchmark_name=benchmark.name,
                                task_demo_path_list=task_demo_path_list,
                                mode=eval_cfg.ts_mode)
    if ('ag' in task_spec_modalities) or ('ai' in task_spec_modalities):
        ag_task_specs, ai_task_specs = get_audio_specification(
                                benchmark_name=benchmark.name,
                                task_list=task_list,
                                cfg=cfg,
                                mode=eval_cfg.ts_mode)
    if 'inst' in task_spec_modalities:
        inst_embs, inst_tokens = get_task_embs(cfg, dataset_inst, spec_type='inst', mode=eval_cfg.ts_mode)
    if 'gl' in task_spec_modalities:
        gl_embs, gl_tokens = get_task_embs(cfg, dataset_gl, spec_type='gl', mode=eval_cfg.ts_mode)

    benchmark.set_gl_embs(gl_embs)
    benchmark.set_inst_embs(inst_embs)
    benchmark.set_inst_tokens(inst_tokens)
    benchmark.set_visual_task_specifications(task_visual_specifications)
    benchmark.set_ag_task_specs(ag_task_specs)
    benchmark.set_ai_task_specs(ai_task_specs)

    gsz = cfg.data.task_group_size
    n_tasks = n_manip_tasks // gsz # number of multitask learning tasks
    cfg.shape_meta = shape_meta
    task_range = range(n_tasks) if eval_cfg.task_id == -1 else [eval_cfg.task_id]
    print("\n=================== Lifelong Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_manip_tasks // gsz}")
    for i in task_range:
        print(f"    - Task {i+1}:")
        for j in range(gsz):
                print(f"        {benchmark.get_task(i*gsz+j).language}")
    print("=======================================================================\n")
    all_tasks = list(range(benchmark.n_tasks)) if eval_cfg.task_id == -1 else [eval_cfg.task_id]
    keys_to_log = ["epoch", "seed", "n_eval", "mean_success", "ci"]
    for task_id in all_tasks:
        task_i = benchmark.get_task(task_id).name
        keys_to_log.append(f"{task_i}")
    eval_logger = EvalLogger(keys_to_log)

    # define multitask algorithm
    algo = safe_device(eval(cfg.lifelong.algo)(n_tasks, cfg, logger=None), cfg.device)
    for ind in range(len(cfg.pretrain_model_path)):
        epoch = -1
        eval_logger.add_kv('epoch', epoch)
        eval_logger.add_kv('seed', cfg.seed)
        eval_logger.add_kv('n_eval', cfg.eval.n_eval)
        algo.policy.load_state_dict(torch_load_model(cfg.pretrain_model_path[ind], device=cfg.device)[0], strict=True)

        print(f"[info] start evaluation with algo {cfg.lifelong.algo}")
         # "result_summary" stores all the sim states of each eval loop.
         # Correct way to access result_summary[task_ind]["sim_states"][eval_number]
        result_summary = {}
        for task_ind in all_tasks:
            result_summary[task_ind] = {}

        algo.eval()
        t0 = time.time()

        ## Runtime specification selection, off unless eval_modality_set is given.
        spec_provider, spec_modalities = None, None
        if cfg.eval_modality_sets is not None:
            spec_modalities = cfg.eval_modality_sets
            spec_provider = TaskSpecProvider(
                    benchmark=benchmark,
                    policy=algo.policy,
                    device=cfg.device,
                    num_task_frames=cfg.policy.num_task_frames,
                    mode=cfg.modality_delivery)
            if spec_provider.mode == TaskSpecProvider.BANK:
                spec_provider.prepare(range(n_manip_tasks), spec_modalities)

        ## Raw log-likelihood recording, off unless record_logprobs is set.
        recorder = None
        if cfg.record_logprobs:
            run_tag = f"{cfg.benchmark_name}_{prefix}ts_{eval_cfg.ts_mode}_{eval_cfg.model_name}"
            wandb_sink = None
            if cfg.record_wandb:
                wandb_sink = WandbLogProbSink(
                        project=cfg.wandb_eval_project,
                        run_name=run_tag,
                        mode=cfg.wandb_eval_mode,
                        dir=eval_cfg.experiment_dir,
                        config={'benchmark': cfg.benchmark_name,
                                'modality_sets': cfg.eval_modality_sets,
                                'modality_delivery': cfg.modality_delivery,
                                'ts_mode': eval_cfg.ts_mode,
                                'model_name': eval_cfg.model_name,
                                'n_eval': cfg.eval.n_eval},
                        trace_episodes=cfg.wandb_trace_episodes)
            recorder = LogProbRecorder(
                    out_path=os.path.join(cfg.record_logprobs_dir, f"{run_tag}.jsonl"),
                    store_actions=cfg.record_actions,
                    sink=wandb_sink)
            recorder.set_context(
                    run_id=f"{int(time.time())}_{os.getpid()}",
                    ckpt=eval_cfg.model_name,
                    source='rollout',
                    ts_mode=eval_cfg.ts_mode)

        with torch.no_grad():
            task_embs = bm_set_task_embs(
                                algo=algo,
                                benchmark=benchmark,
                                eval_spec_modalities=cfg.policy.task_spec_modalities,
                                task_range=range(n_manip_tasks),
                                device=cfg.device)
            success_rates, result_summary = evaluate_multitask_training_success(
                                                                cfg=cfg,
                                                                algo=algo,
                                                                benchmark=benchmark,
                                                                task_ids=all_tasks,
                                                                result_summary=result_summary,
                                                                spec_provider=spec_provider,
                                                                recorder=recorder,
                                                                spec_modalities=spec_modalities)
            if recorder is not None:
                recorder.close()
                print(f"[info] wrote {recorder.n_written} log-prob rows to {recorder.out_path}")
            for index, task_id in enumerate(all_tasks):
                task_i = benchmark.get_task(task_id).name
                eval_logger.add_kv(f"{task_i}", success_rates[index])
            success_rate = np.mean(success_rates)
            ci = confidence_interval(success_rate, cfg.eval.n_eval)
            eval_logger.add_kv("mean_success", success_rate)
            eval_logger.add_kv("ci", ci)
        t1 = time.time()
        print(f"[info] succ: {success_rate:4.2f} ± {ci:4.2f} | time: {(t1-t0)/60:4.2f}", flush=True)
        if eval_cfg.save_video:
            for task_ind in all_tasks:
                task_i = benchmark.get_task(task_ind)
                summary2video(
                        task_ind=task_ind,
                        task_i=task_i,
                        result_summary=result_summary,
                        eval_cfg=eval_cfg,
                        cfg=cfg)
        eval_logger.save(cfg.eval_csv_filename)

    print("[info] finished evaluating\n")
    del algo

if __name__ == "__main__":
    main()
