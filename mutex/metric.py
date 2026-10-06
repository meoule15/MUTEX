import gc
import copy
import gc
from collections import OrderedDict
import cv2
import imageio
import numpy as np
import os
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import time
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from mutex.pmi import action_lift, expected_lift, SpecificationSelector, RandomSpecificationSelector
from robosuite import load_controller_config
from time import gmtime, strftime
from torch.multiprocessing import Array
from torch.utils.data import DataLoader
from tqdm import trange

from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from mutex.utils import *


def _expand_task_emb(task_emb, env_num):
    """Broadcast one task embedding across the parallel envs."""
    if len(task_emb.shape) == 1: ## adds a new dimension and repeats along it
        return task_emb.repeat(env_num, 1)
    elif len(task_emb.shape) == 2:
        return task_emb.repeat(env_num, 1, 1)
    elif len(task_emb.shape) == 3 and task_emb.shape[0] == env_num:
        return task_emb ## already one embedding per env
    else:
        raise NotImplementedError


def raw_obs_to_tensor_obs(obs, task_emb, cfg):
    """
        Prepare the tensor observations as input for the algorithm.
    """
    env_num = len(obs)
    task_emb = _expand_task_emb(task_emb, env_num)

    data = {
        "obs": {
            "agentview_rgb"  : [],
            "eye_in_hand_rgb": [],
            "gripper_states" : [],
            "joint_states"   : [],
        },
        "task_emb": task_emb,
    }

    for k in range(env_num):
        data["obs"]["agentview_rgb"].append(ObsUtils.process_obs(
                torch.from_numpy(obs[k]["agentview_image"]),
                obs_key="agentview_rgb"))

        data["obs"]["eye_in_hand_rgb"].append(ObsUtils.process_obs(
                torch.from_numpy(obs[k]["robot0_eye_in_hand_image"]),
                obs_key="eye_in_hand_rgb"))

        data["obs"]["gripper_states"].append(torch.from_numpy(np.array(
                obs[k]["robot0_gripper_qpos"])).float()),

        data["obs"]["joint_states"].append(torch.from_numpy(np.array(
                obs[k]["robot0_joint_pos"])).float()),

    for key in data["obs"]:
        data["obs"][key] = torch.stack(data["obs"][key])

    data = TensorUtils.map_tensor(data,
                                  lambda x: safe_device(x, device=cfg.device))
    return data


def evaluate_one_task_success(
                        cfg,
                        algo,
                        task,
                        task_emb,
                        task_id,
                        sim_states=None,
                        task_str="",
                        spec_provider=None,
                        recorder=None,
                        spec_modalities=None,
    ):
    """
        Evaluate a single task's success rate
        sim_states: if not None, will keep track of all simulated states during
                    evaluation, mainly for visualization and debugging purpose
        task_str:   the key to access sim_states dictionary
        spec_provider:   if not None, a TaskSpecProvider supplying task embeddings
                    per step instead of the fixed `task_emb`
        recorder:   if not None, a LogProbRecorder receiving the log-likelihood of
                    each executed action under every scored specification
        spec_modalities: modality keys to score when spec_provider is given
    """

    t0 = time.time()
    algo.eval()

    # initiate evaluation envs
    env_args = {
        "bddl_file_name": os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
        "camera_heights": cfg.data.img_h,
        "camera_widths": cfg.data.img_w,
    }

    env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
    eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

    if env_num == 1:
        env = OffScreenRenderEnv(**env_args)
    else:
        env = SubprocVectorEnv([
            lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
    env.seed(cfg.seed)

    # Evaluation loop
    num_success = 0

    # get fixed init states to control the experiment randomness
    init_states_path = os.path.join(cfg.init_states_folder,
                                    task.problem_folder,
                                    task.init_states_file)
    init_states = torch.load(init_states_path)

    for i in range(eval_loop_num):
        env.reset()

        indices = np.arange(i*env_num, (i+1)*env_num) % init_states.shape[0]
        init_states_ = init_states[indices]

        dones = [False] * env_num
        steps = 0
        algo.reset()
        obs = env.set_init_state(init_states_) if env_num > 1 else env.set_init_state(init_states_[0])

        # dummy actions [env_num, 7] all zeros for initial physics simulation
        dummy = np.zeros((env_num, 7)) if env_num > 1 else np.zeros((7,))
        for _ in range(5):
            obs, _, _, _ = env.step(dummy)

        episode_offset = i * env_num
        if spec_provider is None:
            ## Unchanged published path. Note only the last k survives this loop;
            ## inert at the shipped use_mp=False, where env_num == 1.
            for k in range(env_num):
                task_emb_eval = task_emb[(episode_offset+k) % task_emb.shape[0]]
            spec_indices = None
        else:
            ## With a provider the specification is indexed explicitly per env.
            n_specs = spec_provider.num_specs(task_id, spec_modalities[0])
            spec_indices = [(episode_offset+k) % n_specs for k in range(env_num)]
            task_emb_eval = None

        if task_str != "":
            sim_state = env.get_sim_state()
            for k in range(env_num):
                if i*env_num+k < cfg.eval.n_eval:
                    sim_states[i*env_num+k].append(sim_state if env_num == 1 else sim_state[k])

        selector = None
        if getattr(cfg, 'spec_selection', 'fixed') == 'expected_lift':
            selector = SpecificationSelector(spec_modalities, env_num,
                        cfg.selection_interval, cfg.selection_margin, cfg.selection_uncertainty,
                        initial_key=getattr(cfg,'selection_initial_modality',None))
        elif getattr(cfg, 'spec_selection', 'fixed') == 'random':
            selector = RandomSpecificationSelector(spec_modalities, env_num,
                                                   cfg.selection_interval, cfg.seed + episode_offset)

        while steps < cfg.eval.max_steps:
            steps += 1

            if env_num == 1: obs = [obs]

            if spec_provider is None and recorder is None:
                ## Unchanged published path.
                data = raw_obs_to_tensor_obs(obs, task_emb_eval, cfg)
                actions = algo.policy.get_action(data)
            else:
                if spec_provider is not None:
                    task_embs = OrderedDict(
                            (key, torch.stack([spec_provider.get(task_id, key, s)
                                               for s in spec_indices], dim=0))
                            for key in spec_modalities)
                    acting_key = spec_modalities[0]
                    data = raw_obs_to_tensor_obs(obs, task_embs[acting_key], cfg)
                    if getattr(cfg, 'record_pmi', False):
                        for key in spec_modalities:
                            task_embs['blank:' + key] = torch.stack([
                                spec_provider.get_blank(task_id, key, s)
                                for s in spec_indices], dim=0)
                    dists = algo.policy.get_action_dists(data, task_embs)
                else:
                    data = raw_obs_to_tensor_obs(obs, task_emb_eval, cfg)
                    acting_key = '_'.join(algo.policy.task_spec_modalities)
                    _, dist = algo.policy.get_action(data, return_dist=True)
                    dists = OrderedDict([(acting_key, dist)])

                ## One executed action, scored under every candidate specification,
                ## so rows for the same (episode, step) are directly comparable.
                selection_scores = {}
                switched = [False] * env_num
                if selector is not None:
                    if selector.due(steps) and cfg.spec_selection == 'expected_lift':
                        devices = [torch.device(cfg.device).index or 0] if str(cfg.device).startswith('cuda') else []
                        with torch.random.fork_rng(devices=devices):
                            for key in spec_modalities:
                                selection_scores[key] = expected_lift(
                                    dists[key], [dists['blank:' + key]], cfg.pmi_samples)
                        switched = selector.update(steps, selection_scores)
                    elif selector.due(steps):
                        switched = selector.update(steps)
                    selected = selector.current
                    sampled = {key: dists[key].sample().detach() for key in spec_modalities}
                    act_t = torch.stack([sampled[key][k] for k, key in enumerate(selected)])
                else:
                    selected = [acting_key] * env_num
                    act_t = dists[acting_key].sample().detach()
                if recorder is not None:
                    for key, dist in dists.items():
                        if key.startswith('blank:'):
                            continue
                        logp = dist.log_prob(act_t)
                        pmi = None
                        if getattr(cfg, 'record_pmi', False):
                            ref = dists['blank:' + key]
                            pmi = action_lift(dist, [ref], act_t)
                            # Diagnostic samples must not perturb the acting RNG.
                            devices = [act_t.device.index] if act_t.is_cuda else []
                            if key in selection_scores:
                                pmi.update(selection_scores[key])
                            else:
                                with torch.random.fork_rng(devices=devices):
                                    pmi.update(expected_lift(dist, [ref], cfg.pmi_samples))
                        for k in range(env_num):
                            episode = episode_offset + k
                            if episode >= cfg.eval.n_eval:
                                continue
                            recorder.record(
                                    logp=logp[k],
                                    actions=act_t[k],
                                    modality=key,
                                    spec_index=None if spec_indices is None else spec_indices[k],
                                    episode=episode,
                                    env_index=k,
                                    step=steps,
                                    **({
                                        'reference': 'blank',
                                        'reference_logp': pmi['reference_logp'][k].item(),
                                        'lift': pmi['lift'][k].item(),
                                        'expected_lift': pmi['expected_lift'][k].item(),
                                        'expected_lift_se': pmi['expected_lift_se'][k].item(),
                                        'entropy': pmi['entropy'][k].item(),
                                        'num_samples': cfg.pmi_samples,
                                        'acting_modality': selected[k],
                                        'spec_selection': getattr(cfg, 'spec_selection', 'fixed'),
                                        'selection_decision': selector is not None and selector.due(steps),
                                        'selection_switched': switched[k],
                                        'selection_status': (selector.last_decisions[k]['status'] if
                                            getattr(cfg,'spec_selection','fixed')=='expected_lift' and selector.due(steps) else None),
                                        'selection_interval': getattr(cfg, 'selection_interval', None),
                                        'selection_margin': getattr(cfg, 'selection_margin', None),
                                        'selection_uncertainty': getattr(cfg, 'selection_uncertainty', None),
                                    } if pmi is not None else {}),
                                    action_source='sampled')
                actions = act_t.cpu()
                actions = actions.view(actions.shape[0], -1).numpy()

            if env_num == 1: actions = actions[0]

            obs, reward, done, info = env.step(actions)

            # record the sim states for replay purpose
            if task_str != "":
                sim_state = env.get_sim_state()
                for k in range(env_num):
                    if i*env_num+k < cfg.eval.n_eval:
                        sim_states[i*env_num+k].append(sim_state if env_num == 1 else sim_state[k])

            # check whether succeed
            if env_num == 1:
                dones[0] = done
            else:
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]

            if all(dones): break

        # a new form of success record
        for k in range(env_num):
            if i*env_num+k < cfg.eval.n_eval:
                num_success += int(dones[k])

    success_rate = num_success / cfg.eval.n_eval
    env.close()
    gc.collect()
    t1 = time.time()
    print(f"[info] evaluate task {task_id} takes {(t1-t0)/60:.1f} min")
    return success_rate, sim_states


def evaluate_success(cfg, algo, benchmark, task_ids, result_summary=None):
    """
        Evaluate the success rate for all task in task_ids.
    """
    algo.eval()
    successes = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        task_emb = benchmark.get_task_emb(i)
        task_str = f"k{task_ids[-1]}_p{i}"
        curr_summary = result_summary[task_str] if result_summary is not None else None
        success_rate = evaluate_one_task_success(cfg,
                                                 algo,
                                                 task_i,
                                                 task_emb,
                                                 i,
                                                 sim_states=curr_summary,
                                                 task_str=task_str)
        successes.append(success_rate)
    return np.array(successes)


def evaluate_multitask_training_success(cfg, algo, benchmark, task_ids, result_summary=None,
                                        spec_provider=None, recorder=None, spec_modalities=None):
    """
        Evaluate the success rate for all task in task_ids.
    """
    algo.eval()
    successes = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        task_emb = benchmark.get_task_emb(i) # [num_eval_ts, T, E]
        task_str = f"k{task_ids[-1]}_p{i}"
        curr_summary = {}
        for eval_traj_i in range(cfg.eval.n_eval):
            curr_summary[eval_traj_i] = []

        if recorder is not None:
            recorder.set_context(task_id=i, task_name=task_i.name)

        success_rate, curr_summary = evaluate_one_task_success(cfg=cfg,
                                                 algo=algo,
                                                 task=task_i,
                                                 task_emb=task_emb,
                                                 task_id=i,
                                                 task_str=task_str,
                                                 sim_states=curr_summary,
                                                 spec_provider=spec_provider,
                                                 recorder=recorder,
                                                 spec_modalities=spec_modalities)
        successes.append(success_rate)
        print(f"Task {task_i.name}; Success Rate: {success_rate}")
        result_summary[i].update({'sim_states': copy.deepcopy(curr_summary)})
    return np.array(successes), result_summary


@torch.no_grad()
def evaluate_loss(cfg, algo, benchmark, datasets):
    """
        Evaluate the loss on all datasets.
    """
    algo.eval()
    losses = []
    for i, dataset in enumerate(datasets):
        dataloader = DataLoader(dataset,
                                batch_size=cfg.eval.batch_size,
                                num_workers=cfg.eval.num_workers,
                                shuffle=False)
        test_loss = 0
        for data in dataloader:
            data = TensorUtils.map_tensor(
                    data, lambda x: safe_device(x, device=cfg.device))
            loss = algo.policy.get_loss(data)
            test_loss += loss.item()
        test_loss /= len(dataloader)
        losses.append(test_loss)
    return np.array(losses)
