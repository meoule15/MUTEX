"""Replay saved actions, verify reproduction, and trace goal predicates."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from libero.libero import benchmark
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from evaluate_modality_sufficiency import make_seeded_env, observation_hash


def replay(job):
    root, output, row = job
    root, output = Path(root), Path(output)
    torch.set_num_threads(1)
    bm = benchmark.get_benchmark_dict()['libero_100'](0)
    task = bm.get_task(row['task_id'])
    base = Path('LIBERO/libero/libero')
    env = make_seeded_env(dict(bddl_file_name=str(base/'bddl_files'/task.problem_folder/task.bddl_file),
                              camera_heights=128, camera_widths=128), row['environment_seed'])
    name = f"task{row['task_id']}-{row['modality']}-state{row['initial_state_index']}-spec{row['spec_index']}"
    try:
        env.seed(row['environment_seed']); env.reset()
        initial = torch.load(base/'init_files'/task.problem_folder/task.init_states_file)
        obs = env.set_init_state(initial[row['initial_state_index']])
        for _ in range(5): obs, _, _, _ = env.step(np.zeros(7))
        assert observation_hash(obs) == row['initial_observation_sha256'], 'Replay start mismatch'
        data = np.load(root/f"task{row['task_id']}-{row['modality']}-actions.npz")
        index = data['cases'].tolist().index([row['initial_state_index'], row['spec_index']])
        goals = env.env.parsed_problem['goal_state']
        alternatives = {}
        for t in range(6):
            other = bm.get_task(t)
            alternatives[str(t)] = robosuite_parse_problem(str(base/'bddl_files'/other.problem_folder/other.bddl_file))['goal_state']
        objects = [k for k in env.env.obj_body_id if k in env.env.objects_dict]
        start_pos = {k: env.sim.data.body_xpos[env.env.obj_body_id[k]].copy() for k in objects}
        traces = []; frames = []; steps = []; first_success = -1
        for step in range(row['episode_steps']+1):
            if step:
                obs, _, done, _ = env.step(data['actions'][step-1,index])
                if done and first_success < 0: first_success = step
            predicates = [bool(env.env._eval_predicate(p)) for p in goals]
            movement = {k: float(np.linalg.norm(env.sim.data.body_xpos[env.env.obj_body_id[k]]-start_pos[k])) for k in objects}
            grasp = {k: bool(env.env._check_grasp(env.robots[0].gripper, env.env.objects_dict[k].contact_geoms)) for k in objects}
            traces.append(dict(step=step, predicates=predicates, object_displacement=movement,
                               grasp=grasp, alternative_goals={t: all(env.env._eval_predicate(p) for p in ps) for t, ps in alternatives.items()}))
            if step % 50 == 0 or step == row['episode_steps']:
                frames.append(obs['agentview_image'][::-1].copy()); steps.append(step)
        assert first_success == row['success_step'], f'Replay outcome differs: {first_success} versus {row["success_step"]}'
        result=dict(case=row,goal_predicates=goals,language_variants=task.goal_language[-3:],
                    first_success=first_success,reproduced=True,traces=traces)
        (output/f'{name}.json').write_text(json.dumps(result,indent=2))
        np.savez_compressed(output/f'{name}-frames.npz',frames=np.stack(frames),steps=steps)
        print('REPLAYED', name, 'success_step', first_success, flush=True)
        return name
    finally:
        env.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--selection-file',type=Path,help='Explicit saved episode records to replay')
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads((args.directory/'episodes.json').read_text())
    selected=[]
    for task, failed_spec in ([] if args.selection_file else [(1,0),(2,1),(5,2),(4,2)]):
        candidates=[r for r in rows if r['task_id']==task and r['modality']=='gl' and r['spec_index']==failed_spec and not r['success']]
        for failure in candidates:
            controls=[r for r in rows if r['task_id']==task and r['modality']=='gl' and r['initial_state_index']==failure['initial_state_index'] and r['success']]
            if controls:
                selected.extend([failure,controls[0]])
                selected.append(next(r for r in rows if r['task_id']==task and r['modality']=='vid' and r['initial_state_index']==failure['initial_state_index'] and r['spec_index']==failed_spec))
                break
        else: raise ValueError(f'No matched successful language control for task {task}')
    if args.selection_file:
        selected=json.loads(args.selection_file.read_text())
        assert selected and all(r in rows for r in selected)
    (args.output/'selection.json').write_text(json.dumps(selected,indent=2))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        names=list(pool.map(replay,[(str(args.directory),str(args.output),r) for r in selected]))
    (args.output/'complete.json').write_text(json.dumps(dict(cases=names,all_reproduced=True)))


if __name__=='__main__': main()
