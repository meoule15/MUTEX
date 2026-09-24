"""Tests for the scoring infrastructure.

Deliberately runnable without the pretrained weights, the LIBERO dataset, or the
HuggingFace CLIP download: the heavy imports are stubbed where needed, so these
guard the parts that are ours. Run with:

    python -m pytest tests/test_scoring_infra.py
    python tests/test_scoring_infra.py        # no pytest required
"""

import ast
import json
import os
import random
import sys
import tempfile
import types

import torch
import torch.distributions as D

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _load_function(path, func_name, cls_name=None, namespace=None):
    """Exec one top-level or class-level function without importing its module."""
    tree = ast.parse(open(os.path.join(REPO, path)).read())
    scope = tree.body
    if cls_name:
        scope = [n for n in ast.walk(tree)
                 if isinstance(n, ast.ClassDef) and n.name == cls_name][0].body
    fn = [n for n in scope if isinstance(n, ast.FunctionDef) and n.name == func_name][0]
    ns = dict(namespace or {})
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<extracted>', 'exec'), ns)
    return ns[func_name]


POLICY = 'mutex/models/policy/bc_mutex_policy.py'
DATASET = 'mutex/lf_datasets.py'
METRIC = 'mutex/metric.py'


# --------------------------------------------------------------------------
# resolve_modalities: the string-vs-list trap
# --------------------------------------------------------------------------

def test_resolve_modalities():
    resolve = _load_function(POLICY, 'resolve_modalities')
    default = ['gl', 'inst', 'img', 'vid', 'ai', 'ag']

    assert resolve(None, default) == default
    assert resolve(None, default) is not default, "must copy, not alias the config list"

    # The trap: eval.py passes a string. len("gl") == 2 would flip the
    # num_modalities > 1 branches and silently change published success rates.
    assert resolve('gl', default) == ['gl']
    assert len(resolve('gl', default)) == 1 != len('gl')

    assert resolve('gl_inst', default) == ['gl', 'inst']
    assert resolve(['gl'], default) == ['gl']
    assert resolve(('gl', 'vid'), default) == ['gl', 'vid']


# --------------------------------------------------------------------------
# LogProbRecorder
# --------------------------------------------------------------------------

def _gmm(B, T, M, A):
    means = torch.randn(B, T, M, A)
    stds = torch.rand(B, T, M, A) + 0.1
    return D.MixtureSameFamily(D.Categorical(logits=torch.randn(B, T, M)),
                               D.Independent(D.Normal(means, stds), 1))


def test_recorder_writes_raw_logp():
    from mutex.logprob_recorder import LogProbRecorder
    B, T, A = 2, 3, 7
    gmm = _gmm(B, T, 5, A)
    actions = torch.randn(B, T, A)
    logp = gmm.log_prob(actions)
    assert logp.shape == (B, T)

    path = os.path.join(tempfile.mkdtemp(), 'nested', 'task_0.jsonl')
    rec = LogProbRecorder(path, store_actions=True)
    rec.set_context(run_id='r1', source='offline', task_id=0, ts_mode='eval', spec_index=8)
    with rec.context(modality='gl'):
        rec.record(logp=logp, actions=actions, action_source='ground_truth')
    rec.close()

    rows = [json.loads(line) for line in open(path)]
    assert len(rows) == B * T
    for row in rows:
        b, t = row['batch_index'], row['seq_index']
        # Raw: not negated, not scaled by loss_coef, nothing subtracted.
        assert abs(row['logp'] - float(logp[b, t])) < 1e-6
        assert row['modality'] == 'gl'
        assert row['ts_mode'] == 'eval' and row['spec_index'] == 8
        assert row['action_source'] == 'ground_truth'
        assert len(row['action']) == A
    assert 'modality' not in rec._context, "context() must pop on exit"


def test_recorder_is_inert_when_unused():
    from mutex.logprob_recorder import LogProbRecorder
    path = os.path.join(tempfile.mkdtemp(), 'never', 'x.jsonl')
    LogProbRecorder(path).close()
    assert not os.path.exists(os.path.dirname(path)), "must not create dirs when unused"


def test_wandb_sink_degrades_without_wandb():
    """Recording must survive wandb being absent or failing to init."""
    import builtins
    from mutex.logprob_recorder import LogProbRecorder, WandbLogProbSink

    real_import = builtins.__import__

    def no_wandb(name, *a, **k):
        if name == 'wandb':
            raise ImportError('simulated: wandb not installed')
        return real_import(name, *a, **k)

    builtins.__import__ = no_wandb
    try:
        sink = WandbLogProbSink(project='t', mode='offline')
        assert sink.enabled is False
        path = os.path.join(tempfile.mkdtemp(), 'x.jsonl')
        rec = LogProbRecorder(path, sink=sink)
        rec.record(logp=torch.tensor(1.5), modality='gl', task_id=0, episode=0, step=0)
        rec.close()
        assert len(open(path).readlines()) == 1, "NDJSON must be written regardless"
    finally:
        builtins.__import__ = real_import


def test_wandb_sink_bounds_traces():
    """Per-step traces are capped; aggregates still cover every row."""
    from mutex.logprob_recorder import WandbLogProbSink

    sink = WandbLogProbSink.__new__(WandbLogProbSink)  # no wandb init
    sink.trace_episodes, sink.max_trace_steps = 2, 10
    sink._stats, sink._traces, sink._wandb = {}, {}, None

    for episode in range(5):
        for step in range(50):
            sink.add({'task_id': 0, 'modality': 'gl', 'logp': 0.1,
                      'episode': episode, 'step': step})

    assert sorted({k[1] for k in sink._traces}) == [0, 1], "only first N episodes traced"
    assert all(len(v) == 10 for v in sink._traces.values()), "trace length capped"
    assert sink._stats[(0, 'gl')][0] == 250, "aggregates still see every row"


def test_recorder_scalar_and_batch_shapes():
    from mutex.logprob_recorder import LogProbRecorder
    path = os.path.join(tempfile.mkdtemp(), 'shapes.jsonl')
    rec = LogProbRecorder(path)
    rec.record(logp=torch.tensor(1.5), modality='gl')          # scalar
    rec.record(logp=torch.tensor([1.0, 2.0]), modality='gl')   # [B]
    rec.record(logp=torch.zeros(2, 3), modality='gl')          # [B, T]
    rec.close()
    rows = [json.loads(line) for line in open(path)]
    assert len(rows) == 1 + 2 + 6
    assert rows[0]['batch_index'] is None and rows[0]['seq_index'] is None
    assert rows[1]['batch_index'] == 0 and rows[1]['seq_index'] is None
    assert rows[3]['seq_index'] == 0


# --------------------------------------------------------------------------
# TaskSpecProvider: bank and reembed must agree
# --------------------------------------------------------------------------

NTS, T_EMB, E = 3, 4, 8


class _StubBenchmark:
    def _d(self, i, tag, shape):
        g = torch.Generator().manual_seed(abs(hash((i, tag))) % (2 ** 31))
        return torch.randn(*shape, generator=g)

    def get_visual_task_specification(self, i):
        return {'vid_task_spec': [self._d(i, f'v{k}', (20, E)) for k in range(NTS)],
                'vid_task_spec_mask': [torch.ones(20) for _ in range(NTS)],
                'img_task_spec': [self._d(i, f'i{k}', (50, E)) for k in range(NTS)]}

    def get_inst_emb(self, i):
        return self._d(i, 'inst', (NTS, 10, E))

    def get_gl_emb(self, i):
        return self._d(i, 'gl', (NTS, E))

    def get_ai_task_spec(self, i):
        return {'ai_task_spec': self._d(i, 'ai', (NTS, 6, E)),
                'ai_task_spec_mask': torch.ones(NTS, 6)}

    def get_ag_task_spec(self, i):
        return {'ag_task_spec': self._d(i, 'ag', (NTS, 6, E)),
                'ag_task_spec_mask': torch.ones(NTS, 6)}


class _StubPolicy:
    """Batch-independent per-item transform, as the real ts_transform_modules are."""

    def get_task_embs(self, data_dict, modalities=None):
        keys = sorted(k for k, v in data_dict.items() if torch.is_tensor(v))
        ref = data_dict[keys[0]]
        rows = []
        for b in range(ref.shape[0]):
            flat = ref[b].flatten()
            rep = flat.repeat((T_EMB * E) // flat.numel() + 1)[:T_EMB * E]
            rows.append(rep.reshape(T_EMB, E) * (len(keys) + 1))
        return torch.stack(rows), None, modalities, None


def _stub_heavy_imports():
    """Stub the robomimic/transformers import chain the provider pulls in."""
    pol = types.ModuleType('mutex.models.policy.bc_mutex_policy')
    pol.resolve_modalities = _load_function(POLICY, 'resolve_modalities')
    sys.modules.setdefault('mutex.models.policy.bc_mutex_policy', pol)
    u = types.ModuleType('mutex.utils')
    u.sample_frames = lambda num_frames, vlen, sample: list(range(min(num_frames, vlen)))
    sys.modules.setdefault('mutex.utils', u)


def test_bank_matches_reembed():
    _stub_heavy_imports()
    from mutex.task_spec_provider import TaskSpecProvider

    bench, pol = _StubBenchmark(), _StubPolicy()
    keys = ['gl', 'inst', 'img', 'vid', 'ai', 'ag']
    tasks = [0, 1, 2]

    bank = TaskSpecProvider(bench, pol, 'cpu', num_task_frames=16,
                            mode='bank').prepare(tasks, keys)
    reembed = TaskSpecProvider(bench, pol, 'cpu', num_task_frames=16,
                               mode='reembed', cache_reembed=False)

    for t in tasks:
        for k in keys:
            for s in range(NTS):
                a, b = bank.get(t, k, s), reembed.get(t, k, s)
                assert a.shape == (T_EMB, E)
                assert torch.equal(a, b), f"bank != reembed at {(t, k, s)}"

    many = bank.get_many(0, ['gl', 'vid'], 1)
    assert list(many.keys()) == ['gl', 'vid'], "get_many must preserve order"
    assert bank.num_specs(0, 'gl') == NTS
    assert bank.get(0, 'gl_inst', 0).shape == (T_EMB, E), "joined modality keys"


# --------------------------------------------------------------------------
# Dataset spec pinning must not disturb the training RNG stream
# --------------------------------------------------------------------------

def test_pick_spec_id_preserves_rng_stream():
    pick = _load_function(DATASET, '_pick_spec_id', cls_name='MLMTaskDataset',
                          namespace={'random': random})

    class _D:
        pass

    d = _D()
    d.fixed_task_spec_id = None
    n = 11

    random.seed(0)
    mine = [pick(d, n) for _ in range(500)]
    random.seed(0)
    orig = [random.randint(0, n - 1) for _ in range(500)]
    assert mine == orig, "default path must consume the RNG exactly as before"

    d.fixed_task_spec_id = 8
    random.seed(0)
    before = random.random()
    random.seed(0)
    [pick(d, n) for _ in range(500)]
    assert random.random() == before, "pinned path must not consume RNG"
    assert all(pick(d, n) == 8 for _ in range(10))
    assert pick(d, 3) == 2, "must stay in range for modalities with fewer variants"


# --------------------------------------------------------------------------
# Signatures: existing call patterns must still bind
# --------------------------------------------------------------------------

def _signature(path, func_name, cls_name=None):
    tree = ast.parse(open(os.path.join(REPO, path)).read())
    scope = tree.body
    if cls_name:
        scope = [n for n in ast.walk(tree)
                 if isinstance(n, ast.ClassDef) and n.name == cls_name][0].body
    fn = [n for n in scope if isinstance(n, ast.FunctionDef) and n.name == func_name][0]
    args = [a.arg for a in fn.args.args]
    n_def = len(fn.args.defaults)
    return args, (args[:len(args) - n_def] if n_def else args)


def test_signatures_backward_compatible():
    cases = [
        (POLICY, 'forward', 'BCMutexPolicy',
         ['self', 'data'], ['reduction', 'modalities']),
        (POLICY, 'get_action', 'BCMutexPolicy',
         ['self', 'data'], ['return_dist']),
        (POLICY, 'get_task_embs', 'BCMutexPolicy',
         ['self', 'data'], ['modalities']),
        (METRIC, 'evaluate_one_task_success', None,
         ['cfg', 'algo', 'task', 'task_emb', 'task_id'],
         ['sim_states', 'task_str', 'spec_provider', 'recorder', 'spec_modalities']),
        (METRIC, 'evaluate_multitask_training_success', None,
         ['cfg', 'algo', 'benchmark', 'task_ids'],
         ['result_summary', 'spec_provider', 'recorder', 'spec_modalities']),
    ]
    for path, fn, cls, required, optional in cases:
        args, actual_required = _signature(path, fn, cls)
        assert actual_required == required, f"{fn}: {actual_required} != {required}"
        for opt in optional:
            assert opt in args, f"{fn}: missing optional arg {opt}"


if __name__ == '__main__':
    passed = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f"  PASS {name}")
            passed += 1
    print(f"\n{passed} passed")
