"""Raw action log-likelihood recording.

Records the log-likelihood the policy assigns to an action, tagged with the
conditioning that produced it. It deliberately knows nothing about how those
numbers are later combined: no subtraction, no negation, no scaling. Any
pointwise-mutual-information form is assembled offline by grouping rows on the
conditioning tuple

    (task_id, modality, spec_index, ts_mode, source, episode, env_index, step,
     batch_index, seq_index, action_source)

Output is newline-delimited JSON, appended and flushed incrementally so a killed
rollout still leaves usable rows behind. Read it back with

    pandas.read_json(path, lines=True)

This module imports nothing from MUTEX so it can be unit-tested on its own.
"""

import json
import os
from contextlib import contextmanager


class LogProbRecorder:
    """Append-only sink for raw per-action log-likelihoods.

    Sticky fields set via set_context/context are merged into every row, so call
    sites only pass what varies.
    """

    def __init__(self, out_path, store_actions=False, flush_every=1000, sink=None):
        self.out_path = out_path
        self.store_actions = store_actions
        self.flush_every = flush_every
        ## Optional live view (e.g. WandbLogProbSink). The NDJSON stays the source
        ## of truth; a sink only summarises.
        self.sink = sink
        self._context = {}
        self._buffer = []
        self._n_written = 0
        self._fh = None

    # -- context ---------------------------------------------------------

    def set_context(self, **kv):
        """Set sticky fields merged into every subsequent row."""
        self._context.update(kv)

    @contextmanager
    def context(self, **kv):
        """Temporarily add sticky fields, restoring the previous values on exit."""
        previous = {k: self._context.get(k, KeyError) for k in kv}
        self._context.update(kv)
        try:
            yield self
        finally:
            for k, old in previous.items():
                if old is KeyError:
                    self._context.pop(k, None)
                else:
                    self._context[k] = old

    # -- recording -------------------------------------------------------

    def record(self, logp, actions=None, **row_fields):
        """Record log-likelihoods.

        logp: scalar, [B] or [B, T] tensor of RAW log-probabilities, exactly as
              returned by the distribution. Never negated or scaled here.
        actions: matching [B, ...] or [B, T, ...] action tensor, stored only when
              store_actions is set.
        row_fields: per-call fields, overriding the sticky context.
        """
        values = logp.detach().cpu()
        acts = actions.detach().cpu() if (actions is not None and self.store_actions) else None

        if values.ndim == 0:
            self._emit(float(values), None if acts is None else acts.reshape(-1).tolist(),
                       row_fields, batch_index=None, seq_index=None)
            return

        if values.ndim == 1:
            # Ambiguous by shape alone; treat as a batch of single steps.
            for b in range(values.shape[0]):
                action = None if acts is None else acts[b].reshape(-1).tolist()
                self._emit(float(values[b]), action, row_fields,
                           batch_index=b, seq_index=None)
            return

        if values.ndim == 2:
            for b in range(values.shape[0]):
                for t in range(values.shape[1]):
                    action = None if acts is None else acts[b, t].reshape(-1).tolist()
                    self._emit(float(values[b, t]), action, row_fields,
                               batch_index=b, seq_index=t)
            return

        raise ValueError(
                f"logp must be scalar, [B] or [B, T]; got shape {tuple(values.shape)}")

    def _emit(self, logp, action, row_fields, batch_index, seq_index):
        row = dict(self._context)
        row.update(row_fields)
        row.setdefault('batch_index', batch_index)
        row.setdefault('seq_index', seq_index)
        row['logp'] = logp
        if action is not None:
            row['action'] = action
        self._buffer.append(row)
        if self.sink is not None:
            self.sink.add(row)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    # -- output ----------------------------------------------------------

    def flush(self):
        if not self._buffer:
            return
        if self._fh is None:
            directory = os.path.dirname(self.out_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._fh = open(self.out_path, 'a')
        for row in self._buffer:
            self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        self._n_written += len(self._buffer)
        self._buffer = []

    def close(self):
        self.flush()
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self.sink is not None:
            self.sink.close(ndjson_path=self.out_path if self._n_written else None)

    @property
    def n_written(self):
        return self._n_written

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class WandbLogProbSink:
    """Live wandb view over recorded log-likelihoods.

    The NDJSON file is the source of truth; this only summarises, because a full
    sweep is ~1.2M rows per modality and streaming every one of them to wandb is
    neither useful nor affordable. What it does log:

      * running mean log-likelihood per (task, modality), as a summary Table
      * per-step traces for a bounded number of episodes -- the raw material for
        a per-timestep profile, plotted as one line per modality
      * the NDJSON itself as a run artifact, so the detail is recoverable

    wandb is imported lazily and every call is defensive: if wandb is missing or
    init fails, the sink degrades to a no-op and recording carries on. Follows the
    retry-then-offline pattern already used in main_masked_modeling.py.
    """

    def __init__(self, project, run_name=None, mode='online', dir=None,
                 config=None, trace_episodes=2, max_trace_steps=1000,
                 num_attempts=3):
        self.trace_episodes = trace_episodes
        self.max_trace_steps = max_trace_steps
        self._stats = {}        # (task_id, modality) -> [count, sum, min, max]
        self._traces = {}       # (task_id, episode, modality) -> [(step, logp), ...]
        self._wandb = None

        try:
            import wandb
        except ImportError:
            print("[warn] wandb not installed; log-prob sink disabled")
            return

        for attempt in range(num_attempts):
            try:
                wandb.init(project=project, dir=dir, config=config,
                           mode="offline" if attempt == num_attempts - 1 else mode,
                           reinit=True)
                if run_name:
                    wandb.run.name = run_name
                wandb.define_metric("logprob/step")
                wandb.define_metric("logprob/*", step_metric="logprob/step")
                self._wandb = wandb
                break
            except Exception:
                print(f"[warn] wandb init attempt #{attempt + 1} failed")

    @property
    def enabled(self):
        return self._wandb is not None

    def add(self, row):
        """Accumulate one recorded row. Cheap and allocation-light: this runs per
        action per modality inside the rollout loop."""
        modality = row.get('modality')
        task_id = row.get('task_id')
        logp = row.get('logp')
        if logp is None:
            return

        key = (task_id, modality)
        stat = self._stats.get(key)
        if stat is None:
            self._stats[key] = [1, logp, logp, logp]
        else:
            stat[0] += 1
            stat[1] += logp
            stat[2] = min(stat[2], logp)
            stat[3] = max(stat[3], logp)

        ## Bounded per-step traces, for the first few episodes only.
        episode, step = row.get('episode'), row.get('step')
        if episode is None or step is None or episode >= self.trace_episodes:
            return
        trace = self._traces.setdefault((task_id, episode, modality), [])
        if len(trace) < self.max_trace_steps:
            trace.append((step, logp))

    def close(self, ndjson_path=None):
        """Flush summaries, upload the NDJSON, and end the run."""
        if self._wandb is None:
            return
        wandb = self._wandb
        try:
            if self._stats:
                table = wandb.Table(columns=["task_id", "modality", "n",
                                             "mean_logp", "min_logp", "max_logp"])
                for (task_id, modality), (n, total, lo, hi) in sorted(
                        self._stats.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
                    table.add_data(task_id, modality, n, total / n, lo, hi)
                wandb.log({"logprob/summary": table})
                wandb.summary["logprob/rows"] = sum(v[0] for v in self._stats.values())

            ## One chart per (task, episode); a line per modality on shared steps.
            by_episode = {}
            for (task_id, episode, modality), points in self._traces.items():
                by_episode.setdefault((task_id, episode), {})[modality] = dict(points)
            for (task_id, episode), series in by_episode.items():
                modalities = sorted(series)
                steps = sorted({s for m in modalities for s in series[m]})
                tbl = wandb.Table(columns=["step"] + modalities)
                for s in steps:
                    tbl.add_data(s, *[series[m].get(s) for m in modalities])
                wandb.log({f"logprob/trace_task{task_id}_ep{episode}": tbl})

            if ndjson_path and os.path.exists(ndjson_path):
                art = wandb.Artifact(f"logprobs-{wandb.run.id}", type="logprobs")
                art.add_file(ndjson_path)
                wandb.log_artifact(art)
        except Exception as exc:
            print(f"[warn] wandb log-prob sink failed to flush: {exc}")
        finally:
            try:
                wandb.finish()
            except Exception:
                pass
            self._wandb = None
