# AGENT.md

This file records the current downstream rules that must not be forgotten when
working on `liveweb-arena`.

## Scope

`liveweb-arena` is responsible for:

- episode execution
- browser / environment behavior
- task registry integration
- runtime-profile-gated behavior

It is **not** the canonical source of benchmark scoring policy by itself.
Downstream benchmark code decides how a run is executed and how it is judged.

## Runtime Profiles

Downstream `liveweb-arena` supports two explicit runtime profiles:

- `strict_eval`
- `fast_collect`

Reference implementation:

- [liveweb_arena/core/runtime_profiles.py](/home/xmyf/liveweb-arena/liveweb_arena/core/runtime_profiles.py)

### `strict_eval`

Use this for any path that claims upstream-compatible capability measurement.

`strict_eval` should stay semantically aligned with upstream
`AffineFoundation/liveweb-arena`.

Do **not** enable the following by default in `strict_eval`:

- collect-only local recovery
- disallowed-domain correction
- invalid-generated-url correction
- taostats list recovery
- downstream-only prompt profiles
- extra downstream-only loopguard that changes episode semantics

### `fast_collect`

Use this for:

- SFT sampling
- RL rollout collection
- candidate-trajectory generation
- debugging / diagnostics

`fast_collect` may enable:

- local recovery
- fail-fast / early-stop
- loop detection
- invalid URL correction
- collection-specific routing / timeout policy

These tricks may improve runtime efficiency.
They must not redefine final benchmark scoring.

## Recovery Guardrails

Recent RL debugging established several constraints that must not be forgotten.

### Format Recovery Must Be Bounded

`strict_eval`/`fast_collect` recovery paths may be used, but recovery must never
be allowed to rebuild an effectively unbounded prompt.

Current requirements:

- recovery trimming must be token-aware, not only `len(content)//4`
- the final observation/user message may be truncated or summarized if it alone
  exceeds budget
- recovery should prioritize:
  - system prompt
  - current task description
  - the most recent 1-2 steps
  - the recovery remediation message
- recovery must not blindly retain long accessibility trees or the full raw
  browser history

### Recovery Overflow Must Fail Fast

If recovery still exceeds budget or hits model-context limits, it must fail as a
sample-level error, not continue retrying indefinitely.

Expected classifications include:

- `recoverable_context_overflow`
- `format_recovery_overflow`
- `llm_context_overflow`

These should terminate the current recovery attempt quickly rather than sending
another oversized request.

### Long-Tail Protection Matters More Than Saving One Sample

The most expensive historical RL failures came from a single bad trajectory
triggering recovery, overflowing context, and then never being cleaned up
properly downstream.

When in doubt:

- prefer early termination of the bad sample
- do not preserve a pathological trajectory at the cost of blocking a whole
  rollout round

## Browser / Proxy Notes

### Browser Proxy Is Allowed for External Sites

Current downstream RL experiments intentionally allow the browser to use the
system proxy for external websites, because direct access was measured to be
materially slower for several important domains.

Practical rule:

- browser-side external traffic may use the system proxy
- local control-plane traffic and local LLM traffic must not be forced through
  that proxy

Observed high-value domains where system proxy was measured faster than direct
access on the current machine:

- `taostats`
- `coingecko`
- `stooq`
- `hackernews`

### Proxy Logic Must Stay Explicit

If browser proxy behavior changes, keep the decision path explicit and simple.
Do not assume "inherit whatever the shell has" is always safe.

In particular:

- local service traffic should continue to honor `NO_PROXY`
- proxy-specific code paths in browser setup must stay import-safe and
  regression-tested

## Integration Contract

Downstream benchmark code is expected to use:

- strict evaluation:
  - run with `strict_eval`
  - judge with `strict_eval`
- accelerated collection:
  - run with `fast_collect`
  - judge with `strict_eval`

The arena runtime profile controls execution only.
Final benchmark semantics come from the downstream strict judge.

## Online-Aligned Reference

The current online-aligned `LIVEWEB` definition is derived from:

- `affine-cortex`
  - `/tmp/affine-cortex/affine/database/system_config.json`
  - `/tmp/affine-cortex/affine/core/environments.py`
- upstream `liveweb-arena`
  - `/tmp/liveweb-arena-origin-main/liveweb_arena/core/task_registry.py`

### Current online sampling config

From `affine-cortex` `LIVEWEB`:

- `dataset_range = [0, 78060000]`
- `sampling_count = 300`
- `rotation_count = 4`
- `min_completeness = 0.8`

### Current online eval params

From `affine-cortex` environment config:

- `temperature = 0.0`
- `timeout = 7200`
- `max_concurrency = 10`
- `proxy_timeout = 7300`

Notes:

- `max_steps` is not a single fixed constant in upstream; upstream `env.py`
  derives an effective value from task expectations unless explicitly
  overridden.
- no explicit online `max_completion_tokens` constant has been confirmed in the
  known affine/upstream config files.

### Current online task-space assumptions

At the time of writing:

- `num_tasks` only takes `2/3/4`
- there is no online `task1` in the current upstream task-id space
- over the configured dataset range, the `2/3/4` ratio is exactly `1:1:1`

### Current online site/plugin families

For the current configured dataset range, active site families are:

- `coingecko`
- `stooq`
- `taostats`
- `hybrid`
- `hackernews`

The current online-aligned range does not include:

- `openlibrary`
- `openmeteo`
- `arxiv`
- `weather`

This can change if upstream registry ordering or affine dataset range changes.

## Downstream Support Requirement

Downstream-supported tasks must stay aligned with upstream task space in
`strict_eval`.

Practical requirements:

- keep downstream plugin coverage aligned with upstream plugin coverage
- keep strict task registry semantics aligned with upstream task registry
- do not silently fork strict parser / protocol semantics

## Mandatory Maintenance Checks

When changing runtime behavior or task support, always re-check:

1. `strict_eval` still matches upstream semantics
2. `fast_collect` changes are gated behind runtime profile checks
3. downstream task registry still matches upstream for strict-eval paths
4. real downstream vs upstream parity still passes on fixture tasks
5. current online `LIVEWEB` config has not changed upstream

## Source Documents

Read these before changing policy-sensitive behavior:

- [docs/runtime-profiles.md](/home/xmyf/liveweb-arena/docs/runtime-profiles.md)
- [docs/downstream-alignment.md](/home/xmyf/liveweb-arena/docs/downstream-alignment.md)
- [sampling-eval-rl-policy.md](/home/xmyf/liveweb-capability-bench/docs/sampling-eval-rl-policy.md)
