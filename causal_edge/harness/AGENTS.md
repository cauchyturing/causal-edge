# Harness — Agent Guide

## I want to...

### Run strategies
```
causal-edge run                     → run_pipeline(config) → yields PipelineEvent
causal-edge run --strategy my_strat → run single strategy
```

### Understand the pipeline
```
pipeline.py  → generator: yields events per phase (run/validate/dashboard)
lifecycle.py → 7-step execution per strategy (load/validate/compute/hooks/write)
types.py     → PipelineEvent, SignalResult (frozen dataclasses)
```

### Add lifecycle hooks
```yaml
# In strategies.yaml, per strategy:
strategies:
  - id: my_strat
    hooks:
      pre_write: "my_module.my_pre_hook"    # called before trade log write
      post_write: "my_module.my_post_hook"  # called after trade log write
```

Hook signature: `fn(strategy_cfg: dict, positions: ndarray, dates: DatetimeIndex)`

### Consume pipeline events programmatically
```python
from causal_edge.config import load_config
from causal_edge.harness.pipeline import run_pipeline

for event in run_pipeline(load_config()):
    if event.phase == "strategy" and event.status == "ok":
        print(f"{event.data['id']}: {event.data['result'].n_days} days")
    elif event.phase == "pipeline" and event.status == "done":
        print(f"Done: {event.data['strategies_ok']} ok")
```

### Debug a strategy failure
```
SignalResult.lifecycle_log shows exactly where it failed:
  ["load:ok", "validate:ok", "compute:FAIL"]
  → Engine's compute_signals() raised an exception

  ["load:FAIL"]
  → Engine module couldn't be imported (check engine path in strategies.yaml)
```
