# Plugin Zoo Implementation Plan

The plugin zoo is now a flat Swift package under `FXPlugins/`. Each plugin has
its own folder and owns all implementation variants. The shared `API/` folder is
restricted to registry, runtime, backend bridge, tests, and documentation.

## Required Plugin Layout

```text
FXPlugins/<plugin_name>/
  <PluginName>Plugin.swift
  CPU/
  Metal/        # when useful
  PyTorch/      # when useful
  TensorFlow/   # when useful
  NLP/          # when useful
```

## Backend Selection Rules

- CPU is mandatory and is the deterministic reference.
- Metal is required for large batch scans, tree scoring, dense projection, or
  parameter sweeps where Apple GPU execution materially reduces runtime.
- PyTorch is required for sequence, RL, transformer, memory, and world-model
  implementations where MPS support is useful on Apple Silicon M2/M3-class hosts.
- TensorFlow is required only where the model has a strong Keras/TensorFlow
  reference path and TensorFlow Metal can accelerate it.
- NLP is required only where text/event context is a first-class signal.
- CoreML/Neural Engine remains undeclared until export/load/predict/parity code
  exists for a concrete plugin.

## Active Gates

The current source of truth is:

- `PLUGIN_REFERENCE_IMPLEMENTATION_SCORECARD.md`
- `PLUGIN_REFERENCE_IMPLEMENTATION_AUDIT.md`
- `PLUGIN_99_REFERENCE_IMPLEMENTATION_PLAN.md`
- `PLUGIN_100_LIVE_RUNTIME_COMPLETION_PLAN.md`

All gates use FXDatabase/FXDataEngine M1 OHLCV contracts and SineTest smoke
fixtures. Market data access outside FXDatabase APIs is not permitted.
