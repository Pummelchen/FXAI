# Full Swift Plugin Zoo Plan

This document tracks the current Swift plugin zoo, not a retired source tree.
Each plugin owns its complete implementation under `FXPlugins/<plugin>/`:

- `CPU/`: deterministic Swift reference implementation.
- `Metal/`: plugin-local Apple GPU implementation when useful.
- `PyTorch/`: plugin-local PyTorch implementation with required Apple MPS runtime when useful.
- `TensorFlow/`: plugin-local TensorFlow implementation with required TensorFlow Metal GPU runtime when useful.
- `NLP/`: plugin-local text/event runtime when useful.

Shared code is limited to the FXDataEngine/FXPlugins API surface, registry, runtime
selection, certification, and backend process bridge. Plugin-specific kernels,
models, tokenizers, and persistence stay inside the plugin folder.

## Certification Requirements

1. CPU implementation is complete and deterministic.
2. M1 OHLCV contracts are honored; volume is used whenever the dataset has positive volume.
3. Accelerators are declared only when executable runtime code and parity evidence exist.
4. PyTorch backends require MPS on Apple Silicon M2/M3-class hosts and fall back only by explicit policy.
5. TensorFlow backends require local GPU devices through TensorFlow Metal and fall back only by explicit policy.
6. Metal backends compile and run buffers on the active Apple GPU with CPU parity fixtures.
7. NLP backends define tokenizer/event contracts and no-text fallback behavior.
8. No plugin reads market data directly; all market data flows through FXDatabase and FXDataEngine contracts.

## Current Scope

The Swift package exposes the full plugin registry, SineTest runtime smoke checks,
backend discovery, runtime selection, and the strict certification registry. The
detailed per-plugin reference scores live in `PLUGIN_REFERENCE_IMPLEMENTATION_SCORECARD.md`;
remaining runtime quality work is tracked in `PLUGIN_100_LIVE_RUNTIME_COMPLETION_PLAN.md`.
