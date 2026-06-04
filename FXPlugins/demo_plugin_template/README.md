# Demo Plugin Template

`demo_plugin_template` is the top-level template for new FXAI plugins. It is
compile-checked by the `FXPlugins` package, excluded from
`FXAIPluginRegistry.availablePlugins()`, and intentionally returns no-trade
predictions until a real plugin replaces the shell logic.

## Runtime Surfaces

The template covers every runtime surface currently supported by FXAI:

| Runtime | Template path | Contract |
| --- | --- | --- |
| Swift CPU/reference | `CPU/DemoPluginTemplate.swift` | `FXAIPlannedPlugin`, `PluginManifestV4`, `TrainRequestV4`, `PredictRequestV4`, `PredictionV4`, parameter rows, and CPU fallback. |
| Metal | `Metal/DemoPluginTemplateMetal.swift` | Plugin-local Metal descriptor plus kernel source string. |
| PyTorch MPS | `PyTorch/demo_plugin_template_torch.py` | Python bridge module with `predict_batch` and `train_step`; prefers Apple Silicon MPS. |
| TensorFlow Metal | `TensorFlow/demo_plugin_template_tensorflow.py` | Python bridge module with `predict_batch` and `train_step`; runs on Python 3.12 with `tensorflow==2.18.1` and `tensorflow-metal==1.2.0`. |
| NLP | `NLP/demo_plugin_template_nlp.py` | Text-event/token feature scaffold with `merge_into_numeric_features`, `predict_batch`, and `train_step` helpers for `foundationNLP` payloads. |
| ONNX Runtime | `ONNX/` | Inference-only exported-model layout and manifest example. |
| Remote RPC | `RemoteRPC/` | Inference-only external server contract and response example. |

Do not add a runtime backend to a production plugin's acceleration plan until
that runtime has real implementation logic and passing certification evidence.

## New Plugin Workflow

1. Copy this directory to `FXPlugins/<plugin_id>/`, where `<plugin_id>` is the
   exact `PluginManifestV4.aiName`.
2. Rename Swift types and files:
   - `DemoPluginTemplate` -> `<PluginName>Plugin`
   - `DemoPluginTemplateMetal` -> `<PluginName>Metal`
   - `DemoPluginParameterTemplate` and configuration types -> plugin-specific
     names when the plugin owns custom parameter schema.
3. Replace `aiID`, `aiName`, `family`, capability mask, feature groups, horizon
   limits, and sequence limits in the manifest.
4. Replace the CPU no-trade prediction with deterministic reference logic. CPU
   remains the fallback and parity authority for every accelerator.
5. Keep only runtime folders that the plugin actually uses. Remove undeclared
   runtime folders, or keep them out of `accelerationPlan`.
6. For PyTorch, TensorFlow, NLP, and ONNX runtime paths, keep the plugin-local
   file naming convention:
   - `PyTorch/<plugin_id>_torch.py`
   - `TensorFlow/<plugin_id>_tensorflow.py`
   - `NLP/<plugin_id>_nlp.py`
   - `ONNX/<plugin_id>.onnx`
   - `ONNX/<plugin_id>.manifest.json`
   PyTorch and TensorFlow modules must expose `predict_batch` and `train_step`.
   New neural backends should keep `financial_targets` and
   `financial_loss_config` keyword parameters on `train_step`. The dispatcher
   passes them only when the backend opts in, and the values map to
   FXDataEngine's `MLFinancialTrainingTargets` and `MLFinancialLossSpec`.
   Import `fxai_financial_loss.py` from `FXPlugins/API/Backends/Python/` to use
   the shared hybrid objective instead of a pure MSE/accuracy or pure
   Sharpe/Sortino loss.
   Foundation NLP modules must expose `merge_into_numeric_features`; keeping
   `predict_batch` and `train_step` helpers in the module makes the runtime
   surface consistent for future bridge changes.
7. For Remote RPC, implement the external server and configure:
   - `FXAI_ENABLE_REMOTE_RPC=1`
   - `FXAI_REMOTE_INFERENCE_ENDPOINT`
   - optional `FXAI_REMOTE_INFERENCE_AUTH_TOKEN`
   - optional `FXAI_REMOTE_INFERENCE_TIMEOUT_SECONDS`
8. Add the plugin to `FXAIPluginRegistry.availablePlugins()` only after the CPU
   reference path validates and every declared runtime has evidence.
9. Add focused tests under `FXPlugins/API/Tests/FXAIPluginsTests/` for manifest,
   CPU prediction, runtime folders, and any declared accelerator.
10. Update `FXPluginCertificationRegistry` evidence only when the required live
    certification gates are genuinely satisfied.

## Parameter Contract

Configuration data belongs in FXDatabase ClickHouse tables through the
versioned FXBacktest API. New plugins should expose default, minimum, step, and
maximum values for every plugin and accelerator parameter. The template's
configuration rows are neutral examples only.

## Python Runtime Baseline

Plugin-local Python backends run through the FXDataEngine Python bridge. The
current backend baseline is:

- Python 3.12.x.
- `torch==2.12.0` with MPS for PyTorch accelerator runtime paths.
- `tensorflow==2.18.1` and `tensorflow-metal==1.2.0` for TensorFlow Metal.
- `onnxruntime==1.26.0` for ONNX Runtime.

Set `FXAI_PYTHON` to the Python 3.12 executable when running plugin tests
manually. The runtime resolver must not fall back to generic Homebrew `python3`.

## Financial Training Objective

FXDataEngine serializes trading-aware targets into each `MLTrainingPayload`:
signed realized move, sample weight, MFE/MAE, hit timing, path flags,
path/fill risk, min-move, and price-cost context. PyTorch and TensorFlow
plugins that accept `financial_targets` and `financial_loss_config` should use
the shared helper in `FXPlugins/API/Backends/Python/fxai_financial_loss.py`.

The default objective is intentionally hybrid:

- weighted class CE for direction/skip labels,
- Huber and pinball losses for expected move and quantiles,
- high penalty for wrong active direction during large adverse moves,
- cost, path-risk, and fill-risk penalty on active trade probability,
- activity discipline to avoid always-trade and always-skip collapse,
- downside utility regularization.

Sharpe and Sortino remain walk-forward validation and promotion metrics. Do not
make a pure batch Sharpe/Sortino ratio the primary plugin training loss without
a separate governance ticket and walk-forward evidence.

## Verification

Run the focused checks before adding a new plugin to the registry:

```bash
git diff --check
swift test --package-path FXPlugins --filter PluginBoundaryTests
swift test --package-path FXPlugins --filter PluginZooLayoutTests
FXAI_PYTHON=/opt/homebrew/bin/python3.12 swift test --package-path FXPlugins --filter PluginRuntimeIntegrationTests
FXAI_PYTHON=/opt/homebrew/bin/python3.12 swift test --package-path FXPlugins --filter PluginExternalBackendRuntimeTests
```

Before claiming production readiness, run the full plugin suite and root
certification:

```bash
FXAI_PYTHON=/opt/homebrew/bin/python3.12 swift test --package-path FXPlugins
./fxai certify --build-only
```
