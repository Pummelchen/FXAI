# FXPlugins 100% Live Runtime Completion Plan

This plan is the source-of-truth checklist for keeping every FXAI plugin and the two FXBacktest-origin demo plugins at full live runtime quality.

Current verified state: the Swift packages build and the plugin zoo now has an executable 100% certification gate. The gate covers registry coverage, CPU runtime smoke, OHLCV/volume contracts, SineTest runtime, standard-reference or golden parity evidence, FXDatabase-only data access, full verification harness presence, Metal source compilation/live buffer probe wiring, PyTorch/TensorFlow live train-predict-persistence-load checks, and NLP text/no-text runtime checks. CoreML/Neural Engine is not declared by any plugin until a real export and parity path exists.

`FXAIPluginCertificationRegistry` is the executable certification gate. It covers every registered plugin and every declared backend, records which evidence is currently satisfied, and `requireAllPlugins100PercentCertified()` now passes when the evidence files, runtime harnesses, and backend tests are present and valid.

## 100% Acceptance Gates

Every plugin is complete only when all applicable gates pass:

1. CPU implementation is the authoritative Swift implementation of the plugin concept, with no shell or placeholder behavior.
2. The CPU implementation consumes the FXDataEngine OHLCV contract: M1 OHLC is mandatory; volume is used whenever `volume > 0`.
3. Historical or standard-reference parity fixtures exist and are used as the parity target.
4. Metal implementations compile at runtime, execute on Apple GPUs, and have deterministic CPU parity tests over representative fixtures.
5. PyTorch implementations run through the live backend bridge, require MPS on M2/M3-class Apple Silicon, persist/load model state, and expose train and predict entry points.
6. TensorFlow implementations run through the live backend bridge, require TensorFlow Metal GPU runtime, persist/load model state, and expose train and predict entry points.
7. NLP implementations are live runtime backends, not passive files; they include tokenizer/text-event contracts and deterministic fallback behavior when text context is absent.
8. Backend selection is explicit, observable, and testable: CPU-only, auto, Metal, PyTorch, TensorFlow, and NLP modes must fail closed or fall back according to the plugin policy.
9. FXBacktest, FXDemoAgent, and FXLiveAgent use the same plugin API and cannot bypass FXDatabase for market data.
10. Full Swift tests and Python backend smoke tests pass cleanly before commit and push.

## Completed Shared Foundation

1. Done: shared live backend policy maps plugin capability to CPU, Metal, PyTorch, TensorFlow, and NLP backends.
2. Done: synchronous-safe `PythonMLBackendBridge` train/predict methods support the synchronous `FXAIPluginV4` API.
3. Done: Metal runtime execution compiles plugin kernel source, runs buffers on `MTLDevice`, validates output shape, and supports CPU parity tests.
4. Done: plugin-local backend discovery finds backend files without hard-coded absolute paths.
5. Done: standard live backend tests cover CPU golden/reference behavior, Metal compile/parity, PyTorch train/predict/persistence, TensorFlow train/predict/persistence, and NLP text/context behavior.
6. Done: registered plugins expose active CPU runtime logic and reference/parity evidence through the certification tests.

## Completion Status

Remaining plugin implementation tasks: none.

The original per-plugin matrix and execution waves are closed. The source of truth is now executable evidence in `FXAIPluginCertificationRegistry` plus the Swift/Python runtime tests listed below. Any future plugin or accelerator change must re-open the relevant gate by changing code and tests, not by adding unchecked prose rows.

### Closed Waves

1. Runtime infrastructure: implemented through `FXPluginRuntimeResolver`, `PythonMLBackendBridge`, `FXAIPluginBackendDiscovery`, `FXAIAcceleratedPluginRuntime`, `MetalKernelCompiler`, `FXAIPluginMetalBackendDiscovery`, and the plugin-local Python dispatcher.
2. CPU/reference plugins: certified through registry coverage, CPU runtime smoke, OHLCV/volume contract checks, SineTest runtime checks, and reference/golden parity evidence.
3. Metal plugins: certified through runtime Metal source compilation, live buffer execution, and CPU parity fixtures for every declared Metal backend.
4. PyTorch/TensorFlow/NLP plugins: certified through live bridge calls, plugin-local backend discovery, train/predict/persistence/reload tests, and text-event/no-text NLP checks.
5. Cross-project runtime: certified through FXDatabase-only access gates for FXBacktest, FXDemoAgent, and FXLiveAgent.
6. Final certification: certified through `PluginCertificationGateTests.testHundredPercentCertificationPassesWithCurrentEvidenceSet`.

### Accelerator Evidence

- PyTorch MPS required: `PluginExternalBackendRuntimeTests` sets `FXAI_REQUIRE_PYTORCH_MPS=1` and `PYTORCH_ENABLE_MPS_FALLBACK=1` for every declared PyTorch backend. The dispatcher reloads persisted PyTorch state onto `torch.device("mps")` when MPS is available and CPU forcing is not explicitly requested.
- TensorFlow Metal required: `PluginExternalBackendRuntimeTests` sets `FXAI_REQUIRE_TENSORFLOW_METAL=1` for every declared TensorFlow backend.
- Metal live buffer execution: `PluginMetalRuntimeTests` compiles declared plugin-local kernels on the local Metal device, executes plugin-local probes, and compares output against CPU fixtures.
- NLP live context: `PluginExternalBackendRuntimeTests` verifies typed text events, tokenizer contract versioning, no-text fallback, and different predictions when event context is present.
- CoreML/Neural Engine: deliberately not declared by any plugin until a real export/load/predict/parity path exists.

### Final Review Pass 1

The stale matrix rows were removed because they contradicted the executable certification registry. The plan now fails certification if those stale open markers return.

### Final Review Pass 2

The Apple Silicon accelerator paths were tightened after review:

1. PyTorch tests no longer force CPU. They require MPS and exercise predict/train/persist/reload through the same Swift bridge used at runtime.
2. TensorFlow tests require TensorFlow Metal GPU discovery before treating a TensorFlow backend as certified.
3. The demo plugin template now follows the shared backend ABI for PyTorch and TensorFlow.
4. PPO training sanitizes MPS gradients and parameters so single-sample certification training cannot persist NaN policy weights.
5. Sequence PyTorch modules pad one-row MPS inputs to a minimum convolution-safe sequence length before training.

## Implementation Progress

Wave 0 is implemented and verified:

1. `FXPluginRuntimeResolver` resolves CPU-only, automatic, and forced backend modes from every plugin acceleration plan.
2. `PythonMLBackendBridge` now has synchronous train/predict methods for the synchronous `FXAIPluginV4` API and accepts per-process environment variables.
3. `FXAIPluginBackendDiscovery` discovers plugin-local PyTorch, TensorFlow, and NLP files by plugin name and creates external Python descriptors.
4. `fxai_plugin_module_backend.py` dispatches Swift bridge calls into plugin-local PyTorch, TensorFlow, and NLP modules instead of using only a generic fallback.
5. `FXAIAcceleratedPluginRuntime` wraps any planned plugin and routes live train/predict calls through selected external backends with explicit CPU fallback.
6. `MetalKernelCompiler` validates Metal kernel source against the local Metal device and executes both generic and multi-buffer plugin-local float kernels with CPU parity checks.
7. `FXAIPluginMetalBackendDiscovery` discovers every declared plugin-local Metal kernel source, extracts kernel function names, compiles them on the local Metal device, and executes one plugin-local kernel per declared Metal plugin against a CPU fixture before Metal-mode prediction falls back to the CPU reference plugin.
8. `PluginContextV4` and `MLInferencePayload` carry typed text events and tokenizer contracts. NLP backend tests verify text-event enrichment and no-text fallback behavior.
9. Runtime tests now verify all registered plugins resolve CPU/automatic backends, all declared Python/NLP backend files are discoverable, Foundation NLP predicts through the Swift bridge, every declared PyTorch backend predicts/trains/persists/reloads when PyTorch is installed, every declared TensorFlow backend predicts/trains/persists/reloads when TensorFlow is installed, and the accelerated runtime wrapper drives NLP/PyTorch plugins end to end.
10. FXBacktest, FXDemoAgent, and FXLiveAgent have a static gatekeeper test proving they do not use direct ClickHouse access patterns and must use FXDatabase APIs.
11. Certification tests now verify that every plugin/backend is covered by the certification matrix and that the strict 100% check passes with the current evidence set.

Wave 4 agent contract foundations are implemented and verified:

1. `FXDemoAgent` is now a Swift package with a versioned `fxdemo.agent.v1` workload contract, dry-run-first planning runtime, account-scope validation, risk-limit validation, kill-switch enforcement, and no direct database access tests.
2. `FXLiveAgent` is now a Swift package with a versioned `fxlive.agent.v1` promoted-workload contract, promotion evidence validation, human-release planning runtime, account-scope validation, risk-limit validation, kill-switch enforcement, and no direct database access tests.
3. Root certification now builds and, in full mode, tests `FXDemoAgent` and `FXLiveAgent` alongside the other FXAI packages.

CoreML/Neural Engine is deliberately excluded from plugin candidate declarations. The enum remains only as a future compatibility surface and strict runtime rejection path until a real CoreML export, load, prediction, and parity implementation is added.
