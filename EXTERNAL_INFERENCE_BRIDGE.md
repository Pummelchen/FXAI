# FXAI External Inference Bridge Plan

## Goal

Implement an optional external model inference bridge for FXAI that lets plugins declare and use models outside the in-process Swift/Metal/Python training paths. MT5 is out of scope for this work; it remains only a possible data source. The bridge belongs in the FXAI plugin runtime and must preserve the current deterministic CPU fallback, certification, and fallback diagnostics behavior.

## Scope

1. Add two new optional plugin backend declarations:
   - `onnxRuntime`: a Python-hosted ONNX Runtime inference backend for exported `.onnx` models.
   - `remoteRPC`: a Swift-native remote inference client backend for GPU-backed inference servers.
2. Keep both new backends opt-in. Existing plugins must not change runtime behavior unless they explicitly declare one of the new backends and the runtime environment enables it.
3. Make the first `remoteRPC` transport JSON over HTTP POST to avoid adding a heavy gRPC dependency to the core Swift package. The public Swift transport protocol must be narrow enough to allow a later gRPC adapter without changing plugin/runtime semantics.
4. Make both new backends inference-first. Training stays on the base plugin unless a future backend explicitly declares safe remote training semantics.
5. Preserve existing safety posture:
   - No command-shell expansion.
   - Strict endpoint and environment validation.
   - Explicit environment enablement.
   - Prediction API version validation.
   - CPU fallback diagnostics on external inference failure.

## Detailed Steps

### Step 0 - Branch And Repository Constraints

1. Stay on `main`.
2. Do not create or switch branches.
3. Check `git status` before and after implementation.
4. Do not revert unrelated local changes if any appear.

### Step 1 - Backend Contract Expansion

1. Extend `MLFramework` with:
   - `onnxRuntime`
   - `remoteRPC`
2. Extend `MLBackendMode` with a remote descriptor:
   - `remoteRPC(endpoint: String)`
3. Keep `MLInferencePayload` unchanged so external servers receive the same feature, sequence, volume, tokenizer, and text-event payloads as current Python backends.
4. Update `MLBackendFactory.framework(for:)` so remote descriptors produce `.remoteRPC`.
5. Update the Python bridge framework allowlist to accept `.onnxRuntime`.

### Step 2 - Remote RPC Swift Bridge

1. Add a `RemoteRPCMLBackendConfiguration` value type with:
   - endpoint URL
   - optional auth token
   - timeout seconds
2. Add a `RemoteRPCMLBackendRequest` value type with:
   - latest FXAI plugin API version
   - operation (`predict`)
   - inference payload
3. Add a `RemoteRPCMLBackendResponse` value type with:
   - latest FXAI plugin API version
   - success flag
   - optional `PredictionV4`
   - optional error string
   - optional model metadata and latency fields for diagnostics
4. Add `RemoteRPCMLBackendTransport` protocol so tests can inject a fake transport and future work can add gRPC without changing runtime code.
5. Add `URLSessionRemoteRPCMLBackendTransport`:
   - Only allow `http` and `https` URLs.
   - Send JSON POST with `Content-Type: application/json`.
   - Add `Authorization: Bearer <token>` only when a nonempty token is configured.
   - Enforce timeout.
   - Reject non-2xx HTTP status codes.
6. Add `RemoteRPCMLBackendBridge` implementing `ExternalMLBackend`:
   - Validate latest API request.
   - Validate latest API response.
   - Validate `PredictionV4`.
   - Reject training as unsupported.

### Step 3 - Plugin Runtime Backend Policy

1. Extend `FXPluginAccelerationBackend` with:
   - `onnxRuntime`
   - `remoteRPC`
2. Extend `FXPluginRuntimeMode` with matching forced modes.
3. Extend `FXPluginRuntimeEnvironment` with:
   - `onnxRuntimeAvailable`
   - `remoteInferenceAvailable`
   - `remoteInferenceEndpoint`
   - `remoteInferenceAuthToken`
   - `remoteInferenceTimeoutSeconds`
4. Read local runtime flags:
   - `FXAI_ENABLE_ONNX_RUNTIME=1`
   - `FXAI_ENABLE_REMOTE_RPC=1`
   - `FXAI_REMOTE_INFERENCE_ENDPOINT`
   - `FXAI_REMOTE_INFERENCE_AUTH_TOKEN`
   - `FXAI_REMOTE_INFERENCE_TIMEOUT_SECONDS`
5. Support rules:
   - `onnxRuntime` requires a Python executable and explicit ONNX enablement.
   - `remoteRPC` requires explicit remote enablement and a nonempty endpoint.
6. Automatic preference should consider new explicitly declared external inference backends before existing local Python GPU backends:
   - `remoteRPC`
   - `onnxRuntime`
   - `pyTorchMPS`
   - `tensorFlowMetal`
   - `metal`
   - `foundationNLP`
   - `accelerate`
   - `swiftSIMD`
   - `swiftScalar`

### Step 4 - FXPlugins Backend Discovery

1. Add ONNX discovery:
   - Plugin model path: `FXPlugins/<pluginName>/ONNX/<pluginName>.onnx`
   - Optional manifest path: `FXPlugins/<pluginName>/ONNX/<pluginName>.manifest.json`
2. Return an external Python descriptor for `onnxRuntime` using the existing module backend script and `supportsTraining=false`.
3. Return no Python descriptor for `remoteRPC`; it uses the Swift bridge.

### Step 5 - ONNX Python Runtime

1. Extend `fxai_plugin_module_backend.py` to recognize framework `onnxRuntime`.
2. Load ONNX models from:
   - `FXAI_ONNX_MODEL_PATH` when present, otherwise
   - the plugin-local ONNX path from Step 4.
3. Load optional model manifest from:
   - `FXAI_ONNX_MANIFEST_PATH` when present, otherwise
   - `<model>.manifest.json`.
4. Validate manifest fields when present:
   - plugin name must match the requested model identifier.
   - model SHA-256 must match the `.onnx` file when provided.
   - optional input/output names may override ONNX Runtime defaults.
5. Use `onnxruntime.InferenceSession`.
6. Convert `MLInferencePayload.xWindow + x` into a float32 tensor.
7. Normalize common output shapes:
   - 3-class probabilities or logits.
   - Dictionary/object outputs when possible.
8. Return a latest-version `PredictionV4` JSON payload.
9. Return `ok=true` and no prediction for training because ONNX is inference-only in this slice.

### Step 6 - Accelerated Runtime Dispatch

1. Prediction:
   - `onnxRuntime` uses the external Python path.
   - `remoteRPC` uses the Swift remote bridge.
   - Both participate in the existing CPU fallback diagnostic path on failure.
2. Training:
   - `onnxRuntime` delegates to the base plugin.
   - `remoteRPC` delegates to the base plugin.
3. Keep cycle-direction adjustment after prediction so external predictions pass through the same intrahour calibration layer as local predictions.

### Step 7 - Certification Gates

1. Add certification gates:
   - `onnxRuntimeModelDiscovery`
   - `onnxRuntimeLivePredict`
   - `remoteRPCConfiguration`
   - `remoteRPCLivePredict`
2. Required gates:
   - `onnxRuntime` requires external backend discovery, model discovery, and live predict evidence.
   - `remoteRPC` requires remote configuration and live predict evidence.
3. Satisfaction evidence:
   - ONNX discovery evidence comes from a plugin-local `.onnx` model file.
   - ONNX live evidence comes from runtime tests referencing ONNX Runtime.
   - Remote configuration evidence comes from runtime tests covering endpoint configuration.
   - Remote live evidence comes from runtime tests covering bridge prediction.

### Step 8 - Tests

1. Add `FXDataEngine` tests for runtime policy:
   - `onnxRuntime` is selected only when enabled and Python exists.
   - `remoteRPC` is selected only when enabled and endpoint exists.
   - Forced modes fallback or throw consistently with current policy.
2. Add `FXDataEngine` tests for `RemoteRPCMLBackendBridge`:
   - Success path with a fake transport.
   - Unsupported training path.
   - Invalid endpoint rejection.
   - Response API version rejection.
3. Update existing Python bridge configuration tests to include `.onnxRuntime`.
4. Update exhaustive test switches that map backend to runtime mode.
5. Update Python token tests that assert backend framework expansion.

### Step 9 - Documentation

1. Update root `README.md` with:
   - New optional external inference bridge.
   - Environment variables.
   - Inference-only first-slice limitation.
2. Update `FXPlugins/README.md` with plugin author guidance:
   - How to declare `onnxRuntime` and `remoteRPC`.
   - Expected ONNX folder layout.
   - Remote RPC response contract summary.
3. Update governance docs with release gates for external inference.
4. Update git wiki where operator-facing runtime or governance information exists.

### Step 10 - Verification

1. Run focused Swift tests for:
   - ML backend bridge.
   - runtime backend policy.
   - plugin runtime/certification compile touchpoints.
2. Run Python backend-framework token tests.
3. Run a final build/compile command for the touched Swift packages.
4. Run `git diff --check`.
5. Manually review the changed code against this plan.
6. Do a bug audit over:
   - API version handling.
   - endpoint validation.
   - data payload shape.
   - fallback behavior.
   - certification gate logic.
   - docs matching code.

### Step 11 - Commit And Push

1. Confirm `main` is still checked out.
2. Commit implementation, tests, and documentation.
3. Push `main`.

## Plan Review And Corrections

1. **Correction: ignore MT5-specific implementation.** The original feature text mentioned MT5 execution and gating, but the current project direction treats MT5 only as a data provider. The bridge must therefore be implemented in FXAI plugin runtime contracts, not MQL5.
2. **Correction: do not force gRPC in the first slice.** Adding gRPC directly would introduce dependency and packaging risk across the Swift packages. The implemented remote bridge should use a transport protocol plus JSON/HTTP default transport. A future gRPC adapter can conform to the same protocol.
3. **Correction: keep training local.** Remote training has different reproducibility and checkpoint-governance risks. This slice is inference-only for `onnxRuntime` and `remoteRPC`.
4. **Correction: do not change existing plugin behavior.** New backends must be inert until a plugin declares them and the environment enables them.
5. **Correction: add certification gates rather than reusing PyTorch/TensorFlow gates.** ONNX and remote inference have different evidence requirements.
6. **Correction: keep ONNX behind Python.** The repo already has a Python bridge and Python runtime policy; using Python-hosted ONNX Runtime avoids adding a Swift ONNX runtime dependency.
7. **Correction: avoid test-only runtime shortcuts.** Remote bridge tests should use dependency injection through `RemoteRPCMLBackendTransport`, not hidden mock endpoint schemes in production code.

## Implementation Status

1. Backend contract expansion: implemented in `MLFramework`, `MLBackendMode`, `MLBackendFactory`, and `PythonMLBackendBridge`.
2. Remote RPC bridge: implemented as `RemoteRPCMLBackendConfiguration`, request/response DTOs, `RemoteRPCMLBackendTransport`, `URLSessionRemoteRPCMLBackendTransport`, and `RemoteRPCMLBackendBridge`.
3. Runtime backend policy: implemented through new `onnxRuntime` and `remoteRPC` acceleration/runtime modes, explicit environment gates, and remote configuration helpers.
4. Backend discovery: implemented for plugin-local `ONNX/<plugin>.onnx` assets; `remoteRPC` intentionally has no Python descriptor.
5. ONNX Python runtime: implemented in `FXPlugins/API/Backends/Python/fxai_plugin_module_backend.py` with model path overrides, optional manifest validation, provider selection, tensor conversion, and `PredictionV4` normalization.
6. Accelerated runtime dispatch: implemented with ONNX routed through the Python bridge and remote RPC routed through the Swift bridge. Training remains local for both new backends.
7. Certification gates: implemented as ONNX model/live-predict and remote configuration/live-predict gates.
8. Tests: implemented for remote bridge success/failure paths, ONNX Python-bridge acceptance, runtime policy selection, exhaustive backend switches, and documentation/token guards.
9. Documentation: updated in the root README, `FXPlugins/README.md`, governance, and this bridge document.
10. Intentional follow-up: native gRPC transport is not part of this first slice. The transport protocol is the compatibility point for adding it later.
