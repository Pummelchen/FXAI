# Demo Plugin Template Remote RPC

This folder documents the optional external inference-server surface. Remote RPC
is inference-only in the current FXAI runtime. Training stays local unless a
separate governance ticket approves remote training.

Enable a real plugin's remote runtime with:

```bash
FXAI_ENABLE_REMOTE_RPC=1
FXAI_REMOTE_INFERENCE_ENDPOINT=https://inference.example.test/fxai/predict
FXAI_REMOTE_INFERENCE_AUTH_TOKEN=optional-bearer-token
FXAI_REMOTE_INFERENCE_TIMEOUT_SECONDS=10
```

The Swift runtime sends a JSON `RemoteRPCMLBackendRequest` with the latest
`MLInferencePayload` and expects a latest-version `RemoteRPCMLBackendResponse`
containing a valid `PredictionV4`. The first transport is HTTP JSON POST behind
`RemoteRPCMLBackendTransport`; a gRPC transport can be added later without
changing plugin declarations.

Use the example response file as the minimum server contract. A production
server should also return `modelIdentifier`, `modelVersion`, `modelSha256`,
latency, and timestamp metadata for auditability.
