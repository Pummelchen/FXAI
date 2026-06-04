# Demo Plugin Template ONNX Runtime

This folder documents the plugin-local ONNX Runtime surface. It is a template
only; do not add `onnxRuntime` to a production plugin acceleration plan until a
real exported `.onnx` model exists and live prediction evidence passes.

Expected production layout after copying the template:

```text
FXPlugins/<plugin_id>/ONNX/<plugin_id>.onnx
FXPlugins/<plugin_id>/ONNX/<plugin_id>.manifest.json
```

The Python dispatcher loads `<plugin_id>.onnx` when the plugin declares
`onnxRuntime` and `FXAI_ENABLE_ONNX_RUNTIME=1` is set. Runtime overrides are:

- `FXAI_ONNX_MODEL_PATH`
- `FXAI_ONNX_MANIFEST_PATH`
- `FXAI_ONNX_PROVIDERS`

The manifest is optional but should pin names and hashes for reproducibility.
Use `demo_plugin_template.manifest.json` as the schema example and replace the
placeholder `modelSha256` with the real SHA-256 of the exported model.
