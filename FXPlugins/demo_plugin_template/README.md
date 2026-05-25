# Demo Plugin Template

This folder is a compile-checked template for new FXAI plugins. It is not part
of `FXAIPluginRegistry.availablePlugins()` and must not be used as a trading
strategy until its no-trade shells are replaced with real model logic.

Required structure:

- `CPU/`: Swift plugin contract, manifest, configuration rows, CPU fallback.
- `Metal/`: Metal descriptor and kernel source.
- `PyTorch/`: PyTorch/MPS train and predict scaffold.
- `TensorFlow/`: TensorFlow Metal train and predict scaffold.
- `NLP/`: text/event feature scaffold.

Configuration data belongs in FXDatabase ClickHouse tables through the
versioned FXBacktest API. New plugins should expose default, minimum, step, and
maximum values for every plugin and accelerator parameter.
