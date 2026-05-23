# Installation

FXAI targets macOS 26 with the current Apple Swift toolchain, Metal, Homebrew, Python 3, PyTorch, TensorFlow, and the Python packages used by repo tools and plugin accelerators.

## Installer

From the repo root:

```bash
Scripts/install_macos26.sh
```

Dry run:

```bash
DRY_RUN=1 Scripts/install_macos26.sh
```

The installer is Bash 3 compatible. It:

- Installs or checks Homebrew.
- Checks Xcode Command Line Tools for `swift`, `metal`, and `metallib`.
- Installs Homebrew packages without version pins.
- Scans the repo for Python imports.
- Installs Python packages without version pins.
- Verifies Swift, Metal, PyTorch MPS, TensorFlow devices, pytest, libSQL, and certifi.

## What The Installer Does Not Pin

FXAI intentionally avoids hard dependency versions in the installer. The goal is to follow the latest local macOS, Swift, Homebrew, PyTorch, and TensorFlow packages unless a future reproducibility gate requires a locked environment file.

## Manual Checks

```bash
swift --version
xcrun -find metal
xcrun -find metallib
python3 -m pytest --version
python3 -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
python3 -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices())"
```

## Full Swift Verification

```bash
swift test --package-path FXDatabase
swift test --package-path FXDataEngine
swift test --package-path FXPlugins
swift test --package-path FXBacktest
```
