#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BREW="${SKIP_BREW:-0}"
SKIP_PYTHON="${SKIP_PYTHON:-0}"
PYTHON_REQUIREMENTS_LOCK="${FXAI_PYTHON_REQUIREMENTS:-$REPO_ROOT/requirements/fxai-py312.lock}"
TMP_DIR="${TMPDIR:-/tmp}/fxai-install-$$"

mkdir -p "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

log() {
    printf '%s\n' "$*"
}

run() {
    log "+ $*"
    if [ "$DRY_RUN" != "1" ]; then
        "$@"
    fi
}

have() {
    command -v "$1" >/dev/null 2>&1
}

append_unique() {
    file="$1"
    value="$2"
    if [ -z "$value" ]; then
        return 0
    fi
    if [ ! -f "$file" ] || ! grep -Fxq "$value" "$file"; then
        printf '%s\n' "$value" >> "$file"
    fi
}

ensure_macos() {
    os_name="$(uname -s)"
    if [ "$os_name" != "Darwin" ]; then
        log "This installer is for macOS. Detected: $os_name"
        exit 1
    fi
}

ensure_apple_silicon_m2_m3() {
    machine="$(uname -m)"
    if [ "$machine" != "arm64" ]; then
        log "FXAI targets Apple Silicon M2/M3-class Macs only. Detected architecture: $machine"
        exit 1
    fi

    apple_chip="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
    if [ -z "$apple_chip" ] || [ "$apple_chip" = "Apple processor" ]; then
        apple_chip="$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Chip:/ {print $2; exit}')"
    fi

    case "$apple_chip" in
        *"Apple M2"*|*"Apple M3"*|*"Apple M4"*|*"Apple M5"*|*"Apple M6"*|*"Apple M7"*|*"Apple M8"*|*"Apple M9"*)
            log "Apple Silicon target verified: $apple_chip"
            ;;
        *"Apple M1"*)
            log "FXAI does not target Apple M1. Detected: $apple_chip"
            exit 1
            ;;
        *)
            log "Unable to verify an Apple M2/M3-or-newer chip. Detected chip string: ${apple_chip:-unknown}"
            exit 1
            ;;
    esac
}

ensure_homebrew() {
    if [ "$SKIP_BREW" = "1" ]; then
        log "Skipping Homebrew setup because SKIP_BREW=1."
        return 0
    fi

    if ! have brew; then
        log "Homebrew is missing. Installing latest Homebrew."
        run /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    if ! have brew; then
        if [ -x /opt/homebrew/bin/brew ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [ -x /usr/local/bin/brew ]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi

    if ! have brew; then
        log "Homebrew is still unavailable after install. Add it to PATH and rerun."
        exit 1
    fi
}

ensure_xcode_tools() {
    if ! xcode-select -p >/dev/null 2>&1; then
        log "Xcode Command Line Tools are missing. Starting Apple's installer."
        run xcode-select --install
        log "Finish the Apple installer, then rerun this script."
        exit 1
    fi

    if ! have swift; then
        log "swift is missing from PATH. Checking whether Homebrew can provide a swift formula."
        if have brew && brew info swift >/dev/null 2>&1; then
            run brew install swift
        fi
    fi

    if ! have swift; then
        log "swift is still missing. Install or select Xcode/Command Line Tools, then rerun."
        exit 1
    fi

    if ! xcrun -find metal >/dev/null 2>&1 || ! xcrun -find metallib >/dev/null 2>&1; then
        log "Metal command-line tools are missing. Swift runtime Metal tests can still use the macOS Metal framework, but offline metal/metallib tools require the latest Xcode for macOS 26."
    fi
}

build_brew_formula_list() {
    brew_file="$1"
    : > "$brew_file"

    append_unique "$brew_file" git
    append_unique "$brew_file" python@3.12
    append_unique "$brew_file" clickhouse
    append_unique "$brew_file" cmake
    append_unique "$brew_file" pkg-config
    append_unique "$brew_file" libomp
    append_unique "$brew_file" ripgrep
    if ! have swift; then
        append_unique "$brew_file" swift
    fi

    if grep -R "import Metal" "$REPO_ROOT" --include='*.swift' >/dev/null 2>&1; then
        log "Metal imports found. Metal itself is supplied by Xcode; no Homebrew Metal formula is required."
    fi

    if grep -R "ClickHouse" "$REPO_ROOT/FXDatabase" --include='*.swift' >/dev/null 2>&1; then
        append_unique "$brew_file" clickhouse
    fi

    sort -u "$brew_file" -o "$brew_file"
}

install_brew_formulas() {
    brew_file="$1"
    if [ "$SKIP_BREW" = "1" ]; then
        return 0
    fi

    while IFS= read -r formula; do
        [ -z "$formula" ] && continue
        if ! brew info "$formula" >/dev/null 2>&1; then
            log "Skipping unknown Homebrew formula: $formula"
            continue
        fi
        if brew list --formula "$formula" >/dev/null 2>&1; then
            log "Homebrew formula already installed: $formula"
        else
            run brew install "$formula"
        fi
    done < "$brew_file"
}

resolve_python() {
    if [ -n "${FXAI_PYTHON:-}" ]; then
        PYTHON_BIN="$FXAI_PYTHON"
    elif have python3.12; then
        PYTHON_BIN="$(command -v python3.12)"
    elif have brew; then
        brew_python312_prefix="$(brew --prefix python@3.12 2>/dev/null || true)"
        if [ -n "$brew_python312_prefix" ] && [ -x "$brew_python312_prefix/bin/python3.12" ]; then
            PYTHON_BIN="$brew_python312_prefix/bin/python3.12"
        else
            log "Python 3.12 is missing. Install Homebrew python@3.12 and rerun."
            exit 1
        fi
    else
        log "Python 3.12 is missing. Install Homebrew python@3.12 and rerun."
        exit 1
    fi
    verify_python_312 "$PYTHON_BIN"
    export PYTHON_BIN
}

verify_python_312() {
    python_executable="$1"
    "$python_executable" - <<'PY'
import sys
if sys.version_info.major != 3 or sys.version_info.minor != 12:
    raise SystemExit(f"FXAI requires Python 3.12 for tensorflow-metal compatibility; got {sys.version.split()[0]}")
PY
}

install_python_packages() {
    requirements_file="$1"
    if [ "$SKIP_PYTHON" = "1" ]; then
        log "Skipping Python package setup because SKIP_PYTHON=1."
        return 0
    fi
    if [ ! -f "$requirements_file" ]; then
        log "Python requirements lock is missing: $requirements_file"
        exit 1
    fi
    run "$PYTHON_BIN" -m pip install --requirement "$requirements_file"
}

verify_environment() {
    log "Verifying installed tools."
    run swift --version
    if xcrun -find metal >/dev/null 2>&1; then
        run xcrun -find metal
    else
        log "Warning: xcrun cannot find metal. Install the latest Xcode if offline Metal CLI compilation is needed."
    fi
    if xcrun -find metallib >/dev/null 2>&1; then
        run xcrun -find metallib
    else
        log "Warning: xcrun cannot find metallib. Install the latest Xcode if offline Metal library builds are needed."
    fi

    if [ "$SKIP_PYTHON" != "1" ]; then
        run "$PYTHON_BIN" - <<'PY'
import importlib.util
import importlib.metadata
import platform
import re
import subprocess
import sys

if sys.version_info.major != 3 or sys.version_info.minor != 12:
    raise SystemExit(f"FXAI requires Python 3.12; got {sys.version.split()[0]}")

required = ["pytest", "libsql", "certifi", "torch", "tensorflow", "onnxruntime"]
for name in required:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing Python package: {name}")

machine = platform.machine()
try:
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
except Exception:
    chip = ""
if not chip or chip == "Apple processor":
    try:
        hardware = subprocess.check_output(["system_profiler", "SPHardwareDataType"], text=True)
        match = re.search(r"Chip:\\s*(.+)", hardware)
        chip = match.group(1).strip() if match else chip
    except Exception:
        pass
if machine != "arm64" or not re.search(r"Apple M[2-9]", chip or ""):
    raise SystemExit(f"unsupported FXAI host; expected Apple M2/M3-or-newer arm64, got {machine} {chip or 'unknown'}")

import torch
if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
    raise SystemExit("PyTorch MPS is not available")
print("torch", torch.__version__, "mps_available", torch.backends.mps.is_available())

import tensorflow as tf
if tf.__version__ != "2.18.1":
    raise SystemExit(f"expected tensorflow 2.18.1, got {tf.__version__}")
if importlib.metadata.version("tensorflow-metal") != "1.2.0":
    raise SystemExit(f"expected tensorflow-metal 1.2.0, got {importlib.metadata.version('tensorflow-metal')}")
gpu_devices = tf.config.list_physical_devices("GPU")
if not gpu_devices:
    raise SystemExit("TensorFlow Metal GPU device is not available")
print("tensorflow", tf.__version__, "devices", [device.device_type for device in tf.config.list_physical_devices()])
PY
    fi
}

main() {
    ensure_macos
    ensure_apple_silicon_m2_m3
    ensure_homebrew

    brew_file="$TMP_DIR/brew-formulas.txt"
    build_brew_formula_list "$brew_file"

    log "Homebrew formulas discovered:"
    sed 's/^/  - /' "$brew_file"

    install_brew_formulas "$brew_file"
    ensure_xcode_tools
    resolve_python

    log "Python requirements lock:"
    log "  - $PYTHON_REQUIREMENTS_LOCK"
    if [ ! -f "$PYTHON_REQUIREMENTS_LOCK" ]; then
        log "Python requirements lock is missing: $PYTHON_REQUIREMENTS_LOCK"
        exit 1
    fi
    sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d;s/^/  - /' "$PYTHON_REQUIREMENTS_LOCK"

    install_python_packages "$PYTHON_REQUIREMENTS_LOCK"
    verify_environment

    log "FXAI macOS 26 environment setup complete."
}

main "$@"
