#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BREW="${SKIP_BREW:-0}"
SKIP_PYTHON="${SKIP_PYTHON:-0}"
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
    append_unique "$brew_file" python
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
    elif have python3; then
        PYTHON_BIN="$(command -v python3)"
    elif have brew && [ -x "$(brew --prefix python 2>/dev/null)/bin/python3" ]; then
        PYTHON_BIN="$(brew --prefix python)/bin/python3"
    else
        log "python3 is missing. Install Homebrew python and rerun."
        exit 1
    fi
    export PYTHON_BIN
}

scan_python_imports() {
    imports_file="$1"
    "$PYTHON_BIN" - "$REPO_ROOT" > "$imports_file" <<'PY'
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
scan_roots = [
    root / "FXPlugins",
    root / "FXDataEngine" / "Tools",
    root / "FXImporter",
    root / "FXDatabase",
    root / "FXBacktest",
]
local_modules = {"offline_lab", "testlab", "fxai_testlab", "fxai_offline_lab"}
stdlib = set(getattr(sys, "stdlib_module_names", ()))
modules = set()

for scan_root in scan_roots:
    if not scan_root.exists():
        continue
    for path in scan_root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules.add(node.module.split(".")[0])

for name in sorted(modules):
    if name not in stdlib and name not in local_modules and not name.startswith("_"):
        print(name)
PY
}

build_python_package_list() {
    imports_file="$1"
    package_file="$2"
    optional_file="$3"
    : > "$package_file"
    : > "$optional_file"

    append_unique "$package_file" pip
    append_unique "$package_file" setuptools
    append_unique "$package_file" wheel

    while IFS= read -r module; do
        case "$module" in
            torch)
                append_unique "$package_file" torch
                ;;
            tensorflow)
                append_unique "$package_file" tensorflow-macos
                append_unique "$optional_file" tensorflow-metal
                ;;
            pytest)
                append_unique "$package_file" pytest
                ;;
            libsql)
                append_unique "$package_file" libsql
                ;;
            certifi)
                append_unique "$package_file" certifi
                ;;
            "")
                ;;
            *)
                append_unique "$package_file" "$module"
                ;;
        esac
    done < "$imports_file"

    sort -u "$package_file" -o "$package_file"
    sort -u "$optional_file" -o "$optional_file"
}

install_python_packages() {
    package_file="$1"
    optional_file="$2"
    if [ "$SKIP_PYTHON" = "1" ]; then
        log "Skipping Python package setup because SKIP_PYTHON=1."
        return 0
    fi

    while IFS= read -r package_name; do
        [ -z "$package_name" ] && continue
        run "$PYTHON_BIN" -m pip install --upgrade "$package_name"
    done < "$package_file"

    while IFS= read -r package_name; do
        [ -z "$package_name" ] && continue
        log "Installing Python accelerator package: $package_name"
        if [ "$DRY_RUN" = "1" ]; then
            log "+ $PYTHON_BIN -m pip install --upgrade $package_name"
        else
            run "$PYTHON_BIN" -m pip install --upgrade "$package_name"
        fi
    done < "$optional_file"
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
import platform
import re
import subprocess

required = ["pytest", "libsql", "certifi", "torch", "tensorflow"]
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

    imports_file="$TMP_DIR/python-imports.txt"
    package_file="$TMP_DIR/python-packages.txt"
    optional_file="$TMP_DIR/python-optional-packages.txt"

    scan_python_imports "$imports_file"
    build_python_package_list "$imports_file" "$package_file" "$optional_file"

    log "Python imports discovered:"
    sed 's/^/  - /' "$imports_file"
    log "Python packages to install:"
    sed 's/^/  - /' "$package_file"
    if [ -s "$optional_file" ]; then
        log "Python accelerator packages:"
        sed 's/^/  - /' "$optional_file"
    fi

    install_python_packages "$package_file" "$optional_file"
    verify_environment

    log "FXAI macOS 26 environment setup complete."
}

main "$@"
