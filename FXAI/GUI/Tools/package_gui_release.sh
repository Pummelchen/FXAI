#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FXAI_ROOT="$(cd "${GUI_ROOT}/.." && pwd)"
OUTPUT_ROOT="${1:-${GUI_ROOT}/Artifacts/Release}"
VERSION="${2:-${FXAI_GUI_VERSION:-0.7.0}}"
APP_NAME="FXAI GUI"
BUNDLE_NAME="FXAIGUI.app"
BUILD_DIR="${GUI_ROOT}/.build/release"
EXECUTABLE_PATH="${BUILD_DIR}/FXAIGUI"
BUNDLE_ROOT="${OUTPUT_ROOT}/${BUNDLE_NAME}"
MACOS_DIR="${BUNDLE_ROOT}/Contents/MacOS"
RESOURCES_DIR="${BUNDLE_ROOT}/Contents/Resources"
INFO_PLIST="${BUNDLE_ROOT}/Contents/Info.plist"
GIT_SHA="$(git -C "${GUI_ROOT}/../.." rev-parse --short HEAD 2>/dev/null || printf 'unknown')"

mapfile -t GUI_CONFIG < <(FXAI_ROOT="${FXAI_ROOT}" python3 - <<'PY'
from __future__ import annotations
import os
from pathlib import Path
import sys

root = Path(os.environ["FXAI_ROOT"])
sys.path.insert(0, str(root / "Tools"))
from testlab.toolchain import load_toolchain_config

config = load_toolchain_config(project_root_hint=root)
print(config.profile)
print(config.gui_minimum_macos)
print(config.gui_release_archive)
PY
)

TOOLCHAIN_PROFILE="${GUI_CONFIG[0]:-headless_ci}"
MINIMUM_MACOS="${FXAI_GUI_MINIMUM_MACOS:-${GUI_CONFIG[1]:-14.0}}"
ARCHIVE_NAME="${FXAI_GUI_RELEASE_ARCHIVE:-${GUI_CONFIG[2]:-FXAIGUI-macos.zip}}"
ARCHIVE_PATH="${OUTPUT_ROOT}/${ARCHIVE_NAME}"

mkdir -p "${OUTPUT_ROOT}"

printf 'Building FXAIGUI in release mode...\n'
printf 'Using FXAI toolchain profile: %s\n' "${TOOLCHAIN_PROFILE}"
swift build --package-path "${GUI_ROOT}" -c release

if [[ ! -x "${EXECUTABLE_PATH}" ]]; then
  printf 'error: expected release executable not found at %s\n' "${EXECUTABLE_PATH}" >&2
  exit 1
fi

rm -rf "${BUNDLE_ROOT}" "${ARCHIVE_PATH}"
mkdir -p "${MACOS_DIR}" "${RESOURCES_DIR}"

cp "${EXECUTABLE_PATH}" "${MACOS_DIR}/FXAIGUI"
chmod +x "${MACOS_DIR}/FXAIGUI"

cat > "${INFO_PLIST}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>FXAIGUI</string>
  <key>CFBundleIdentifier</key>
  <string>com.pummelchen.fxai.gui</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>${APP_NAME}</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>${VERSION}</string>
  <key>CFBundleVersion</key>
  <string>${GIT_SHA}</string>
  <key>LSMinimumSystemVersion</key>
  <string>${MINIMUM_MACOS}</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
PLIST

ditto -c -k --keepParent "${BUNDLE_ROOT}" "${ARCHIVE_PATH}"

printf 'Created app bundle: %s\n' "${BUNDLE_ROOT}"
printf 'Created zip archive: %s\n' "${ARCHIVE_PATH}"
