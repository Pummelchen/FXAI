#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${1:-${GUI_ROOT}/Artifacts/Release}"
APP_NAME="FXAI GUI"
BUNDLE_NAME="FXAIGUI.app"
BUILD_DIR="${GUI_ROOT}/.build/release"
EXECUTABLE_PATH="${BUILD_DIR}/FXAIGUI"
BUNDLE_ROOT="${OUTPUT_ROOT}/${BUNDLE_NAME}"
MACOS_DIR="${BUNDLE_ROOT}/Contents/MacOS"
RESOURCES_DIR="${BUNDLE_ROOT}/Contents/Resources"
INFO_PLIST="${BUNDLE_ROOT}/Contents/Info.plist"
ARCHIVE_PATH="${OUTPUT_ROOT}/FXAIGUI-macos26.zip"
GIT_SHA="$(git -C "${GUI_ROOT}/../.." rev-parse --short HEAD 2>/dev/null || printf 'unknown')"

mkdir -p "${OUTPUT_ROOT}"

printf 'Building FXAIGUI in release mode...\n'
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
  <string>0.6.0</string>
  <key>CFBundleVersion</key>
  <string>${GIT_SHA}</string>
  <key>LSMinimumSystemVersion</key>
  <string>26.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
PLIST

ditto -c -k --keepParent "${BUNDLE_ROOT}" "${ARCHIVE_PATH}"

printf 'Created app bundle: %s\n' "${BUNDLE_ROOT}"
printf 'Created zip archive: %s\n' "${ARCHIVE_PATH}"
