#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${1:-${GUI_ROOT}/Artifacts/GUISnapshots}"

mkdir -p "${OUTPUT_DIR}"

cd "${GUI_ROOT}"
echo "[fxai-gui] running full Swift test suite"
swift test

echo "[fxai-gui] exporting operator screenshots to ${OUTPUT_DIR}"
FXAI_GUI_WRITE_SNAPSHOTS=1 \
FXAI_GUI_SNAPSHOT_DIR="${OUTPUT_DIR}" \
swift test --filter GUIOperatorSnapshotTests

echo "[fxai-gui] validation complete"
