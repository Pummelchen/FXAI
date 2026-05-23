#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${1:-${GUI_ROOT}/Artifacts/GUISnapshots}"

mkdir -p "${OUTPUT_DIR}"

cd "${GUI_ROOT}"
echo "[fxgui] running full Swift test suite"
swift test

echo "[fxgui] exporting operator screenshots to ${OUTPUT_DIR}"
FXGUI_WRITE_SNAPSHOTS=1 \
FXGUI_SNAPSHOT_DIR="${OUTPUT_DIR}" \
swift test --filter GUIOperatorSnapshotTests

echo "[fxgui] validation complete"
