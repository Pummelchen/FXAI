#!/usr/bin/env zsh
set -euo pipefail

cd "$(dirname "$0")"

swift test
swift build
exec swift run FXAIGUI
