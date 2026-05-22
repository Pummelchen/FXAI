# FXAI Legacy Folder

This folder no longer owns FXAI framework code or handbook documentation.

The old in-repo handbook has been removed. Active documentation and
source now live with the owning root-level projects:

- `../README.md`: top-level FXAI project overview.
- `../FXDataEngine/`: Swift data-engine package and former FXAI toolchain under
  `../FXDataEngine/Tools/`.
- `../FXPlugins/`: Swift-era plugin package and plugin conversion plan.
- `../FXBacktest/`: Swift/Metal offline backtest framework.
- `../FXDatabase/`: data source and remaining MT5 exporter bridge.
- `../FXDataEngineGUI/`: SwiftUI operator GUI and GUI documentation.

Any remaining untracked files in this folder are local compatibility artifacts
from the retired MQL5-era layout and are not the source of truth for new FXAI
work.
