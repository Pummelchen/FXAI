# Pair Network Implementation Note

## Repo Mapping

- Offline Lab contracts, config, math, graph construction, and resolver logic live in:
  - `<FXAI_ROOT>/Tools/OfflineLab/offline_lab/pair_network_contracts.py`
  - `<FXAI_ROOT>/Tools/OfflineLab/offline_lab/pair_network_config.py`
  - `<FXAI_ROOT>/Tools/OfflineLab/offline_lab/pair_network_math.py`
  - `<FXAI_ROOT>/Tools/OfflineLab/offline_lab/pair_network.py`
- CLI entry points are wired through:
  - `<FXAI_ROOT>/Tools/OfflineLab/offline_lab/cli_parser.py`
  - `<FXAI_ROOT>/Tools/OfflineLab/offline_lab/cli_commands.py`
- Runtime integration is in:
- GUI ingestion and display are in:
  - `<FXAI_ROOT>/FXGUI/Sources/FXGUICore/Models/PairNetworkModels.swift`
  - `<FXAI_ROOT>/FXGUI/Sources/FXGUICore/Services/PairNetworkArtifactReader.swift`
  - `<FXAI_ROOT>/FXGUI/Sources/FXGUIApp/Features/PairNetwork/PairNetworkView.swift`

## Phase-1 Scope

- Builds a structural currency and factor graph for the active FX universe.
- Adds empirical pair dependence only when dataset overlap support is high enough.
- Resolves candidate trades against:
  - live positions
  - pending orders
  - same-cycle control-plane peer candidates
- Emits deterministic runtime decisions and append-only history per symbol.
- Publishes Offline Lab status and graph reports plus a GUI operator view.

## Repo-Specific Deviations

- Same-cycle candidate coordination reuses the existing control-plane snapshot files instead of introducing a new candidate bus.
- Phase 1 uses fixed currency-factor profiles plus optional empirical dependency augmentation; it does not attempt full covariance optimization or graph learning.

## Verification Path

- `python3 Tools/fxai_offline_lab.py pair-network-validate`
- `python3 Tools/fxai_offline_lab.py pair-network-build --profile continuous`
- `python3 Tools/fxai_offline_lab.py pair-network-report --profile continuous`
- `PYTHONPATH=Tools python3 -m pytest Tools/tests/test_pair_network.py Tools/tests/test_dashboard.py -q`
- `cd FXGUI && swift test && swift build`
- `python3 Tools/fxai_testlab.py compile-main`
