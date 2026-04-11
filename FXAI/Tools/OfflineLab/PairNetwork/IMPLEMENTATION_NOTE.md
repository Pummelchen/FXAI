# Pair Network Implementation Note

## Repo Mapping

- Offline Lab contracts, config, math, graph construction, and resolver logic live in:
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/pair_network_contracts.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/pair_network_config.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/pair_network_math.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/pair_network.py`
- CLI entry points are wired through:
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/cli_parser.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/cli_commands.py`
- Runtime integration is in:
  - `/Users/andreborchert/FXAI-main2/FXAI/Engine/Runtime/Trade/runtime_trade_pair_network.mqh`
  - `/Users/andreborchert/FXAI-main2/FXAI/Engine/Runtime/Trade/runtime_trade_risk.mqh`
  - `/Users/andreborchert/FXAI-main2/FXAI/Engine/Runtime/runtime_trade_helpers.mqh`
  - `/Users/andreborchert/FXAI-main2/FXAI/FXAI.mq5`
- GUI ingestion and display are in:
  - `/Users/andreborchert/FXAI-main2/FXAI/GUI/Sources/FXAIGUICore/Models/PairNetworkModels.swift`
  - `/Users/andreborchert/FXAI-main2/FXAI/GUI/Sources/FXAIGUICore/Services/PairNetworkArtifactReader.swift`
  - `/Users/andreborchert/FXAI-main2/FXAI/GUI/Sources/FXAIGUIApp/Features/PairNetwork/PairNetworkView.swift`

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

- The runtime already had exposure and pressure controls in `runtime_trade_risk.mqh`, so the pair network is integrated there instead of creating a separate order engine.
- Same-cycle candidate coordination reuses the existing control-plane snapshot files instead of introducing a new candidate bus.
- Phase 1 uses fixed currency-factor profiles plus optional empirical dependency augmentation; it does not attempt full covariance optimization or graph learning.

## Verification Path

- `python3 Tools/fxai_offline_lab.py pair-network-validate`
- `python3 Tools/fxai_offline_lab.py pair-network-build --profile continuous`
- `python3 Tools/fxai_offline_lab.py pair-network-report --profile continuous`
- `PYTHONPATH=Tools python3 -m pytest Tools/tests/test_pair_network.py Tools/tests/test_dashboard.py -q`
- `cd GUI && swift test && swift build`
- `python3 Tools/fxai_testlab.py compile-main`
