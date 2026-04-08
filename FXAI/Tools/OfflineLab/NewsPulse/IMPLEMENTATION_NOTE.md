# NewsPulse Implementation Note

This note maps the NewsPulse design onto the actual FXAI repository structure.

## Real Integration Points

- MT5 scheduled macro collector:
  - `/Users/andreborchert/FXAI-main2/FXAI/Services/FXAI_NewsPulseCalendar.mq5`
  - Runs as an MT5 Service, not a chart EA.
  - Exports calendar feed and state into `FILE_COMMON/FXAI/Runtime/`.

- Python collectors, fusion, policy, and daemon:
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/newspulse_contracts.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/newspulse_config.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/newspulse_policy.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/newspulse_calendar.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/newspulse_gdelt.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/newspulse_official.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/newspulse_story.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/newspulse_replay.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/newspulse_fusion.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/newspulse_daemon.py`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tools/offline_lab/newspulse_service.py`

- Runtime gating adapter:
  - `/Users/andreborchert/FXAI-main2/FXAI/Engine/Runtime/Trade/runtime_trade_newspulse.mqh`
  - `/Users/andreborchert/FXAI-main2/FXAI/Engine/Runtime/Trade/runtime_trade_risk.mqh`
  - Consumes flat, file-backed pair state without changing the canonical model vector.

- Audit and replay seam:
  - `/Users/andreborchert/FXAI-main2/FXAI/Tests/Scenarios/audit_newspulse_replay.mqh`
  - `/Users/andreborchert/FXAI-main2/FXAI/Tests/Scenarios/audit_context_series.mqh`
  - Pulls replay timelines into the existing macro-event audit scenario instead of inventing a parallel audit flow.

- GUI operator surface:
  - `/Users/andreborchert/FXAI-main2/FXAI/GUI/Sources/FXAIGUICore/Models/NewsPulseModels.swift`
  - `/Users/andreborchert/FXAI-main2/FXAI/GUI/Sources/FXAIGUICore/Services/NewsPulseArtifactReader.swift`
  - `/Users/andreborchert/FXAI-main2/FXAI/GUI/Sources/FXAIGUIApp/Features/NewsPulse/NewsPulseView.swift`
  - `/Users/andreborchert/FXAI-main2/FXAI/GUI/Sources/FXAIGUICore/Services/IncidentBuilder.swift`

## Why This Structure

NewsPulse is shared execution and observability infrastructure. It does not belong in:
- the plugin zoo
- the canonical feature-vector contract
- a per-symbol collector path

The actual repo already has:
- `Services/` for MT5 background work
- `Tools/offline_lab/` for machine-local daemons and shared research/runtime artifacts
- `Engine/Runtime/Trade/` for execution overlays
- `GUI/` for operator surfaces

So NewsPulse was added at those seams instead of inventing new top-level systems.

## Phase-1 Safe Decisions

- Scheduled events use MT5 Economic Calendar only.
- Breaking news uses GDELT plus a narrow optional official-feed rail.
- Pair state is derived from currency state.
- Runtime integration is gating-first, not model-retraining-first.
- Missing or stale state is treated as unknown.

## New Phase-2 Style Upgrades Landed Here

1. Official-feed rail:
   - `newspulse_official.py`

2. Replay-native history and rebuild path:
   - `newspulse_replay.py`
   - `audit_newspulse_replay.mqh`

3. Symbol and session-aware calibration:
   - `newspulse_policy.py`
   - runtime consumption in `runtime_trade_newspulse.mqh`

4. Evolving story clustering:
   - `newspulse_story.py`

5. Daemon supervision and health:
   - `newspulse_daemon.py`
   - GUI and incident integration via the NewsPulse status payload

6. Operator-editable watchlists and broker mapping:
   - `Tools/OfflineLab/NewsPulse/newspulse_policy.json`

7. GUI drill-down and timelines:
   - `NewsPulseView.swift`
   - `NewsPulseArtifactReader.swift`

## Deliberate Deviations From The Original Plan

- The first operator-editable watchlist and broker mapping path is file-driven through `newspulse_policy.json`, not a GUI editor, because FXAI already relies on versioned operator artifacts and terminal-first workflows.
- Replay is integrated into the existing macro-event audit scenario instead of adding a completely separate audit scenario family.
- The GUI reads `newspulse_status.json`, which is the richer local operator surface, while the runtime still consumes the flat runtime artifact for safety and simplicity.
