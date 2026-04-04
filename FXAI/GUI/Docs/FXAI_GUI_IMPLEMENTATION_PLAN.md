# FXAI GUI Implementation Plan

## Objective

Build a useful, efficient, practical macOS 26 GUI on top of FXAI using Swift, SwiftUI, Swift Charts, AppKit, and Metal where it provides a real advantage.

The terminal remains first-class. The GUI is an optional operator layer that:
- reduces time-to-understanding
- makes role-specific workflows obvious
- surfaces reports, artifacts, plugin choices, and runtime state clearly
- maps every important action back to a concrete FXAI command or artifact

## Design Principles

- Terminal-first, GUI-assisted
- Read-only by default for live and promotion-sensitive areas
- Role-oriented defaults with expert drill-down
- Real artifact awareness over placeholder-only UX
- Minimalistic dark visual language with high information density
- Strong information hierarchy, low chrome
- Zero duplication of framework logic where file-backed or command-backed integration is enough

## User Matrix

| User | Main Goal | Primary GUI Value | Default Workspace |
|---|---|---|---|
| Live Trader | Observe and trust current live state | profile clarity, artifact health, runtime status, fast interpretation | Live Overview |
| Demo Trader | Observe behavior safely | compare runtime behavior vs audit expectation | Demo Overview |
| Backtester | Launch focused evaluations | quick run setup, scenario awareness, result comparison | Backtest Builder |
| EA Researcher | Improve models and promote better configs | plugin zoo, report browsing, offline lab workflows, lineage | Research Workspace |
| System Architect | Operate the research OS safely | governance, Turso health, recovery, operator dashboard | Platform Control |

## Information Architecture

### Shared Shell
- Home / Overview
- Role Workspaces
- Plugin Zoo
- Reports Explorer
- Command Center
- Settings

### Advanced Areas
- Audit Lab
- Backtest Builder
- Offline Lab
- Promotion Center
- Runtime Monitor
- Turso / Research OS Admin
- Recovery / Incident Tools

## Data Sources

The GUI should be built around FXAI’s actual artifact surfaces:

- project tree: `FXAI.mq5`, `Tests/`, `Plugins/`, `Tools/`
- plugin family directories under `Plugins/`
- baselines under `Tools/Baselines/`
- Offline Lab profiles, ResearchOS, distillation, and bundle artifacts
- compiled outputs such as `FXAI.ex5`, `FXAI_AuditRunner.ex5`, `FXAI_OfflineExportRunner.ex5`
- Turso-backed research state indirectly through emitted files and commands
- `FILE_COMMON` runtime artifacts in later phases

## UX Rules

- Show exact command for all actions that matter
- Let users copy commands instead of retyping them
- Never hide artifact paths
- Separate `live`, `demo`, `research`, and `production` contexts clearly
- Basic, Advanced, and Expert levels for complex forms
- Searchable, filterable tables wherever model or report inventory grows

## Technical Architecture

### Package Layout

`FXAI/GUI`
- `Package.swift`
- `Docs/`
- `Sources/FXAIGUICore/`
- `Sources/FXAIGUIApp/`
- `Tests/FXAIGUICoreTests/`

### Core Layer

`FXAIGUICore`
- immutable models
- project discovery
- project scanning
- report and plugin inventory building
- command recipe generation
- theme and reusable visual surfaces

### App Layer

`FXAIGUIApp`
- app entrypoint
- app state
- navigation shell
- feature views
- AppKit window configuration
- copy/open interactions

## Phase Plan

### Phase 0: Foundations
- define information architecture
- define role matrix
- define command-to-workflow mapping
- define package structure
- define dark theme and reusable surfaces
- define data adapters for current artifact surfaces

Exit criteria:
- implementation reference stored
- package boundary fixed
- no open architecture ambiguity

### Phase 1: Read-Only Operator Dashboard

Status: Implemented

Scope:
- project root discovery
- sidebar navigation
- dark dashboard
- role workspace directory
- plugin-zoo explorer
- reports explorer
- command center
- settings and path switching
- AppKit window polish
- Swift Charts overview cards

Primary data sources:
- `Plugins/`
- `Tools/Baselines/`
- `Tools/OfflineLab/ResearchOS/`
- `Tools/OfflineLab/Profiles/`
- compiled MT5 targets

Exit criteria:
- GUI builds cleanly
- surfaces real FXAI state
- polished enough to use daily for inspection

### Phase 2: Run Builders

Status: Implemented

Scope:
- Audit Lab runner UI
- Backtest builder
- Offline Lab campaign builder
- preset management
- generated command preview and execution handoff

Exit criteria:
- users can configure common runs without reading raw docs
- every form maps cleanly to terminal commands

Implemented in the current package as:
- `Features/AuditLab/AuditLabBuilderView.swift`
- `Features/Backtest/BacktestBuilderView.swift`
- `Features/OfflineLab/OfflineLabBuilderView.swift`
- `FXAIGUICore/Services/RunBuilderCommandFactory.swift`

### Phase 3: Runtime And Promotion Operations

Status: Implemented

Scope:
- live runtime monitor
- deployment profile viewer
- student router viewer
- supervisor-service viewer
- promotion center
- lineage and minimal-bundle viewer

Exit criteria:
- live traders and operators can inspect actual deployed state without reading TSV manually

Implemented in the current package as:
- `Features/Runtime/RuntimeMonitorView.swift`
- `Features/Promotion/PromotionCenterView.swift`
- `FXAIGUICore/Services/RuntimeArtifactReader.swift`
- `FXAIGUICore/Models/RuntimeArtifactModels.swift`

### Phase 4: Turso / Research OS Control

Status: Implemented

Scope:
- Turso environment diagnostics
- branch workflows
- PITR restore flows
- audit-log ingestion state
- vector-backed analog retrieval browser
- recovery tools

Exit criteria:
- system architect can operate the research OS from one coherent surface

Implemented in the current package as:
- `Features/ResearchOS/ResearchOSControlView.swift`
- `FXAIGUICore/Services/ResearchOSArtifactReader.swift`
- `FXAIGUICore/Services/ResearchOSCommandFactory.swift`
- `FXAIGUICore/Models/ResearchOSControlModels.swift`

### Phase 5: Advanced Visualization And Metal Surfaces

Scope:
- Metal-backed dense heatmaps where SwiftUI/Charts are not sufficient
- world-plan visualizer
- plugin-family stress maps
- promotion and attribution timelines
- large artifact diff surfaces

Use Metal only where:
- Charts become too limited
- very dense multi-series surfaces need smooth interaction
- large heatmaps or tensor-like views need better rendering performance

Exit criteria:
- advanced visualization is materially better than plain SwiftUI alternatives

### Phase 6: Production-Grade UX Completion

Scope:
- saved views
- role-specific onboarding
- incident workflows
- recovery wizards
- keyboard-first navigation
- final documentation and release packaging

Exit criteria:
- the GUI is practical for daily operator use
- terminal users still feel respected, not replaced

## Screen Plan

### Overview
- global health ribbon
- build targets card
- plugin zoo distribution chart
- report/artifact footprint chart
- recent research outputs
- role quick launch cards

### Role Workspaces
- one card set per role
- what this role should use
- what this role should ignore
- recommended first command
- recommended first artifact

### Plugin Zoo
- search
- family filters
- plugin list
- split vs monolithic status
- quick family counts
- source-path open actions

### Reports Explorer
- category filters
- artifact list
- last modified timestamps
- file size
- quick open in Finder

### Command Center
- terminal-first recipes
- copy command
- grouped by role and task
- “what this does” explanations

### Settings
- project root selector
- detected root validation
- active docs and artifact paths
- environment summary

## Visual Direction

- Dark-only default
- Ink, graphite, slate, teal, and electric-blue palette
- Glass-like panel surfaces with restrained highlights
- Rounded rectangles, subtle separators, no noisy outlines
- High-density layout with generous whitespace between logical groups
- Charts with low-noise labeling and bold focus metrics

## Risks And Controls

### Risk
- GUI becomes a parallel control plane with hidden behavior

Control:
- keep command preview visible
- treat terminal as source of action truth

### Risk
- GUI drifts from actual artifact shapes

Control:
- scan current FXAI outputs directly
- add tests for project scanner behavior

### Risk
- visual polish hurts clarity

Control:
- prioritize legibility, hierarchy, and role-based defaults over decoration

## Definition Of Done

The GUI work is successful when:
- users can understand the current FXAI state in under one minute
- common workflows are obvious by role
- the GUI points at real FXAI artifacts, not only samples
- the package builds cleanly on macOS 26
- the codebase is modular enough to grow phase-by-phase without rewrites
