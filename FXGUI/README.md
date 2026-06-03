# FXGUI

FXGUI is an optional macOS operator surface for the FXAI framework.

The terminal remains first-class. The GUI exists to make the framework easier to inspect, faster to operate, and less error-prone for humans. Every major GUI workflow is designed to map back to an explicit FXAI command, artifact, or report.

The primary app surface is the FXAI operator shell. The shipped app opens into the FXAI workspaces, builders, reports, runtime monitor, and Research OS views.

## Current Scope

Phase 1 through Phase 6 are implemented here:
- polished dark macOS dashboard
- first-class role workspaces for Live Overview, Demo Overview, Research Workspace, and Platform Control
- role-based workspaces
- plugin-zoo explorer
- report explorer over current FXAI artifact surfaces
- command center with terminal-first command previews
- settings and project-root detection
- Audit Lab run builder
- backtest prep builder
- Offline Lab workflow builder
- runtime monitor for deployed symbol state
- NewsPulse surface for source health, currency heatmap, pair risk, and recent event tape
- Rates Engine surface for provider health, currency policy state, pair divergence, rates-aware gates, and policy-event tape
- Microstructure surface for per-symbol execution regime, liquidity stress, hostile execution, stop-run proxy flags, session handoff state, and runtime gating reasons
- Adaptive Router surface for live regime state, plugin routing weights, suppressed plugins, posture reasons, and replay transitions
- Dynamic Ensemble surface for post-inference plugin trust, participation weights, suppression state, and routed final action context
- Probabilistic Calibration surface for calibrated probabilities, expected move quantiles, edge-after-costs, selected tier support, and abstention reasons
- Execution Quality surface for expected price-cost, slippage stress, fill quality, latency sensitivity, liquidity fragility, replay transitions, and current execution-state reasons
- promotion center for champions, tiers, and set paths
- Research OS control for Turso environment diagnostics, branch and PITR flows, audit-log visibility, analog vector browsing, and recovery command generation
- advanced visualization surfaces with Metal-backed heatmaps, world-plan charts, family stress maps, artifact diff surfaces, and promotion/attribution timelines
- persistent saved views for repeatable operator workflows
- role-specific onboarding and keyboard-first navigation
- incident detection with generated recovery playbooks
- detached startup with explicit connect/disconnect handling and soft auto reconnect every 10 seconds
- customizable overview dashboard with draggable categories, movable widgets, 1 cm grid-based resizing, automatic layout persistence, and reset-to-default controls
- customizable role workspaces with draggable and resizable panels, automatic layout persistence per dashboard, and per-workspace reset controls
- automatic resource guards that back off glass, Metal, blur, and background polling under memory, thermal, or inactive-app pressure
- release packaging support for a polished macOS app bundle with configurable minimum-version metadata

## Command Security

FXGUI remains terminal-first, but command cards are not arbitrary shell launchers.
Generated command previews pass through `FXAICommandSecurityPolicy` before they
can be copied as commands or handed off to Terminal. The policy requires a
project-local `cd` context, rejects shell chaining, redirection, expansion,
backslash escapes, unsupported control characters, and non-approved executables,
and allows only curated FXAI Swift package checks, approved FXAI Python tool
entry points, the root `./fxai certify --all` command, BacktestAgent self-check,
and the GUI release packager. Rejected commands are not copied to the pasteboard
and are surfaced through the app error message path.

## Theme Support

The app also ships a shared operator theme under:

- `FinancialDashboardThemeV1`

This is a reference-driven theme and rendering system exposed through:

- `Sources/FXGUICore/Theme/`
- `Sources/FXGUICore/Layout/`
- `Sources/FXGUICore/SVG/`
- `Sources/FXGUICore/Rendering/`
- `Sources/FXGUIApp/Components/`
- `Sources/FXGUIApp/Debug/`

The theme system includes:
- reusable theme tokens for colors, gradients, glows, shadow stacks, typography, radii, materials, layout metrics, and chart style
- semantic component styles, render-tier policy, and a shared theme registry/environment
- an adaptive dashboard layout engine with compact, standard, wide, and ultra-wide classes
- a custom operator-theme renderer stack using SwiftUI plus AppKit/WebKit and Metal where useful
- debug calibration mode with reference overlay, compare mode, layout guides, frame outlines, live scale display, and effect toggles
- a detached-safe startup path so the GUI can launch even before FXAI or MT5 is fully ready

Operational GUI references are stored in:
- [Docs/FXGUI_RELEASE_CHECKLIST.md](Docs/FXGUI_RELEASE_CHECKLIST.md)
- [Docs/FINANCE_APP_THEME_MIGRATION.md](Docs/FINANCE_APP_THEME_MIGRATION.md)
- [Docs/ADDING_THEME_V2.md](Docs/ADDING_THEME_V2.md)

## Build

```bash
cd /path/to/FXAI/FXGUI
./start.sh
./Tools/package_gui_release.sh
```

The GUI release packager reads the repo-root toolchain config, falling back to `../FXDataEngine/fxai.toml` when no root `fxai.toml` exists. It respects `FXGUI_MINIMUM_MACOS` or `FXGUI_RELEASE_ARCHIVE` overrides when you need a different packaging target.

Toolchain and GUI config precedence is defined in the root
[`CONFIGURATION_SEMANTICS.md`](../CONFIGURATION_SEMANTICS.md). The short version:
real process environment overrides `.env`, `.env` may select `FXAI_CONFIG`, an
environment `FXAI_TOOLCHAIN_PROFILE` overrides TOML, profile-specific TOML paths
override global TOML paths, and explicit path environment variables override all
TOML paths.

## GUI Validation

The GUI includes a dedicated validation suite for layout quality, resize behavior, and operator-shell rendering fidelity across multiple desktop sizes.

- `swift test` runs the core layout and GUI snapshot tests, including the `FXGUIAppTests` target
- `./Tools/run_gui_validation_suite.sh` reruns the full GUI test suite and exports real screenshots for the operator shell
- validation scenarios cover compact desktop, MacBook 14, standard desktop, wide desktop, 4K-like ultra-wide, and 8K-like ultra-wide sizes
- the exported screenshots are written to `FXGUI/Artifacts/GUISnapshots/` and are ignored by git

The validation suite is intended to catch:
- spacing regressions
- typography clamp problems
- poor adaptive reflow decisions
- unreadable chart/card placement at narrow widths
- dead or visually broken operator surfaces at very large widths

The internal reference assets are bundled from:

- `Sources/FXGUICore/Resources/Reference/FXAI-theme-reference.svg`
- `Sources/FXGUICore/Resources/Reference/FXAI-theme-reference.png`

## Connection Behavior

The GUI can start cleanly even when MT5 or the FXAI project tree is not ready yet.

- if a valid FXAI project root is available, it connects automatically
- if not, it starts in a detached shell mode without failing
- it can softly retry connection every 10 seconds in the normal case, and automatically backs off the retry cadence when the app is inactive or under resource pressure
- operators can explicitly disconnect and reconnect from the toolbar or settings

## Design Principles

- terminal-first, GUI-assisted
- read-only by default for sensitive runtime workflows
- role workspaces are real operator entry points, not just onboarding labels
- role-oriented information architecture
- minimalistic dark visual language
- adaptive, not fixed-canvas
- premium shadow/glow/material rendering over generic stock controls
- real FXAI artifact awareness instead of placeholder-only UI
- phased delivery so the GUI can grow without being rewritten
