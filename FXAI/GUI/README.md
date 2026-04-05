# FXAI GUI

FXAI GUI is an optional macOS 26 operator surface for the FXAI framework.

The terminal remains first-class. The GUI exists to make the framework easier to inspect, faster to operate, and less error-prone for humans. Every major GUI workflow is designed to map back to an explicit FXAI command, artifact, or report.

The primary app surface is the FXAI operator shell, not the SVG-derived theme showcase. The `GUI.svg` work remains in the codebase as the first production theme and calibration reference, but the shipped app opens into the FXAI workspaces, builders, reports, runtime monitor, and Research OS views.

## Current Scope

Phase 1 through Phase 6 are implemented here:
- polished dark macOS dashboard
- role-based workspaces
- plugin-zoo explorer
- report explorer over current FXAI artifact surfaces
- command center with terminal-first command previews
- settings and project-root detection
- Audit Lab run builder
- backtest prep builder
- Offline Lab workflow builder
- runtime monitor for deployed symbol state
- promotion center for champions, tiers, and set paths
- Research OS control for Turso environment diagnostics, branch and PITR flows, audit-log visibility, analog vector browsing, and recovery command generation
- advanced visualization surfaces with Metal-backed heatmaps, world-plan charts, family stress maps, artifact diff surfaces, and promotion/attribution timelines
- persistent saved views for repeatable operator workflows
- role-specific onboarding and keyboard-first navigation
- incident detection with generated recovery playbooks
- detached startup with explicit connect/disconnect handling and soft auto reconnect every 10 seconds
- release packaging support for a polished macOS app bundle

## Theme Support

The app also ships a production finance theme under:

- `FinancialDashboardThemeV1`

This is an SVG-driven theme and rendering system built from the canonical `GUI.svg` asset and exposed through:

- `Sources/FXAIGUICore/Theme/`
- `Sources/FXAIGUICore/Layout/`
- `Sources/FXAIGUICore/SVG/`
- `Sources/FXAIGUICore/Rendering/`
- `Sources/FXAIGUIApp/Components/`
- `Sources/FXAIGUIApp/Debug/`

Theme V1 includes:
- reusable theme tokens for colors, gradients, glows, shadow stacks, typography, radii, materials, layout metrics, and chart style
- semantic component styles, render-tier policy, and a shared theme registry/environment
- an adaptive dashboard layout engine with compact, standard, wide, and ultra-wide classes
- a custom finance dashboard renderer using SwiftUI plus AppKit/WebKit and Metal where useful
- debug calibration mode with SVG overlay, PNG compare mode, layout guides, frame outlines, live scale display, and effect toggles
- a detached-safe startup path so the GUI can launch even before FXAI or MT5 is fully ready

The phased implementation reference is stored in:
- [Docs/FXAI_GUI_IMPLEMENTATION_PLAN.md](/Users/andreborchert/FXAI-main2/FXAI/GUI/Docs/FXAI_GUI_IMPLEMENTATION_PLAN.md)
- [Docs/FXAI_GUI_RELEASE_CHECKLIST.md](/Users/andreborchert/FXAI-main2/FXAI/GUI/Docs/FXAI_GUI_RELEASE_CHECKLIST.md)
- [Docs/FINANCE_APP_THEME_MIGRATION.md](/Users/andreborchert/FXAI-main2/FXAI/GUI/Docs/FINANCE_APP_THEME_MIGRATION.md)
- [Docs/ADDING_THEME_V2.md](/Users/andreborchert/FXAI-main2/FXAI/GUI/Docs/ADDING_THEME_V2.md)

## Build

```bash
cd /Users/andreborchert/FXAI-main2/FXAI/GUI
swift test
swift build
swift run FXAIGUI
./Tools/package_gui_release.sh
```

The canonical reference assets are bundled from:

- `Sources/FXAIGUICore/Resources/Reference/GUI.svg`
- `Sources/FXAIGUICore/Resources/Reference/GUI-reference.png`

## Connection Behavior

The GUI can start cleanly even when MT5 or the FXAI project tree is not ready yet.

- if a valid FXAI project root is available, it connects automatically
- if not, it starts in a detached shell mode without failing
- it can softly retry connection every 10 seconds when auto reconnect is enabled
- operators can explicitly disconnect and reconnect from the toolbar or settings

## Design Principles

- terminal-first, GUI-assisted
- read-only by default for sensitive runtime workflows
- role-oriented information architecture
- minimalistic dark visual language
- SVG is the source of truth for Theme V1 geometry and styling
- adaptive, not fixed-canvas
- premium shadow/glow/material rendering over generic stock controls
- real FXAI artifact awareness instead of placeholder-only UI
- phased delivery so the GUI can grow without being rewritten
