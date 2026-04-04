# FXAI GUI

FXAI GUI is an optional macOS 26 operator surface for the FXAI framework.

The terminal remains first-class. The GUI exists to make the framework easier to inspect, faster to operate, and less error-prone for humans. Every major GUI workflow is designed to map back to an explicit FXAI command, artifact, or report.

## Current Scope

Phase 1 is implemented here:
- polished dark macOS dashboard
- role-based workspaces
- plugin-zoo explorer
- report explorer over current FXAI artifact surfaces
- command center with terminal-first command previews
- settings and project-root detection

The phased implementation reference is stored in:
- [Docs/FXAI_GUI_IMPLEMENTATION_PLAN.md](/Users/andreborchert/FXAI-main2/FXAI/GUI/Docs/FXAI_GUI_IMPLEMENTATION_PLAN.md)

## Build

```bash
cd /Users/andreborchert/FXAI-main2/FXAI/GUI
swift build
swift run FXAIGUI
```

## Design Principles

- terminal-first, GUI-assisted
- read-only by default for sensitive runtime workflows
- role-oriented information architecture
- minimalistic dark visual language
- real FXAI artifact awareness instead of placeholder-only UI
- phased delivery so the GUI can grow without being rewritten
