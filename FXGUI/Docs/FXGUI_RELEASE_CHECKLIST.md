# FXGUI Release Checklist

Use this checklist when packaging the FXGUI for operator use.

## Verification

- `swift test`
- `swift build`
- `python3 ../FXDataEngine/Tools/fxai_testlab.py verify-all`
- confirm the latest GUI state renders:
  - saved views
  - onboarding
  - incident center
  - runtime monitor
  - Research OS control
  - advanced visuals

## Packaging

```bash
cd /path/to/FXAI/FXGUI
./Tools/package_gui_release.sh
```

Outputs:
- `FXGUI/Artifacts/Release/FXGUI.app`
- `FXGUI/Artifacts/Release/FXGUI-macos.zip` by default, or the archive name configured in the repo-root toolchain config

## Operator Checks

- app launches cleanly on the configured minimum macOS target
- project-root switching works
- command previews still match the terminal workflows
- incident center opens generated recovery commands
- saved views survive app relaunch

## Release Notes

Document:
- implemented GUI phases
- verification results
- changed operator workflows
- known limitations, if any
