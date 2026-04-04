# FXAI GUI Release Checklist

Use this checklist when packaging the macOS 26 FXAI GUI for operator use.

## Verification

- `swift test`
- `swift build`
- `python3 ../Tools/fxai_testlab.py verify-all`
- confirm the latest GUI state renders:
  - saved views
  - onboarding
  - incident center
  - runtime monitor
  - Research OS control
  - advanced visuals

## Packaging

```bash
cd /Users/andreborchert/FXAI-main2/FXAI/GUI
./Tools/package_gui_release.sh
```

Outputs:
- `FXAI/GUI/Artifacts/Release/FXAIGUI.app`
- `FXAI/GUI/Artifacts/Release/FXAIGUI-macos26.zip`

## Operator Checks

- app launches cleanly on macOS 26
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
