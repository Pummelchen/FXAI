# FXAI Configuration Semantics

This file is the tracked contract for FXAI toolchain configuration used by the
Python OfflineLab/test tooling and the Swift GUI project scanner.

## Files

FXAI supports two local configuration files:

- `.env`: simple `KEY=value` overrides.
- `fxai.toml`: structured toolchain, path, profile, and GUI release settings.

`FXAI_ENV_FILE` can point to a specific `.env` file. `FXAI_CONFIG` can point to
a specific TOML file and may be provided either by the real process environment
or by `.env`.

## Precedence

Configuration precedence is:

1. Locate and load `.env`.
2. Merge real process environment over `.env`.
3. Resolve `FXAI_CONFIG` from the merged environment.
4. Load `fxai.toml`.
5. Resolve `FXAI_TOOLCHAIN_PROFILE`: real/merged environment wins over TOML;
   `auto` or an empty value means auto-detect.
6. Merge TOML path sections: `[profiles.<profile>.paths]` overrides `[paths]`.
7. Resolve individual path keys: environment variables override TOML, TOML
   overrides profile defaults.

Whitespace-only environment values are ignored. Profile tokens are trimmed and
lowercased before comparison.

## Path Bases

TOML path values are project-root-relative by default. The GUI keeps a
config-directory fallback only for older local configs that already used
config-file-relative paths, but project-root-relative paths win when both exist.

Environment path values accept absolute paths, `~`, `$VAR`, and `${VAR}`.
Relative environment path values are resolved relative to the active project
root.

## Supported Keys

Toolchain:

- `FXAI_TOOLCHAIN_PROFILE`
- `[toolchain].profile`

Path keys:

- `FXAI_PROJECT_ROOT` / `project_root`
- `FXAI_REPO_ROOT` / `repo_root`
- `FXAI_MT5_ROOT` / `mt5_root`
- `FXAI_METAEDITOR` / `metaeditor`
- `FXAI_TERMINAL` / `terminal`
- `FXAI_WINE` / `wine`
- `FXAI_COMMON_FILES` / `common_files`
- `FXAI_RUNTIME_DIR` / `runtime_dir`
- `FXAI_TESTER_PRESET_DIR` / `tester_preset_dir`
- `FXAI_COMMON_INI` / `common_ini`
- `FXAI_TERMINAL_INI` / `terminal_ini`
- `FXAI_MT5_LOG_DIR` / `mt5_log_dir`
- `FXAI_DEFAULT_DB` / `default_db`

GUI release keys:

- `FXGUI_MINIMUM_MACOS` / `[gui].minimum_macos`
- `FXGUI_RELEASE_ARCHIVE` / `[gui].release_archive`

## Test Gates

The executable contract is covered by:

- `FXDataEngine/Tools/tests/test_toolchain_config.py`
- `FXGUI/Tests/FXGUICoreTests/FXAIProjectConfigurationResolverTests.swift`
- `FXGUI/Tests/FXGUICoreTests/RuntimeArtifactPathResolverTests.swift`
- `FXGUI/Tests/FXGUICoreTests/ProjectScannerTests.swift`
