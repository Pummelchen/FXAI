from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_TOOLCHAIN_PROFILES = ("macos_wine", "windows_native", "headless_ci")
AUTO_PROFILE = "auto"
DEFAULT_WINE_BINARY = Path("/Applications/MetaTrader 5.app/Contents/SharedSupport/wine/bin/wine64")


@dataclass(frozen=True)
class FXAIToolchainConfig:
    profile: str
    config_path: Path | None
    env_path: Path | None
    project_root: Path
    repo_root: Path
    mt5_root: Path
    metaeditor: Path
    terminal: Path
    wine: Path | None
    common_files: Path
    runtime_dir: Path
    tester_preset_dir: Path
    common_ini: Path
    terminal_ini: Path
    mt5_log_dir: Path
    default_db: Path
    gui_minimum_macos: str
    gui_release_archive: str
    path_sources: dict[str, str]

    @property
    def uses_wine(self) -> bool:
        return self.profile == "macos_wine"

    def path_argument(self, path: Path) -> str:
        value = path.expanduser()
        if self.uses_wine:
            return "Z:\\" + str(value).replace("/", "\\").lstrip("\\")
        return str(value)

    def metaeditor_compile_command(self, source_path: Path, log_path: Path) -> list[str]:
        cmd: list[str] = []
        if self.uses_wine:
            if self.wine is None:
                raise RuntimeError("macos_wine profile requires a configured Wine binary")
            cmd.append(str(self.wine))
        cmd.append(str(self.metaeditor))
        cmd.append(f"/compile:{self.path_argument(source_path)}")
        cmd.append(f"/log:{self.path_argument(log_path)}")
        return cmd

    def terminal_launch_command(self, *, config_path: Path | None = None, portable: bool = False) -> list[str]:
        cmd: list[str] = []
        if self.uses_wine:
            if self.wine is None:
                raise RuntimeError("macos_wine profile requires a configured Wine binary")
            cmd.append(str(self.wine))
        cmd.append(str(self.terminal))
        if config_path is not None:
            cmd.append(f"/config:{self.path_argument(config_path)}")
        if portable:
            cmd.append("/portable")
        return cmd

    def describe(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "config_path": (str(self.config_path) if self.config_path else ""),
            "env_path": (str(self.env_path) if self.env_path else ""),
            "project_root": str(self.project_root),
            "repo_root": str(self.repo_root),
            "mt5_root": str(self.mt5_root),
            "metaeditor": str(self.metaeditor),
            "terminal": str(self.terminal),
            "wine": (str(self.wine) if self.wine else ""),
            "common_files": str(self.common_files),
            "runtime_dir": str(self.runtime_dir),
            "tester_preset_dir": str(self.tester_preset_dir),
            "common_ini": str(self.common_ini),
            "terminal_ini": str(self.terminal_ini),
            "mt5_log_dir": str(self.mt5_log_dir),
            "default_db": str(self.default_db),
            "gui_minimum_macos": self.gui_minimum_macos,
            "gui_release_archive": self.gui_release_archive,
            "path_sources": dict(self.path_sources),
        }


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_path(raw: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    expanded = os.path.expandvars(os.path.expanduser(text))
    candidate = Path(expanded)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = (base_dir / candidate).resolve()
    return candidate


def _parse_dotenv(path: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    if not path.exists() or not path.is_file():
        return payload
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        token = key.strip()
        if not token:
            continue
        payload[token] = value.strip().strip("'\"")
    return payload


def _read_toml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or not path.is_file():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _nested_dict(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _detect_profile(project_root: Path, env: Mapping[str, str]) -> str:
    explicit = str(env.get("FXAI_TOOLCHAIN_PROFILE", "") or "").strip().lower()
    if explicit and explicit != AUTO_PROFILE:
        return explicit
    if os.name == "nt":
        return "windows_native"
    if str(env.get("CI", "")).strip() or str(env.get("GITHUB_ACTIONS", "")).strip():
        return "headless_ci"
    normalized_root = str(project_root).replace("\\", "/")
    if "/drive_c/" in normalized_root or DEFAULT_WINE_BINARY.exists():
        return "macos_wine"
    return "headless_ci"


def _drive_c_root(project_root: Path) -> Path | None:
    normalized = str(project_root.resolve()).replace("\\", "/")
    marker = "/drive_c/"
    idx = normalized.find(marker)
    if idx < 0:
        return None
    return Path(normalized[: idx + len(marker)])


def _derive_macos_common_files(project_root: Path, env: Mapping[str, str]) -> Path:
    drive_root = _drive_c_root(project_root)
    if drive_root is None:
        return project_root / "FILE_COMMON"
    users_root = drive_root / "users"
    preferred_user = str(env.get("FXAI_MT5_WINE_USER", "") or "").strip() or Path.home().name
    candidates: list[Path] = []
    if preferred_user:
        candidates.append(users_root / preferred_user)
    if users_root.exists() and users_root.is_dir():
        for directory in sorted(users_root.iterdir()):
            if directory.is_dir() and directory not in candidates:
                candidates.append(directory)
    for user_root in candidates:
        common_files = user_root / "AppData/Roaming/MetaQuotes/Terminal/Common/Files"
        if common_files.exists():
            return common_files
    return users_root / preferred_user / "AppData/Roaming/MetaQuotes/Terminal/Common/Files"


def _derive_windows_common_files(env: Mapping[str, str]) -> Path:
    appdata = str(env.get("APPDATA", "") or "").strip()
    if appdata:
        return Path(appdata) / "MetaQuotes/Terminal/Common/Files"
    return Path.home() / "AppData/Roaming/MetaQuotes/Terminal/Common/Files"


def _resolve_value(
    *,
    name: str,
    config_paths: Mapping[str, Any],
    env: Mapping[str, str],
    base_dir: Path,
    env_key: str,
    default_value: Path | str | None,
    source_map: dict[str, str],
) -> Path | str | None:
    if env_key in env and str(env.get(env_key, "") or "").strip():
        source_map[name] = "environment"
        raw = env[env_key]
    elif name in config_paths and str(config_paths.get(name, "") or "").strip():
        source_map[name] = "fxai.toml"
        raw = str(config_paths[name])
    else:
        source_map[name] = "profile_default"
        raw = default_value
    if isinstance(default_value, Path) or name.endswith("_path") or name in {
        "project_root",
        "repo_root",
        "mt5_root",
        "metaeditor",
        "terminal",
        "wine",
        "common_files",
        "runtime_dir",
        "tester_preset_dir",
        "common_ini",
        "terminal_ini",
        "mt5_log_dir",
        "default_db",
    }:
        return _normalize_path(raw, base_dir=base_dir)
    return (str(raw).strip() if raw is not None else None)


def load_toolchain_config(
    *,
    project_root_hint: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> FXAIToolchainConfig:
    env_map = dict(env or os.environ)
    default_project_root = (project_root_hint or _default_project_root()).resolve()
    config_path = _normalize_path(env_map.get("FXAI_CONFIG"), base_dir=default_project_root)
    if config_path is None:
        config_path = default_project_root / "fxai.toml"
    env_path = _normalize_path(env_map.get("FXAI_ENV_FILE"), base_dir=default_project_root)
    if env_path is None:
        env_path = default_project_root / ".env"
    dotenv_values = _parse_dotenv(env_path)
    merged_env = dict(dotenv_values)
    merged_env.update(env_map)

    raw_config = _read_toml(config_path)
    toolchain_config = _nested_dict(raw_config, "toolchain")
    requested_profile = str(toolchain_config.get("profile", AUTO_PROFILE) or AUTO_PROFILE).strip().lower()
    if not requested_profile or requested_profile == AUTO_PROFILE:
        profile = _detect_profile(default_project_root, merged_env)
    else:
        profile = requested_profile
    if profile not in SUPPORTED_TOOLCHAIN_PROFILES:
        raise ValueError(f"Unsupported FXAI toolchain profile: {profile}")

    profile_block = _nested_dict(_nested_dict(raw_config, "profiles"), profile)
    config_paths = _nested_dict(raw_config, "paths")
    config_paths.update(_nested_dict(profile_block, "paths"))
    gui_config = _nested_dict(raw_config, "gui")
    gui_config.update(_nested_dict(profile_block, "gui"))

    source_map: dict[str, str] = {"profile": ("environment" if merged_env.get("FXAI_TOOLCHAIN_PROFILE") else "fxai.toml" if requested_profile != AUTO_PROFILE else "auto_detected")}

    project_root = _resolve_value(
        name="project_root",
        config_paths=config_paths,
        env=merged_env,
        base_dir=(config_path.parent if config_path else default_project_root),
        env_key="FXAI_PROJECT_ROOT",
        default_value=default_project_root,
        source_map=source_map,
    )
    assert isinstance(project_root, Path)
    repo_root = _resolve_value(
        name="repo_root",
        config_paths=config_paths,
        env=merged_env,
        base_dir=project_root,
        env_key="FXAI_REPO_ROOT",
        default_value=project_root.parent,
        source_map=source_map,
    )
    assert isinstance(repo_root, Path)

    if profile == "headless_ci":
        default_mt5_root = project_root / "MT5"
        default_common_files = project_root / "FILE_COMMON"
    else:
        try:
            default_mt5_root = project_root.parents[2]
        except IndexError:
            default_mt5_root = project_root / "MT5"
        default_common_files = (
            _derive_windows_common_files(merged_env)
            if profile == "windows_native"
            else _derive_macos_common_files(project_root, merged_env)
        )

    mt5_root = _resolve_value(
        name="mt5_root",
        config_paths=config_paths,
        env=merged_env,
        base_dir=project_root,
        env_key="FXAI_MT5_ROOT",
        default_value=default_mt5_root,
        source_map=source_map,
    )
    assert isinstance(mt5_root, Path)

    metaeditor = _resolve_value(
        name="metaeditor",
        config_paths=config_paths,
        env=merged_env,
        base_dir=project_root,
        env_key="FXAI_METAEDITOR",
        default_value=mt5_root / "MetaEditor64.exe",
        source_map=source_map,
    )
    terminal = _resolve_value(
        name="terminal",
        config_paths=config_paths,
        env=merged_env,
        base_dir=project_root,
        env_key="FXAI_TERMINAL",
        default_value=mt5_root / "terminal64.exe",
        source_map=source_map,
    )
    common_files = _resolve_value(
        name="common_files",
        config_paths=config_paths,
        env=merged_env,
        base_dir=project_root,
        env_key="FXAI_COMMON_FILES",
        default_value=default_common_files,
        source_map=source_map,
    )
    runtime_dir = _resolve_value(
        name="runtime_dir",
        config_paths=config_paths,
        env=merged_env,
        base_dir=project_root,
        env_key="FXAI_RUNTIME_DIR",
        default_value=common_files / "FXAI/Runtime",
        source_map=source_map,
    )
    tester_preset_dir = _resolve_value(
        name="tester_preset_dir",
        config_paths=config_paths,
        env=merged_env,
        base_dir=project_root,
        env_key="FXAI_TESTER_PRESET_DIR",
        default_value=mt5_root / "MQL5/Profiles/Tester",
        source_map=source_map,
    )
    common_ini = _resolve_value(
        name="common_ini",
        config_paths=config_paths,
        env=merged_env,
        base_dir=project_root,
        env_key="FXAI_COMMON_INI",
        default_value=mt5_root / "config/common.ini",
        source_map=source_map,
    )
    terminal_ini = _resolve_value(
        name="terminal_ini",
        config_paths=config_paths,
        env=merged_env,
        base_dir=project_root,
        env_key="FXAI_TERMINAL_INI",
        default_value=mt5_root / "config/terminal.ini",
        source_map=source_map,
    )
    mt5_log_dir = _resolve_value(
        name="mt5_log_dir",
        config_paths=config_paths,
        env=merged_env,
        base_dir=project_root,
        env_key="FXAI_MT5_LOG_DIR",
        default_value=mt5_root / "logs",
        source_map=source_map,
    )
    default_db = _resolve_value(
        name="default_db",
        config_paths=config_paths,
        env=merged_env,
        base_dir=project_root,
        env_key="FXAI_DEFAULT_DB",
        default_value=project_root / "Tools/OfflineLab/fxai_offline_lab.turso.db",
        source_map=source_map,
    )
    wine_value = _resolve_value(
        name="wine",
        config_paths=config_paths,
        env=merged_env,
        base_dir=project_root,
        env_key="FXAI_WINE",
        default_value=(DEFAULT_WINE_BINARY if profile == "macos_wine" else None),
        source_map=source_map,
    )
    gui_minimum_macos = _resolve_value(
        name="minimum_macos",
        config_paths=gui_config,
        env=merged_env,
        base_dir=project_root,
        env_key="FXAI_GUI_MINIMUM_MACOS",
        default_value="14.0",
        source_map=source_map,
    )
    gui_release_archive = _resolve_value(
        name="release_archive",
        config_paths=gui_config,
        env=merged_env,
        base_dir=project_root,
        env_key="FXAI_GUI_RELEASE_ARCHIVE",
        default_value="FXAIGUI-macos.zip",
        source_map=source_map,
    )

    assert isinstance(metaeditor, Path)
    assert isinstance(terminal, Path)
    assert isinstance(common_files, Path)
    assert isinstance(runtime_dir, Path)
    assert isinstance(tester_preset_dir, Path)
    assert isinstance(common_ini, Path)
    assert isinstance(terminal_ini, Path)
    assert isinstance(mt5_log_dir, Path)
    assert isinstance(default_db, Path)
    assert isinstance(gui_minimum_macos, str)
    assert isinstance(gui_release_archive, str)

    return FXAIToolchainConfig(
        profile=profile,
        config_path=(config_path if config_path.exists() else None),
        env_path=(env_path if env_path.exists() else None),
        project_root=project_root,
        repo_root=repo_root,
        mt5_root=mt5_root,
        metaeditor=metaeditor,
        terminal=terminal,
        wine=(wine_value if isinstance(wine_value, Path) else None),
        common_files=common_files,
        runtime_dir=runtime_dir,
        tester_preset_dir=tester_preset_dir,
        common_ini=common_ini,
        terminal_ini=terminal_ini,
        mt5_log_dir=mt5_log_dir,
        default_db=default_db,
        gui_minimum_macos=gui_minimum_macos,
        gui_release_archive=gui_release_archive,
        path_sources=source_map,
    )
