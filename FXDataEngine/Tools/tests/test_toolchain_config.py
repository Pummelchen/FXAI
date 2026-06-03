from __future__ import annotations

from pathlib import Path

from testlab.toolchain import load_toolchain_config


ROOT = Path(__file__).resolve().parents[3]


def test_toolchain_config_reads_fxai_toml_and_env_override(tmp_path: Path):
    project_root = tmp_path / "FXDataEngine"
    project_root.mkdir(parents=True)
    (project_root / "fxai.toml").write_text(
        """
        [toolchain]
        profile = "headless_ci"

        [paths]
        common_files = "FILE_COMMON"
        default_db = "Tools/OfflineLab/custom.db"

        [gui]
        minimum_macos = "15.0"
        release_archive = "Custom.zip"
        """,
        encoding="utf-8",
    )

    config = load_toolchain_config(
        project_root_hint=project_root,
        env={
            "FXAI_RUNTIME_DIR": str(project_root / "override" / "FXAI" / "Runtime"),
        },
    )

    assert config.profile == "headless_ci"
    assert config.common_files == project_root / "FILE_COMMON"
    assert config.runtime_dir == project_root / "override" / "FXAI" / "Runtime"
    assert config.default_db == project_root / "Tools/OfflineLab/custom.db"
    assert config.gui_minimum_macos == "15.0"
    assert config.gui_release_archive == "Custom.zip"


def test_toolchain_config_reads_fxai_config_from_dotenv(tmp_path: Path):
    project_root = tmp_path / "FXDataEngine"
    config_dir = project_root / "Config"
    config_dir.mkdir(parents=True)
    (project_root / ".env").write_text(
        """
        FXAI_CONFIG=Config/fxai-toolchain.toml
        """,
        encoding="utf-8",
    )
    config_path = config_dir / "fxai-toolchain.toml"
    config_path.write_text(
        """
        [toolchain]
        profile = "headless_ci"

        [paths]
        runtime_dir = "ConfiguredRuntime"
        """,
        encoding="utf-8",
    )

    config = load_toolchain_config(project_root_hint=project_root, env={})

    assert config.config_path == config_path
    assert config.profile == "headless_ci"
    assert config.runtime_dir == project_root / "ConfiguredRuntime"
    assert config.path_sources["runtime_dir"] == "fxai.toml"


def test_toolchain_config_process_env_overrides_dotenv_config_path(tmp_path: Path):
    project_root = tmp_path / "FXDataEngine"
    config_dir = project_root / "Config"
    config_dir.mkdir(parents=True)
    (project_root / ".env").write_text(
        """
        FXAI_CONFIG=Config/dotenv.toml
        """,
        encoding="utf-8",
    )
    (config_dir / "dotenv.toml").write_text(
        """
        [toolchain]
        profile = "macos_wine"
        """,
        encoding="utf-8",
    )
    process_config = config_dir / "process.toml"
    process_config.write_text(
        """
        [toolchain]
        profile = "headless_ci"
        """,
        encoding="utf-8",
    )

    config = load_toolchain_config(
        project_root_hint=project_root,
        env={
            "FXAI_CONFIG": "Config/process.toml",
        },
    )

    assert config.config_path == process_config
    assert config.profile == "headless_ci"


def test_toolchain_config_process_profile_overrides_toml_profile(tmp_path: Path):
    project_root = tmp_path / "FXDataEngine"
    project_root.mkdir(parents=True)
    (project_root / "fxai.toml").write_text(
        """
        [toolchain]
        profile = "macos_wine"
        """,
        encoding="utf-8",
    )

    config = load_toolchain_config(
        project_root_hint=project_root,
        env={
            "FXAI_TOOLCHAIN_PROFILE": " headless_ci ",
        },
    )

    assert config.profile == "headless_ci"
    assert config.path_sources["profile"] == "environment"
    assert not config.uses_wine


def test_toolchain_config_supports_headless_swift_offline_profile(tmp_path: Path):
    project_root = tmp_path / "FXDataEngine"
    config = load_toolchain_config(
        project_root_hint=project_root,
        env={
            "FXAI_TOOLCHAIN_PROFILE": "headless_ci",
            "FXAI_PROJECT_ROOT": str(project_root),
            "FXAI_RUNTIME_DIR": str(project_root / "runtime"),
        },
    )

    assert config.profile == "headless_ci"
    assert not config.uses_wine
    assert config.project_root == project_root
    assert config.runtime_dir == project_root / "runtime"
    assert config.default_db == project_root / "Tools/OfflineLab/fxai_offline_lab.turso.db"


def test_swift_package_declares_current_language_and_platform_standard():
    package = (ROOT / "FXDataEngine/Package.swift").read_text(encoding="utf-8")

    assert "// swift-tools-version: 6.3" in package
    assert '.macOS("26.0")' in package
    assert "swiftLanguageModes: [.v6]" in package
