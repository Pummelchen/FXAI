from __future__ import annotations

from pathlib import Path

from testlab.toolchain import load_toolchain_config


def test_toolchain_config_reads_fxai_toml_and_env_override(tmp_path: Path):
    project_root = tmp_path / "FXAI"
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


def test_toolchain_config_builds_wine_compile_arguments(tmp_path: Path):
    project_root = tmp_path / "FXAI"
    config = load_toolchain_config(
        project_root_hint=project_root,
        env={
            "FXAI_TOOLCHAIN_PROFILE": "macos_wine",
            "FXAI_PROJECT_ROOT": str(project_root),
            "FXAI_MT5_ROOT": str(project_root / "mt5"),
            "FXAI_WINE": str(project_root / "wine64"),
        },
    )

    cmd = config.metaeditor_compile_command(project_root / "FXAI.mq5", project_root / "compile.log")

    assert cmd[0] == str(project_root / "wine64")
    assert cmd[1] == str(project_root / "mt5" / "MetaEditor64.exe")
    assert cmd[2].startswith("/compile:Z:\\")
    assert cmd[3].startswith("/log:Z:\\")
