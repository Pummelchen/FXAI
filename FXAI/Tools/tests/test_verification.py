from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from offline_lab import verification


def test_run_pytest_suite_prefers_repo_local_tools_path():
    captured: dict[str, object] = {}
    saved_run = verification.subprocess.run

    def fake_run(cmd, cwd, env, stdout, stderr, text):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = dict(env)
        return SimpleNamespace(returncode=0, stdout="ok\n")

    verification.subprocess.run = fake_run
    try:
        repo_root = Path("/tmp/fxai-repo")
        payload = verification.run_pytest_suite(repo_root)
    finally:
        verification.subprocess.run = saved_run

    assert payload["ok"] is True
    assert payload["returncode"] == 0
    assert str(captured["cwd"]) == str(repo_root / "FXAI")
    pythonpath = str(dict(captured["env"]).get("PYTHONPATH", ""))
    assert pythonpath.split(verification.os.pathsep)[0] == str(repo_root / "FXAI" / "Tools")
