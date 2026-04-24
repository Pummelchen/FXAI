from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _top_level_plugin_sources() -> list[Path]:
    api_text = _read("API/api.mqh")
    include_paths = re.findall(r'#include "\.\.\\(Plugins\\[^"]+\.mqh)"', api_text)
    return [ROOT / path.replace("\\", "/") for path in include_paths]


def _public_contract_file(source_path: Path) -> Path:
    text = source_path.read_text(encoding="utf-8")
    if all(token in text for token in ("AIId(", "AIName(", "Describe(")):
        return source_path

    stem = source_path.stem
    candidates = [
        source_path.parent / stem / f"{stem}_public.mqh",
        source_path.parent / stem / f"{stem}_public.mqh".replace("__", "_"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise AssertionError(f"no public plugin contract source found for {source_path}")


def test_registry_create_instance_covers_every_top_level_plugin_source():
    api_text = _read("API/api.mqh")
    sources = _top_level_plugin_sources()
    assert sources
    for path in sources:
        assert path.exists(), f"missing plugin source: {path}"

    create_cases = re.findall(r"case \(int\)AI_[A-Z0-9_]+: plugin = new [A-Za-z0-9_]+\(\); break;", api_text)
    assert len(create_cases) == len(sources)
    assert "plugin.Reset();" in api_text


def test_every_top_level_plugin_exposes_required_public_contract_methods():
    for source_path in _top_level_plugin_sources():
        public_path = _public_contract_file(source_path)
        text = public_path.read_text(encoding="utf-8")
        for token in ("AIId(", "AIName(", "Describe("):
            assert token in text, f"{public_path} missing {token}"


def test_plugin_contract_suite_exercises_lifecycle_predict_state_and_synthetic_series_paths():
    suite = _read("Tests/PluginContracts/plugin_contract_suite.mqh")
    required_tokens = [
        "registry.Initialize()",
        "registry.Release()",
        "plugin.DescribeResolved(",
        "plugin.SelfTest()",
        "plugin.Predict(req, hp, prediction)",
        "plugin.SaveStateFile(file_name)",
        "plugin.LoadStateFile(file_name)",
        "plugin.SupportsSyntheticSeries()",
        "plugin.SetSyntheticSeries(",
        "plugin.ClearSyntheticSeries()",
    ]
    for token in required_tokens:
        assert token in suite


def test_audit_runner_can_gate_on_plugin_contract_sanity():
    audit = _read("Tests/FXAI_AuditRunner.mq5")
    audit_core = _read("Tests/audit_core.mqh")
    assert "Audit_RunPluginContractSanity" in audit
    assert "FXAI_AuditPluginContractSelfTest(" in audit
    assert '#include "audit_plugin_contracts.mqh"' in audit_core
