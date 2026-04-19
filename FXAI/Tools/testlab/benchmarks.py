from __future__ import annotations

import csv
import datetime as dt
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .release_gate import (
    DEFAULT_RELEASE_GATE_MIN_SCORE,
    DEFAULT_RELEASE_GATE_MIN_STABILITY,
    release_gate_policy_snapshot,
)
from .reporting import grade, load_current_summary, render_summary_report
from .shared import (
    BASELINES_DIR,
    ROOT,
    SYMBOL_PACKS,
    git_dirty,
    git_head_commit,
    load_json,
    load_oracles,
    sha256_path,
    write_json,
)

BENCHMARKS_DIR = ROOT / "Tools/Benchmarks"
REFERENCE_AUDIT_DIR = BENCHMARKS_DIR / "ReferenceAudit"
RELEASE_NOTES_DIR = BENCHMARKS_DIR / "ReleaseNotes"
BENCHMARK_MATRIX_JSON = BENCHMARKS_DIR / "benchmark_matrix.json"
BENCHMARK_MATRIX_TSV = BENCHMARKS_DIR / "benchmark_matrix.tsv"
BENCHMARK_MATRIX_MD = BENCHMARKS_DIR / "benchmark_matrix.md"
PROMOTION_CRITERIA_JSON = BENCHMARKS_DIR / "promotion_criteria.json"
PROMOTION_CRITERIA_MD = BENCHMARKS_DIR / "promotion_criteria.md"
BENCHMARK_SUITE_MANIFEST = BENCHMARKS_DIR / "benchmark_suite_manifest.json"

DEFAULT_REFERENCE_REPORT = BASELINES_DIR / "sample_baseline.tsv"
DEFAULT_REFERENCE_SUMMARY = BASELINES_DIR / "sample_baseline.summary.json"
DEFAULT_REFERENCE_SYMBOL = "EURUSD"
DEFAULT_REFERENCE_SYMBOL_PACK = ""
DEFAULT_REFERENCE_BROKER_PROFILE = "default"
DEFAULT_REFERENCE_EXECUTION_PROFILE = "default"
DEFAULT_REFERENCE_RUNTIME_MODE = "research"
DEFAULT_REFERENCE_HORIZON = 5
DEFAULT_REFERENCE_STRATEGY_PROFILE_ID = "reference/legacy-audit"
DEFAULT_REFERENCE_STRATEGY_PROFILE_VERSION = 0
DEFAULT_RELEASE_TAG = "reference"

FXAI_ROOT_TOKEN = "<FXAI_ROOT>"

BENCHMARK_TSV_FIELDS = [
    "benchmark_source",
    "profile_name",
    "symbol_pack",
    "symbol",
    "broker_profile",
    "execution_profile",
    "runtime_mode",
    "horizon_minutes",
    "plugin_name",
    "strategy_profile_label",
    "audit_score",
    "ranking_score",
    "walkforward_score",
    "adversarial_score",
    "grade",
    "support_count",
    "issue_count",
    "scenario_count",
    "promotion_tier",
]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_float(value: Any) -> float | None:
    try:
        if value in (None, "", "null"):
            return None
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> int | None:
    try:
        if value in (None, "", "null"):
            return None
        return int(float(value))
    except Exception:
        return None


def _fmt_metric(value: Any, *, precision: int = 2) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{precision}f}"


def _fmt_delta_metric(value: Any, *, precision: int = 2) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:+.{precision}f}"


def _portable_path(path: str | Path, *, root: Path = ROOT) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    candidate = Path(raw)
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return raw
    return f"{FXAI_ROOT_TOKEN}/{relative.as_posix()}"


def _resolve_portable_path(raw: str | Path, *, root: Path = ROOT) -> Path:
    text = str(raw or "").strip()
    if not text:
        return root
    if text.startswith(f"{FXAI_ROOT_TOKEN}/"):
        return root / text[len(FXAI_ROOT_TOKEN) + 1:]
    if text == FXAI_ROOT_TOKEN:
        return root
    return Path(text)


def _relative_link(from_path: Path, target_path: Path) -> str:
    try:
        return Path(os.path.relpath(target_path.resolve(), start=from_path.resolve().parent)).as_posix()
    except Exception:
        pass
    try:
        return target_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return target_path.as_posix()


def _derive_symbol_pack(symbols: list[str], explicit_symbol_pack: str = "") -> str:
    clean_symbols = [str(symbol).upper() for symbol in symbols if str(symbol).strip()]
    if explicit_symbol_pack.strip():
        return explicit_symbol_pack.strip()
    if len(clean_symbols) == 1:
        return f"single:{clean_symbols[0]}"
    symbol_set = sorted(clean_symbols)
    for pack_name, pack_symbols in SYMBOL_PACKS.items():
        if sorted(pack_symbols) == symbol_set:
            return pack_name
    return "custom:" + ",".join(symbol_set)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _top_summary_plugin(summary: dict) -> tuple[str, dict]:
    ranked = sorted(
        summary.get("plugins", {}).items(),
        key=lambda item: float(dict(item[1]).get("score", 0.0) or 0.0),
        reverse=True,
    )
    if not ranked:
        return "unknown", {}
    return str(ranked[0][0]), dict(ranked[0][1])


def _support_metric(rows: list[dict], key: str) -> float | None:
    values = [_as_float(row.get(key)) for row in rows]
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / float(len(filtered))


def _strategy_profile_label(profile_id: str, profile_version: int | None) -> str:
    version = int(profile_version or 0)
    return f"{profile_id}@v{version}"


def _render_benchmark_markdown_rows(rows: list[dict], matrix_path: Path) -> str:
    lines = [
        "# FXAI Public Benchmark Matrix",
        "",
        "This matrix is generated from committed FXAI audit and promotion artifacts. It is intended to show benchmark context, not only top-line claims.",
        "",
        "| Source | Symbol Pack | Symbol | Broker | Execution | Horizon | Strategy Profile | Plugin | Audit | Ranking | Walkforward | Adversarial | Grade | Reference |",
        "|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        artifacts = dict(row.get("artifacts", {}))
        reference_target = artifacts.get("summary_report_md") or artifacts.get("strategy_profile_manifest") or artifacts.get("report_tsv") or ""
        reference_path = _resolve_portable_path(reference_target) if reference_target else None
        if reference_path is not None and reference_path.exists():
            reference = f"[artifact]({_relative_link(matrix_path, reference_path)})"
        elif reference_target:
            reference = reference_target
        else:
            reference = "n/a"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("benchmark_source", "")),
                    str(row.get("symbol_pack", "")),
                    str(row.get("symbol", "")),
                    str(row.get("broker_profile", "")),
                    str(row.get("execution_profile", "")),
                    str(row.get("horizon_minutes", "")),
                    str(row.get("strategy_profile_label", "")),
                    str(row.get("plugin_name", "")),
                    _fmt_metric(row.get("audit_score"), precision=1),
                    _fmt_metric(row.get("ranking_score"), precision=2),
                    _fmt_metric(row.get("walkforward_score"), precision=1),
                    _fmt_metric(row.get("adversarial_score"), precision=1),
                    str(row.get("grade", "")),
                    reference,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _write_benchmark_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_TSV_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in BENCHMARK_TSV_FIELDS})


def _render_promotion_criteria_markdown(policy: dict) -> str:
    defaults = dict(policy.get("defaults", {}))
    artifact_budgets = dict(policy.get("artifact_size_budgets_bytes", {}))
    performance = dict(policy.get("performance_thresholds", {}))
    walkforward = dict(policy.get("walkforward_thresholds", {}))
    adversarial = dict(policy.get("adversarial_thresholds", {}))
    macro = dict(policy.get("macro_dataset_thresholds", {}))
    lines = [
        "# FXAI Promotion Criteria",
        "",
        "These are the exact release-gate thresholds exported from `Tools/testlab/release_gate.py`. Public benchmark cards and release notes should be interpreted against this policy, not against ad hoc screenshots.",
        "",
        "## Core Admission Thresholds",
        "",
        f"- Minimum audit score: {float(defaults.get('min_score', DEFAULT_RELEASE_GATE_MIN_SCORE)):.1f}",
        f"- Minimum cross-symbol stability: {float(defaults.get('min_stability', DEFAULT_RELEASE_GATE_MIN_STABILITY)):.2f}",
        f"- Require market replay pack: {bool(defaults.get('require_market_replay', False))}",
        f"- Fail on unresolved issues: {bool(defaults.get('fail_on_issues', False))}",
        "",
        "## Walkforward Thresholds",
        "",
        f"- PBO max: {float(walkforward.get('pbo_max', 0.45)):.2f}",
        f"- DSR(proxy) min: {float(walkforward.get('dsr_min', 0.35)):.2f}",
        f"- Pass-rate min: {float(walkforward.get('pass_rate_min', 0.55)):.2f}",
        "",
        "## Adversarial Thresholds",
        "",
        f"- Score min: {float(adversarial.get('score_min', 68.0)):.1f}",
        f"- Calibration error max: {float(adversarial.get('calibration_error_max', 0.26)):.3f}",
        f"- Path-quality error max: {float(adversarial.get('path_quality_error_max', 0.50)):.3f}",
        "",
        "## Macro Dataset Thresholds",
        "",
        f"- Schema version min: {int(float(macro.get('schema_version_min', 2.0)))}",
        f"- Source trust min: {float(macro.get('avg_source_trust_min', 0.60)):.2f}",
        f"- Macro coverage min: {float(macro.get('coverage_min', 0.60)):.2f}",
        f"- Macro event-rate min: {float(macro.get('event_rate_min', 0.06)):.2f}",
        f"- Provenance trust min: {float(macro.get('provenance_trust_min', 0.55)):.2f}",
        f"- Currency relevance min: {float(macro.get('currency_relevance_min', 0.45)):.2f}",
        f"- Revision-chain coverage min: {int(float(macro.get('distinct_revision_chains_min', 1.0)))}",
        "",
        "## Runtime Performance Thresholds",
        "",
        f"- Predict mean max: {float(performance.get('predict_mean_ms_max', 1.60)):.3f} ms",
        f"- Update mean max: {float(performance.get('update_mean_ms_max', 1.25)):.3f} ms",
        f"- Working set max: {float(performance.get('working_set_kb_max', 4096.0)):.1f} KB",
        f"- Runtime total mean max: {float(performance.get('runtime_total_mean_ms_max', 12.0)):.3f} ms",
        "",
        "## Artifact Size Budgets",
        "",
    ]
    for artifact_name, budget in sorted(artifact_budgets.items()):
        lines.append(f"- `{artifact_name}`: {int(budget)} bytes max")
    lines.append("")
    return "\n".join(lines)


def _delta(new_value: Any, old_value: Any) -> float | None:
    new_number = _as_float(new_value)
    old_number = _as_float(old_value)
    if new_number is None or old_number is None:
        return None
    return new_number - old_number


def _context_key(row: dict) -> tuple[str, str, str, int, str]:
    return (
        str(row.get("symbol_pack", "")),
        str(row.get("broker_profile", "")),
        str(row.get("execution_profile", "")),
        int(_as_int(row.get("horizon_minutes")) or 0),
        str(row.get("symbol", "")),
    )


def _build_release_delta(previous_rows: list[dict], current_rows: list[dict]) -> list[dict]:
    previous_map = {_context_key(row): row for row in previous_rows}
    current_map = {_context_key(row): row for row in current_rows}
    keys = sorted(set(previous_map) | set(current_map))
    deltas: list[dict] = []
    for key in keys:
        previous = previous_map.get(key)
        current = current_map.get(key)
        if previous is None and current is not None:
            deltas.append({"change_type": "added", "context": key, "current": current})
            continue
        if current is None and previous is not None:
            deltas.append({"change_type": "removed", "context": key, "previous": previous})
            continue
        if previous is None or current is None:
            continue
        deltas.append(
            {
                "change_type": "changed",
                "context": key,
                "previous": previous,
                "current": current,
                "plugin_changed": str(previous.get("plugin_name", "")) != str(current.get("plugin_name", "")),
                "strategy_profile_changed": str(previous.get("strategy_profile_label", "")) != str(current.get("strategy_profile_label", "")),
                "audit_score_delta": _delta(current.get("audit_score"), previous.get("audit_score")),
                "ranking_score_delta": _delta(current.get("ranking_score"), previous.get("ranking_score")),
                "walkforward_score_delta": _delta(current.get("walkforward_score"), previous.get("walkforward_score")),
                "adversarial_score_delta": _delta(current.get("adversarial_score"), previous.get("adversarial_score")),
            }
        )
    return deltas


def _render_release_notes_markdown(payload: dict, notes_path: Path) -> str:
    lines = [
        f"# FXAI Release Notes: {payload.get('release_tag', DEFAULT_RELEASE_TAG)}",
        "",
        str(payload.get("summary", "")),
        "",
        f"Generated at: {payload.get('generated_at_utc', '')}",
        "",
    ]
    for delta in payload.get("deltas", []):
        change_type = str(delta.get("change_type", "changed"))
        if change_type == "added":
            current = dict(delta.get("current", {}))
            lines.extend(
                [
                    f"## Added Benchmark Context: {current.get('symbol_pack', '')} / {current.get('symbol', '')}",
                    "",
                    f"- Broker profile: {current.get('broker_profile', '')}",
                    f"- Execution profile: {current.get('execution_profile', '')}",
                    f"- Horizon: {current.get('horizon_minutes', '')} minutes",
                    f"- Strategy profile: {current.get('strategy_profile_label', '')}",
                    f"- Plugin: {current.get('plugin_name', '')}",
                    f"- Audit score: {_fmt_metric(current.get('audit_score'), precision=1)}",
                    "",
                ]
            )
            continue
        if change_type == "removed":
            previous = dict(delta.get("previous", {}))
            lines.extend(
                [
                    f"## Removed Benchmark Context: {previous.get('symbol_pack', '')} / {previous.get('symbol', '')}",
                    "",
                    f"- Previous plugin: {previous.get('plugin_name', '')}",
                    f"- Previous strategy profile: {previous.get('strategy_profile_label', '')}",
                    "",
                ]
            )
            continue

        previous = dict(delta.get("previous", {}))
        current = dict(delta.get("current", {}))
        lines.extend(
            [
                f"## Benchmark Delta: {current.get('symbol_pack', '')} / {current.get('symbol', '')}",
                "",
                f"- Broker profile: {current.get('broker_profile', '')}",
                f"- Execution profile: {current.get('execution_profile', '')}",
                f"- Horizon: {current.get('horizon_minutes', '')} minutes",
            ]
        )
        if bool(delta.get("plugin_changed", False)):
            lines.append(f"- Model change: {previous.get('plugin_name', '')} -> {current.get('plugin_name', '')}")
        if bool(delta.get("strategy_profile_changed", False)):
            lines.append(
                f"- Strategy profile change: {previous.get('strategy_profile_label', '')} -> {current.get('strategy_profile_label', '')}"
            )
        lines.extend(
            [
                f"- Audit score: {_fmt_metric(previous.get('audit_score'), precision=1)} -> {_fmt_metric(current.get('audit_score'), precision=1)} "
                f"({_fmt_delta_metric(delta.get('audit_score_delta'), precision=1)})",
                f"- Ranking score: {_fmt_metric(previous.get('ranking_score'), precision=2)} -> {_fmt_metric(current.get('ranking_score'), precision=2)} "
                f"({_fmt_delta_metric(delta.get('ranking_score_delta'), precision=2)})",
                f"- Walkforward score: {_fmt_metric(previous.get('walkforward_score'), precision=1)} -> {_fmt_metric(current.get('walkforward_score'), precision=1)} "
                f"({_fmt_delta_metric(delta.get('walkforward_score_delta'), precision=1)})",
                f"- Adversarial score: {_fmt_metric(previous.get('adversarial_score'), precision=1)} -> {_fmt_metric(current.get('adversarial_score'), precision=1)} "
                f"({_fmt_delta_metric(delta.get('adversarial_score_delta'), precision=1)})",
                "",
            ]
        )
    lines.append(f"Reference benchmark matrix: [{BENCHMARK_MATRIX_MD.name}]({_relative_link(notes_path, BENCHMARK_MATRIX_MD)})")
    lines.append("")
    return "\n".join(lines)


def _reference_audit_row(*,
                         root: Path,
                         report_tsv: Path,
                         summary_json: Path,
                         manifest_json: Path,
                         summary_report_md: Path,
                         symbol: str,
                         symbol_pack: str,
                         broker_profile: str,
                         execution_profile: str,
                         runtime_mode: str,
                         horizon_minutes: int,
                         strategy_profile_id: str,
                         strategy_profile_version: int) -> dict[str, Any]:
    summary = load_current_summary(summary_json, load_oracles())
    plugin_name, plugin_info = _top_summary_plugin(summary)
    scenarios = dict(plugin_info.get("scenarios", {}))
    return {
        "benchmark_source": "reference_audit",
        "profile_name": "reference",
        "symbol_pack": symbol_pack,
        "symbol": symbol,
        "broker_profile": broker_profile,
        "execution_profile": execution_profile,
        "runtime_mode": runtime_mode,
        "horizon_minutes": int(horizon_minutes),
        "plugin_name": plugin_name,
        "strategy_profile_id": strategy_profile_id,
        "strategy_profile_version": int(strategy_profile_version),
        "strategy_profile_label": _strategy_profile_label(strategy_profile_id, strategy_profile_version),
        "audit_score": _as_float(plugin_info.get("score")),
        "ranking_score": None,
        "walkforward_score": _as_float(dict(scenarios.get("market_walkforward", {})).get("score")),
        "adversarial_score": _as_float(dict(scenarios.get("market_adversarial", {})).get("score")),
        "grade": str(plugin_info.get("grade", grade(float(plugin_info.get("score", 0.0) or 0.0)))),
        "support_count": 0,
        "issue_count": len(list(plugin_info.get("issues", []))) + len(list(plugin_info.get("findings", []))),
        "scenario_count": len(scenarios),
        "promotion_tier": "reference-only",
        "artifacts": {
            "report_tsv": _portable_path(report_tsv, root=root),
            "summary_json": _portable_path(summary_json, root=root),
            "manifest_json": _portable_path(manifest_json, root=root),
            "summary_report_md": _portable_path(summary_report_md, root=root),
        },
    }


def _build_reference_audit_bundle(*,
                                  root: Path,
                                  output_dir: Path,
                                  reference_report: Path,
                                  reference_summary: Path,
                                  symbol: str,
                                  symbol_pack: str,
                                  broker_profile: str,
                                  execution_profile: str,
                                  runtime_mode: str,
                                  horizon_minutes: int,
                                  strategy_profile_id: str,
                                  strategy_profile_version: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_tsv = output_dir / "sample_audit.tsv"
    summary_json = output_dir / "sample_audit.summary.json"
    manifest_json = output_dir / "sample_audit.manifest.json"
    summary_report_md = output_dir / "sample_audit_summary.md"

    shutil.copy2(reference_report, report_tsv)
    shutil.copy2(reference_summary, summary_json)
    summary = load_current_summary(summary_json, load_oracles())
    summary_report_md.write_text(
        render_summary_report(summary, execution_profile, Path(_portable_path(manifest_json, root=root))),
        encoding="utf-8",
    )

    bundle_manifest = {
        "schema_version": 1,
        "artifact_type": "fxai_reference_audit_bundle",
        "generated_at_utc": _utc_now(),
        "benchmark_context": {
            "symbol": symbol,
            "symbol_pack": symbol_pack,
            "broker_profile": broker_profile,
            "execution_profile": execution_profile,
            "runtime_mode": runtime_mode,
            "horizon_minutes": int(horizon_minutes),
            "strategy_profile_id": strategy_profile_id,
            "strategy_profile_version": int(strategy_profile_version),
        },
        "source_artifacts": {
            "report_tsv": _portable_path(reference_report, root=root),
            "summary_json": _portable_path(reference_summary, root=root),
        },
        "published_artifacts": {
            "report_tsv": _portable_path(report_tsv, root=root),
            "summary_json": _portable_path(summary_json, root=root),
            "manifest_json": _portable_path(manifest_json, root=root),
            "summary_report_md": _portable_path(summary_report_md, root=root),
        },
        "hashes": {
            "report_tsv": sha256_path(report_tsv),
            "summary_json": sha256_path(summary_json),
            "summary_report_md": sha256_path(summary_report_md),
        },
        "reproducibility": {
            "repo_root": str(root),
            "repo_head": git_head_commit(root),
            "repo_dirty": git_dirty(root),
        },
    }
    write_json(manifest_json, bundle_manifest)
    row = _reference_audit_row(
        root=root,
        report_tsv=report_tsv,
        summary_json=summary_json,
        manifest_json=manifest_json,
        summary_report_md=summary_report_md,
        symbol=symbol,
        symbol_pack=symbol_pack,
        broker_profile=broker_profile,
        execution_profile=execution_profile,
        runtime_mode=runtime_mode,
        horizon_minutes=horizon_minutes,
        strategy_profile_id=strategy_profile_id,
        strategy_profile_version=strategy_profile_version,
    )
    bundle_manifest["benchmark_row"] = row
    write_json(manifest_json, bundle_manifest)
    return {
        "row": row,
        "manifest": bundle_manifest,
        "paths": {
            "report_tsv": report_tsv,
            "summary_json": summary_json,
            "manifest_json": manifest_json,
            "summary_report_md": summary_report_md,
        },
    }


def _build_promoted_profile_rows(*, root: Path, profile_name: str) -> list[dict[str, Any]]:
    promoted_best_path = root / f"Tools/OfflineLab/Profiles/{profile_name}/promoted_best.json"
    if not promoted_best_path.exists():
        return []
    rows = json.loads(promoted_best_path.read_text(encoding="utf-8"))
    built_rows: list[dict[str, Any]] = []
    for row in rows:
        params = json.loads(str(row.get("parameters_json", "{}") or "{}"))
        support_rows = json.loads(str(row.get("support_json", "[]") or "[]"))
        symbol = str(row.get("symbol", "")).upper()
        strategy_manifest_path = _resolve_portable_path(str(row.get("strategy_profile_manifest_path", "") or ""), root=root)
        strategy_manifest = load_json(strategy_manifest_path) if strategy_manifest_path.exists() else {}
        context = dict(strategy_manifest.get("context", {}))
        audit_values = dict(dict(strategy_manifest.get("compiled", {})).get("audit_values", {}))
        broker_profile = str(context.get("resolved_broker") or context.get("broker_profile") or params.get("broker_profile") or "default")
        execution_profile = str(audit_values.get("execution_profile") or params.get("execution_profile") or "default")
        runtime_mode = str(context.get("runtime_mode") or params.get("runtime_mode") or "research")
        horizon_minutes = int(audit_values.get("horizon") or params.get("horizon") or DEFAULT_REFERENCE_HORIZON)
        strategy_profile_id = str(row.get("strategy_profile_id") or "strategy/default")
        strategy_profile_version = int(row.get("strategy_profile_version") or 1)
        live_deploy_path = root / f"Tools/OfflineLab/ResearchOS/{profile_name}/live_deploy_{symbol}.json"
        live_deploy = load_json(live_deploy_path) if live_deploy_path.exists() else {}
        built_rows.append(
            {
                "benchmark_source": "promoted_profile",
                "profile_name": profile_name,
                "symbol_pack": _derive_symbol_pack([symbol]),
                "symbol": symbol,
                "broker_profile": broker_profile,
                "execution_profile": execution_profile,
                "runtime_mode": runtime_mode,
                "horizon_minutes": horizon_minutes,
                "plugin_name": str(row.get("plugin_name", "")),
                "strategy_profile_id": strategy_profile_id,
                "strategy_profile_version": strategy_profile_version,
                "strategy_profile_label": _strategy_profile_label(strategy_profile_id, strategy_profile_version),
                "audit_score": _as_float(row.get("score")),
                "ranking_score": _as_float(row.get("ranking_score")),
                "walkforward_score": _support_metric(support_rows, "walkforward_score"),
                "adversarial_score": _support_metric(support_rows, "adversarial_score"),
                "grade": grade(float(row.get("score", 0.0) or 0.0)),
                "support_count": int(row.get("support_count", 0) or 0),
                "issue_count": 0,
                "scenario_count": len(list(audit_values.get("scenario_list", []))),
                "promotion_tier": str(live_deploy.get("promotion_tier", "audit-approved")),
                "artifacts": {
                    "promoted_best_json": _portable_path(promoted_best_path, root=root),
                    "audit_set": str(row.get("audit_set_path", "")),
                    "ea_set": str(row.get("ea_set_path", "")),
                    "strategy_profile_manifest": str(row.get("strategy_profile_manifest_path", "")),
                    "live_deploy_json": _portable_path(live_deploy_path, root=root) if live_deploy_path.exists() else "",
                },
            }
        )
    built_rows.sort(
        key=lambda item: (
            str(item.get("symbol_pack", "")),
            str(item.get("symbol", "")),
            -float(item.get("audit_score", 0.0) or 0.0),
            str(item.get("plugin_name", "")),
        )
    )
    return built_rows


def publish_benchmark_suite(*,
                            root: Path = ROOT,
                            output_dir: Path | None = None,
                            profile_name: str = "bestparams",
                            reference_report: Path = DEFAULT_REFERENCE_REPORT,
                            reference_summary: Path = DEFAULT_REFERENCE_SUMMARY,
                            reference_symbol: str = DEFAULT_REFERENCE_SYMBOL,
                            reference_symbol_pack: str = DEFAULT_REFERENCE_SYMBOL_PACK,
                            reference_broker_profile: str = DEFAULT_REFERENCE_BROKER_PROFILE,
                            reference_execution_profile: str = DEFAULT_REFERENCE_EXECUTION_PROFILE,
                            reference_runtime_mode: str = DEFAULT_REFERENCE_RUNTIME_MODE,
                            reference_horizon: int = DEFAULT_REFERENCE_HORIZON,
                            reference_strategy_profile_id: str = DEFAULT_REFERENCE_STRATEGY_PROFILE_ID,
                            reference_strategy_profile_version: int = DEFAULT_REFERENCE_STRATEGY_PROFILE_VERSION,
                            release_tag: str = DEFAULT_RELEASE_TAG) -> dict[str, Any]:
    root = root.resolve()
    output_dir = (output_dir or BENCHMARKS_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = output_dir / "ReferenceAudit"
    release_notes_dir = output_dir / "ReleaseNotes"
    release_notes_dir.mkdir(parents=True, exist_ok=True)

    symbol_pack = _derive_symbol_pack([reference_symbol], reference_symbol_pack)
    reference_bundle = _build_reference_audit_bundle(
        root=root,
        output_dir=reference_dir,
        reference_report=reference_report,
        reference_summary=reference_summary,
        symbol=reference_symbol,
        symbol_pack=symbol_pack,
        broker_profile=reference_broker_profile,
        execution_profile=reference_execution_profile,
        runtime_mode=reference_runtime_mode,
        horizon_minutes=reference_horizon,
        strategy_profile_id=reference_strategy_profile_id,
        strategy_profile_version=reference_strategy_profile_version,
    )
    current_rows = _build_promoted_profile_rows(root=root, profile_name=profile_name)
    benchmark_rows = [reference_bundle["row"]] + current_rows

    matrix_json = output_dir / "benchmark_matrix.json"
    matrix_tsv = output_dir / "benchmark_matrix.tsv"
    matrix_md = output_dir / "benchmark_matrix.md"
    promotion_criteria_json = output_dir / "promotion_criteria.json"
    promotion_criteria_md = output_dir / "promotion_criteria.md"
    release_notes_json = release_notes_dir / f"{release_tag}_release_notes.json"
    release_notes_md = release_notes_dir / f"{release_tag}_release_notes.md"
    suite_manifest_path = output_dir / "benchmark_suite_manifest.json"

    matrix_payload = {
        "schema_version": 1,
        "artifact_type": "fxai_public_benchmark_matrix",
        "generated_at_utc": _utc_now(),
        "profile_name": profile_name,
        "rows": benchmark_rows,
    }
    write_json(matrix_json, matrix_payload)
    _write_benchmark_tsv(matrix_tsv, benchmark_rows)
    matrix_md.write_text(_render_benchmark_markdown_rows(benchmark_rows, matrix_md), encoding="utf-8")

    promotion_policy = release_gate_policy_snapshot(
        min_score=DEFAULT_RELEASE_GATE_MIN_SCORE,
        min_stability=DEFAULT_RELEASE_GATE_MIN_STABILITY,
    )
    write_json(promotion_criteria_json, promotion_policy)
    promotion_criteria_md.write_text(_render_promotion_criteria_markdown(promotion_policy), encoding="utf-8")

    previous_rows = [reference_bundle["row"]]
    release_notes_payload = {
        "schema_version": 1,
        "artifact_type": "fxai_release_notes",
        "release_tag": release_tag,
        "generated_at_utc": _utc_now(),
        "summary": "Release notes compare the current benchmark snapshot against the bundled reference audit context so model and profile changes are tied to visible deltas.",
        "previous_snapshot": [dict(row) for row in previous_rows],
        "current_snapshot": [dict(row) for row in current_rows],
        "deltas": _build_release_delta(previous_rows, current_rows),
    }
    write_json(release_notes_json, release_notes_payload)
    release_notes_md.write_text(_render_release_notes_markdown(release_notes_payload, release_notes_md), encoding="utf-8")

    suite_manifest = {
        "schema_version": 1,
        "artifact_type": "fxai_benchmark_suite",
        "generated_at_utc": _utc_now(),
        "profile_name": profile_name,
        "release_tag": release_tag,
        "reference_bundle": {
            "manifest_json": _portable_path(reference_bundle["paths"]["manifest_json"], root=root),
            "report_tsv": _portable_path(reference_bundle["paths"]["report_tsv"], root=root),
            "summary_json": _portable_path(reference_bundle["paths"]["summary_json"], root=root),
            "summary_report_md": _portable_path(reference_bundle["paths"]["summary_report_md"], root=root),
        },
        "artifacts": {
            "benchmark_matrix_json": _portable_path(matrix_json, root=root),
            "benchmark_matrix_tsv": _portable_path(matrix_tsv, root=root),
            "benchmark_matrix_md": _portable_path(matrix_md, root=root),
            "promotion_criteria_json": _portable_path(promotion_criteria_json, root=root),
            "promotion_criteria_md": _portable_path(promotion_criteria_md, root=root),
            "release_notes_json": _portable_path(release_notes_json, root=root),
            "release_notes_md": _portable_path(release_notes_md, root=root),
        },
        "hashes": {
            "benchmark_matrix_json": sha256_path(matrix_json),
            "benchmark_matrix_tsv": sha256_path(matrix_tsv),
            "benchmark_matrix_md": sha256_path(matrix_md),
            "promotion_criteria_json": sha256_path(promotion_criteria_json),
            "promotion_criteria_md": sha256_path(promotion_criteria_md),
            "release_notes_json": sha256_path(release_notes_json),
            "release_notes_md": sha256_path(release_notes_md),
        },
    }
    write_json(suite_manifest_path, suite_manifest)
    return suite_manifest


def cmd_publish_benchmarks(args) -> int:
    payload = publish_benchmark_suite(
        root=ROOT,
        output_dir=(Path(args.output_dir).resolve() if str(getattr(args, "output_dir", "") or "").strip() else BENCHMARKS_DIR),
        profile_name=str(getattr(args, "profile", "bestparams") or "bestparams"),
        reference_report=Path(getattr(args, "reference_report", str(DEFAULT_REFERENCE_REPORT))),
        reference_summary=Path(getattr(args, "reference_summary", str(DEFAULT_REFERENCE_SUMMARY))),
        reference_symbol=str(getattr(args, "reference_symbol", DEFAULT_REFERENCE_SYMBOL) or DEFAULT_REFERENCE_SYMBOL),
        reference_symbol_pack=str(getattr(args, "reference_symbol_pack", DEFAULT_REFERENCE_SYMBOL_PACK) or DEFAULT_REFERENCE_SYMBOL_PACK),
        reference_broker_profile=str(getattr(args, "reference_broker_profile", DEFAULT_REFERENCE_BROKER_PROFILE) or DEFAULT_REFERENCE_BROKER_PROFILE),
        reference_execution_profile=str(getattr(args, "reference_execution_profile", DEFAULT_REFERENCE_EXECUTION_PROFILE) or DEFAULT_REFERENCE_EXECUTION_PROFILE),
        reference_runtime_mode=str(getattr(args, "reference_runtime_mode", DEFAULT_REFERENCE_RUNTIME_MODE) or DEFAULT_REFERENCE_RUNTIME_MODE),
        reference_horizon=int(getattr(args, "reference_horizon", DEFAULT_REFERENCE_HORIZON) or DEFAULT_REFERENCE_HORIZON),
        reference_strategy_profile_id=str(getattr(args, "reference_strategy_profile_id", DEFAULT_REFERENCE_STRATEGY_PROFILE_ID) or DEFAULT_REFERENCE_STRATEGY_PROFILE_ID),
        reference_strategy_profile_version=int(getattr(args, "reference_strategy_profile_version", DEFAULT_REFERENCE_STRATEGY_PROFILE_VERSION) or DEFAULT_REFERENCE_STRATEGY_PROFILE_VERSION),
        release_tag=str(getattr(args, "release_tag", DEFAULT_RELEASE_TAG) or DEFAULT_RELEASE_TAG),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
