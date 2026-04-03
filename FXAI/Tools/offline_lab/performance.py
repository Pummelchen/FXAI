from __future__ import annotations

import json
from pathlib import Path

from .common import COMMON_PROMOTION_DIR, RESEARCH_DIR, ensure_dir, safe_token
from testlab.shared import RUNTIME_DIR, runtime_artifact_safe_symbol


DEFAULT_BUDGETS = {
    "runtime_total_ms_max": 12.0,
    "predict_ms_mean_max": 1.60,
    "update_ms_mean_max": 1.25,
    "working_set_kb_max": 4096.0,
}

DEFAULT_ARTIFACT_BUDGETS = {
    "fxai_live_deploy": 24 * 1024,
    "fxai_student_router": 24 * 1024,
    "fxai_attribution": 32 * 1024,
    "fxai_world_plan": 32 * 1024,
    "fxai_supervisor_service": 24 * 1024,
    "fxai_supervisor_command": 16 * 1024,
    "fxai_perf": 128 * 1024,
}


def runtime_performance_manifest_path(symbol: str) -> Path:
    return RUNTIME_DIR / f"fxai_perf_{runtime_artifact_safe_symbol(symbol)}.tsv"


def _load_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = []
    header = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line.strip():
            continue
        cols = raw_line.split("\t")
        if header is None:
            header = cols
            continue
        item = {}
        for idx, key in enumerate(header):
            item[key] = cols[idx] if idx < len(cols) else ""
        rows.append(item)
    return rows


def _load_kv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split("\t", 1)
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def build_symbol_artifact_report(symbol: str,
                                 budgets: dict[str, int] | None = None) -> dict[str, object]:
    budgets = dict(DEFAULT_ARTIFACT_BUDGETS if budgets is None else budgets)
    token = safe_token(symbol)
    files = {
        "fxai_live_deploy": COMMON_PROMOTION_DIR / f"fxai_live_deploy_{token}.tsv",
        "fxai_student_router": COMMON_PROMOTION_DIR / f"fxai_student_router_{token}.tsv",
        "fxai_attribution": COMMON_PROMOTION_DIR / f"fxai_attribution_{token}.tsv",
        "fxai_world_plan": COMMON_PROMOTION_DIR / f"fxai_world_plan_{token}.tsv",
        "fxai_supervisor_service": COMMON_PROMOTION_DIR / f"fxai_supervisor_service_{token}.tsv",
        "fxai_supervisor_command": COMMON_PROMOTION_DIR / f"fxai_supervisor_command_{token}.tsv",
        "fxai_perf": runtime_performance_manifest_path(symbol),
    }
    items: list[dict[str, object]] = []
    total_bytes = 0
    budget_failures: list[str] = []
    for key, path in files.items():
        size_bytes = int(path.stat().st_size) if path.exists() and path.is_file() else 0
        total_bytes += size_bytes
        budget_bytes = int(budgets.get(key, 0))
        over_budget = budget_bytes > 0 and size_bytes > budget_bytes
        if over_budget:
            budget_failures.append(f"{key} {size_bytes} > {budget_bytes}")
        items.append(
            {
                "artifact_key": key,
                "path": str(path),
                "exists": bool(path.exists() and path.is_file()),
                "size_bytes": size_bytes,
                "budget_bytes": budget_bytes,
                "over_budget": over_budget,
            }
        )
    return {
        "symbol": symbol,
        "total_bytes": total_bytes,
        "budget_failures": budget_failures,
        "items": items,
    }


def build_symbol_performance_report(symbol: str,
                                    budgets: dict[str, float] | None = None) -> dict[str, object]:
    budgets = dict(DEFAULT_BUDGETS if budgets is None else budgets)
    deploy = _load_kv(COMMON_PROMOTION_DIR / f"fxai_live_deploy_{safe_token(symbol)}.tsv")
    try:
        budgets["runtime_total_ms_max"] = float(deploy.get("performance_budget_ms", budgets["runtime_total_ms_max"]))
    except Exception:
        pass
    path = runtime_performance_manifest_path(symbol)
    rows = _load_tsv(path)
    report: dict[str, object] = {
        "symbol": symbol,
        "artifact_path": str(path),
        "exists": path.exists(),
        "runtime_total_ms": 0.0,
        "stage_rows": [],
        "plugin_rows": [],
        "budget_failures": [],
        "artifact_report": build_symbol_artifact_report(symbol),
    }
    if not rows:
        return report
    stage_rows = [row for row in rows if row.get("row_type") == "stage"]
    plugin_rows = [row for row in rows if row.get("row_type") == "plugin"]
    report["stage_rows"] = stage_rows
    report["plugin_rows"] = plugin_rows

    total_stage = 0.0
    for row in stage_rows:
        total_stage += _f(row, "mean_ms")
    report["runtime_total_ms"] = total_stage
    if total_stage > budgets["runtime_total_ms_max"]:
        report["budget_failures"].append(f"runtime_total_ms {total_stage:.3f} > {budgets['runtime_total_ms_max']:.3f}")

    for row in plugin_rows:
        mean_predict = _f(row, "predict_mean_ms")
        mean_update = _f(row, "update_mean_ms")
        working_set = _f(row, "working_set_kb")
        if mean_predict > budgets["predict_ms_mean_max"]:
            report["budget_failures"].append(f"{row.get('ai_name', 'unknown')} predict_mean_ms {mean_predict:.3f} > {budgets['predict_ms_mean_max']:.3f}")
        if mean_update > budgets["update_ms_mean_max"]:
            report["budget_failures"].append(f"{row.get('ai_name', 'unknown')} update_mean_ms {mean_update:.3f} > {budgets['update_ms_mean_max']:.3f}")
        if working_set > budgets["working_set_kb_max"]:
            report["budget_failures"].append(f"{row.get('ai_name', 'unknown')} working_set_kb {working_set:.1f} > {budgets['working_set_kb_max']:.1f}")
    return report


def write_performance_reports(symbols: list[str],
                              profile_name: str,
                              budgets: dict[str, float] | None = None) -> list[dict[str, object]]:
    out_dir = RESEARCH_DIR / safe_token(profile_name)
    ensure_dir(out_dir)
    reports = [build_symbol_performance_report(symbol, budgets) for symbol in symbols]
    summary_path = out_dir / "performance_reports.json"
    summary_path.write_text(json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8")
    return reports
