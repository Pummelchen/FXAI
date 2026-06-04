from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FX_M1_BARS_PER_TRADING_YEAR = 52 * 5 * 24 * 60
MAX_AUDIT_BARS = 5_000_000
SUPPORTED_WINDOW_MODES = {"rolling", "anchored"}


class WalkForwardPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class WalkForwardPolicy:
    mode: str
    train_bars: int
    test_bars: int
    purge_bars: int
    embargo_bars: int
    folds: int
    bars_per_year: int = FX_M1_BARS_PER_TRADING_YEAR
    train_years: float = 0.0
    test_years: float = 0.0
    purge_days: float = 0.0
    embargo_days: float = 0.0

    def minimum_bars(self) -> int:
        if self.folds <= 0:
            return self.train_bars + self.purge_bars + self.test_bars
        step = self.test_bars + self.embargo_bars
        if self.mode == "anchored":
            return self.train_bars + (self.folds - 1) * step + self.purge_bars + self.test_bars + self.embargo_bars
        return (self.folds - 1) * step + self.train_bars + self.purge_bars + self.test_bars + self.embargo_bars

    def as_manifest(self, total_bars: int | None = None) -> dict[str, Any]:
        total = int(total_bars or 0)
        return {
            "mode": self.mode,
            "train_bars": self.train_bars,
            "test_bars": self.test_bars,
            "purge_bars": self.purge_bars,
            "embargo_bars": self.embargo_bars,
            "folds": self.folds,
            "bars_per_year": self.bars_per_year,
            "train_years": self.train_years,
            "test_years": self.test_years,
            "purge_days": self.purge_days,
            "embargo_days": self.embargo_days,
            "minimum_required_bars": self.minimum_bars(),
            "total_bars": total,
            "window_plan": [window.as_manifest() for window in build_walkforward_windows(total, self)] if total > 0 else [],
        }


@dataclass(frozen=True)
class WalkForwardWindow:
    fold: int
    mode: str
    train_start_bar: int
    train_end_bar: int
    purge_start_bar: int
    purge_end_bar: int
    test_start_bar: int
    test_end_bar: int
    embargo_start_bar: int
    embargo_end_bar: int

    def as_manifest(self) -> dict[str, int | str]:
        return {
            "fold": self.fold,
            "mode": self.mode,
            "train_start_bar": self.train_start_bar,
            "train_end_bar": self.train_end_bar,
            "purge_start_bar": self.purge_start_bar,
            "purge_end_bar": self.purge_end_bar,
            "test_start_bar": self.test_start_bar,
            "test_end_bar": self.test_end_bar,
            "embargo_start_bar": self.embargo_start_bar,
            "embargo_end_bar": self.embargo_end_bar,
        }


def _float_attr(args: Any, name: str, default: float = 0.0) -> float:
    try:
        value = getattr(args, name, default)
        if value in (None, ""):
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int_attr(args: Any, name: str, default: int) -> int:
    try:
        value = getattr(args, name, default)
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def bars_from_years(years: float, bars_per_year: int = FX_M1_BARS_PER_TRADING_YEAR) -> int:
    years = max(0.0, float(years or 0.0))
    if years <= 0.0:
        return 0
    return max(1, int(math.ceil(years * max(1, int(bars_per_year)))))


def bars_from_days(days: float) -> int:
    days = max(0.0, float(days or 0.0))
    if days <= 0.0:
        return 0
    return max(1, int(math.ceil(days * 24 * 60)))


def resolve_walkforward_policy(args: Any) -> WalkForwardPolicy:
    mode = str(getattr(args, "wf_window_mode", "rolling") or "rolling").strip().lower()
    if mode not in SUPPORTED_WINDOW_MODES:
        raise WalkForwardPolicyError(f"unsupported walk-forward mode: {mode}")

    bars_per_year = max(1, _int_attr(args, "wf_bars_per_year", FX_M1_BARS_PER_TRADING_YEAR))
    train_years = max(0.0, _float_attr(args, "wf_train_years", 0.0))
    test_years = max(0.0, _float_attr(args, "wf_test_years", 0.0))
    purge_days = max(0.0, _float_attr(args, "wf_purge_days", 0.0))
    embargo_days = max(0.0, _float_attr(args, "wf_embargo_days", 0.0))

    train_bars = bars_from_years(train_years, bars_per_year) or max(1, _int_attr(args, "wf_train_bars", 256))
    test_bars = bars_from_years(test_years, bars_per_year) or max(1, _int_attr(args, "wf_test_bars", 64))
    purge_bars = bars_from_days(purge_days) or max(0, _int_attr(args, "wf_purge_bars", 32))
    embargo_bars = bars_from_days(embargo_days) or max(0, _int_attr(args, "wf_embargo_bars", 24))
    folds = max(1, min(_int_attr(args, "wf_folds", 6), 64))

    policy = WalkForwardPolicy(
        mode=mode,
        train_bars=train_bars,
        test_bars=test_bars,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        folds=folds,
        bars_per_year=bars_per_year,
        train_years=train_years,
        test_years=test_years,
        purge_days=purge_days,
        embargo_days=embargo_days,
    )
    if policy.minimum_bars() > MAX_AUDIT_BARS:
        raise WalkForwardPolicyError(
            f"walk-forward policy requires {policy.minimum_bars()} bars, above max audit cap {MAX_AUDIT_BARS}"
        )
    return policy


def build_walkforward_windows(total_bars: int, policy: WalkForwardPolicy) -> list[WalkForwardWindow]:
    total = max(0, int(total_bars or 0))
    if total <= 0:
        return []

    windows: list[WalkForwardWindow] = []
    step = policy.test_bars + policy.embargo_bars
    for fold in range(policy.folds):
        if policy.mode == "anchored":
            train_start = 0
            train_end = policy.train_bars + fold * step - 1
        else:
            train_start = fold * step
            train_end = train_start + policy.train_bars - 1

        purge_start = train_end + 1
        purge_end = purge_start + policy.purge_bars - 1
        test_start = train_end + policy.purge_bars + 1
        test_end = test_start + policy.test_bars - 1
        embargo_start = test_end + 1
        embargo_end = embargo_start + policy.embargo_bars - 1

        if train_start < 0 or train_end < train_start:
            continue
        if test_start <= train_end or test_end < test_start:
            continue
        if test_end >= total:
            break
        windows.append(
            WalkForwardWindow(
                fold=fold + 1,
                mode=policy.mode,
                train_start_bar=train_start,
                train_end_bar=train_end,
                purge_start_bar=purge_start,
                purge_end_bar=max(purge_start - 1, purge_end),
                test_start_bar=test_start,
                test_end_bar=test_end,
                embargo_start_bar=embargo_start,
                embargo_end_bar=min(max(embargo_start - 1, embargo_end), total - 1),
            )
        )
    return windows


def apply_walkforward_policy(args: Any) -> Any:
    policy = resolve_walkforward_policy(args)
    setattr(args, "wf_train_bars", policy.train_bars)
    setattr(args, "wf_test_bars", policy.test_bars)
    setattr(args, "wf_purge_bars", policy.purge_bars)
    setattr(args, "wf_embargo_bars", policy.embargo_bars)
    setattr(args, "wf_folds", policy.folds)
    setattr(args, "wf_window_mode", policy.mode)
    setattr(args, "wf_bars_per_year", policy.bars_per_year)
    setattr(args, "wf_required_bars", policy.minimum_bars())

    bars = _int_attr(args, "bars", 0)
    if policy.train_years > 0.0 or policy.test_years > 0.0:
        if bars <= 0 or bars < policy.minimum_bars():
            setattr(args, "bars", policy.minimum_bars())
    setattr(args, "walkforward_policy", policy)
    return args


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = sum(values) / len(values)
    denom = sum((idx - x_mean) ** 2 for idx in range(len(values)))
    if denom <= 0.0:
        return 0.0
    return sum((idx - x_mean) * (value - y_mean) for idx, value in enumerate(values)) / denom


def _load_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _sidecar_path(report_path: Path) -> Path:
    return report_path.with_suffix(".walkforward.json")


def _rows_from_diagnostics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_plugins = payload.get("plugins", [])
    if isinstance(raw_plugins, dict):
        plugin_items = []
        for name, info in raw_plugins.items():
            if isinstance(info, dict):
                enriched = dict(info)
                enriched.setdefault("aiName", name)
                plugin_items.append(enriched)
    elif isinstance(raw_plugins, list):
        plugin_items = [item for item in raw_plugins if isinstance(item, dict)]
    else:
        plugin_items = []

    for plugin in plugin_items:
        plugin_name = str(plugin.get("aiName", "") or plugin.get("ai_name", "") or plugin.get("plugin", "") or "unknown")
        ai_id = plugin.get("aiID", plugin.get("ai_id", ""))
        family = plugin.get("family", "")
        windows = plugin.get("windows", [])
        if not isinstance(windows, list):
            continue
        for index, window in enumerate(windows, start=1):
            if not isinstance(window, dict):
                continue
            fold = _safe_int(window.get("fold", window.get("index", index)), index)
            passed = window.get("passed", window.get("pass", None))
            overfit = window.get("overfit", None)
            rows.append(
                {
                    "ai_id": ai_id,
                    "ai_name": plugin_name,
                    "family": family,
                    "scenario": "market_walkforward",
                    "window_index": fold,
                    "wf_train_samples": window.get("trainSamples", window.get("train_samples", 0)),
                    "wf_test_samples": window.get("testSamples", window.get("test_samples", 0)),
                    "wf_train_score": window.get("trainScore", window.get("train_score", 0.0)),
                    "wf_test_score": window.get("testScore", window.get("test_score", 0.0)),
                    "wf_gap": window.get("gap", 0.0),
                    "wf_pass_rate": 1.0 if passed is True else 0.0 if passed is False else window.get("pass_rate", 0.0),
                    "wf_pbo": 1.0 if overfit is True else 0.0 if overfit is False else window.get("pbo", 0.0),
                    "wf_dsr": window.get("dsr", 0.0),
                }
            )
    return rows


def analyze_walkforward_rows(rows: list[dict[str, Any]], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    by_plugin: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("scenario", "")) != "market_walkforward":
            continue
        plugin = str(row.get("ai_name", "") or row.get("plugin", "") or "unknown")
        by_plugin.setdefault(plugin, []).append(row)

    policy = dict((manifest or {}).get("walkforward", {}))
    result: dict[str, Any] = {
        "policy": policy,
        "plugins": {},
    }

    for plugin, plugin_rows in sorted(by_plugin.items()):
        plugin_rows.sort(key=lambda row: _safe_int(row.get("window_index", row.get("fold", len(plugin_rows))), 0))
        test_scores = [_safe_float(row.get("wf_test_score")) for row in plugin_rows]
        train_scores = [_safe_float(row.get("wf_train_score")) for row in plugin_rows]
        pass_rates = [_safe_float(row.get("wf_pass_rate")) for row in plugin_rows]
        pbos = [_safe_float(row.get("wf_pbo")) for row in plugin_rows]
        dsrs = [_safe_float(row.get("wf_dsr")) for row in plugin_rows]
        slope = _linear_slope(test_scores)
        first_last_delta = (test_scores[-1] - test_scores[0]) if len(test_scores) >= 2 else 0.0
        average_test = sum(test_scores) / len(test_scores) if test_scores else 0.0
        worst_test = min(test_scores) if test_scores else 0.0
        degradation_flag = slope < -1.0 or first_last_delta < -6.0 or worst_test < 68.0
        result["plugins"][plugin] = {
            "window_count": len(plugin_rows),
            "average_train_score": sum(train_scores) / len(train_scores) if train_scores else 0.0,
            "average_test_score": average_test,
            "worst_test_score": worst_test,
            "test_score_slope": slope,
            "first_last_test_delta": first_last_delta,
            "average_pass_rate": sum(pass_rates) / len(pass_rates) if pass_rates else 0.0,
            "average_pbo": sum(pbos) / len(pbos) if pbos else 0.0,
            "average_dsr": sum(dsrs) / len(dsrs) if dsrs else 0.0,
            "degradation_flag": degradation_flag,
            "windows": [
                {
                    "index": index + 1,
                    "train_score": train_scores[index],
                    "test_score": test_scores[index],
                    "pass_rate": pass_rates[index],
                    "pbo": pbos[index],
                    "dsr": dsrs[index],
                    "raw": plugin_rows[index],
                }
                for index in range(len(plugin_rows))
            ],
        }
    return result


def analyze_walkforward_report(
    report_path: Path,
    manifest_path: Path | None = None,
    diagnostics_path: Path | None = None,
) -> dict[str, Any]:
    rows = _load_tsv_rows(report_path)
    manifest = None
    if manifest_path is not None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics = diagnostics_path if diagnostics_path is not None else _sidecar_path(report_path)
    if diagnostics.exists():
        diagnostic_rows = _rows_from_diagnostics(json.loads(diagnostics.read_text(encoding="utf-8")))
        if diagnostic_rows:
            rows = diagnostic_rows
    return analyze_walkforward_rows(rows, manifest)


def render_walkforward_analysis(payload: dict[str, Any]) -> str:
    lines = ["# Walk-Forward OOS Analysis", ""]
    policy = dict(payload.get("policy", {}))
    if policy:
        lines.append(
            "Policy: "
            f"mode={policy.get('mode', 'rolling')} "
            f"train_bars={policy.get('train_bars', 0)} "
            f"test_bars={policy.get('test_bars', 0)} "
            f"purge_bars={policy.get('purge_bars', 0)} "
            f"embargo_bars={policy.get('embargo_bars', 0)} "
            f"folds={policy.get('folds', 0)}"
        )
        if float(policy.get("train_years", 0.0) or 0.0) > 0.0 or float(policy.get("test_years", 0.0) or 0.0) > 0.0:
            lines.append(
                f"Years: train={float(policy.get('train_years', 0.0) or 0.0):.3f} "
                f"test={float(policy.get('test_years', 0.0) or 0.0):.3f} "
                f"bars_per_year={int(policy.get('bars_per_year', FX_M1_BARS_PER_TRADING_YEAR) or FX_M1_BARS_PER_TRADING_YEAR)}"
            )
        lines.append("")

    plugins = dict(payload.get("plugins", {}))
    if not plugins:
        lines.append("No `market_walkforward` rows were found.")
        return "\n".join(lines) + "\n"

    for plugin, info in sorted(plugins.items(), key=lambda item: float(item[1].get("average_test_score", 0.0)), reverse=True):
        status = "DEGRADING" if info.get("degradation_flag") else "stable"
        lines.append(
            f"## {plugin} | OOS {float(info.get('average_test_score', 0.0)):.2f} | "
            f"worst {float(info.get('worst_test_score', 0.0)):.2f} | {status}"
        )
        lines.append(
            f"slope={float(info.get('test_score_slope', 0.0)):.3f}, "
            f"first_last_delta={float(info.get('first_last_test_delta', 0.0)):.3f}, "
            f"pass_rate={float(info.get('average_pass_rate', 0.0)):.3f}, "
            f"PBO={float(info.get('average_pbo', 0.0)):.3f}, "
            f"DSR={float(info.get('average_dsr', 0.0)):.3f}"
        )
        for window in info.get("windows", []):
            lines.append(
                f"- window {int(window.get('index', 0))}: "
                f"train={float(window.get('train_score', 0.0)):.2f}, "
                f"test={float(window.get('test_score', 0.0)):.2f}, "
                f"pass={float(window.get('pass_rate', 0.0)):.2f}, "
                f"PBO={float(window.get('pbo', 0.0)):.2f}, "
                f"DSR={float(window.get('dsr', 0.0)):.2f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
