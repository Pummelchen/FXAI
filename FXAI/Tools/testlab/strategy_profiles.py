from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any


STRATEGY_PROFILE_SCHEMA_VERSION = 1
STRATEGY_PROFILE_CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "OfflineLab" / "Profiles" / "strategy_profiles.json"
)

_PROFILE_SECTIONS = ("risk", "execution", "features", "model", "audit", "mt5_inputs", "audit_inputs")
_EXECUTION_PROFILE_ENUM = {
    "default": 0,
    "tight-fx": 1,
    "prime-ecn": 2,
    "retail-fx": 3,
    "stress": 4,
}
_POSITION_SIZING_ENUM = {
    "fixed_lot": 0,
    "fixed": 0,
    "conviction": 1,
    "vol_target": 2,
    "volatility_target": 2,
}
_KNOWN_OVERRIDE_MAP: dict[str, tuple[str, str]] = {
    "bars": ("audit", "bars"),
    "horizon": ("features", "horizon"),
    "m1sync_bars": ("features", "m1sync_bars"),
    "normalization": ("features", "normalization"),
    "sequence_bars": ("features", "sequence_bars"),
    "schema_id": ("features", "schema_id"),
    "feature_mask": ("features", "feature_mask"),
    "execution_profile": ("execution", "execution_profile"),
    "commission_per_lot_side": ("execution", "commission_per_lot_side"),
    "cost_buffer_points": ("execution", "cost_buffer_points"),
    "slippage_points": ("execution", "slippage_points"),
    "fill_penalty_points": ("execution", "fill_penalty_points"),
    "wf_train_bars": ("audit", "wf_train_bars"),
    "wf_test_bars": ("audit", "wf_test_bars"),
    "wf_purge_bars": ("audit", "wf_purge_bars"),
    "wf_embargo_bars": ("audit", "wf_embargo_bars"),
    "wf_folds": ("audit", "wf_folds"),
    "seed": ("audit", "seed"),
    "window_start_unix": ("audit", "window_start_unix"),
    "window_end_unix": ("audit", "window_end_unix"),
    "scenario_list": ("audit", "scenario_list"),
    "context_symbols": ("features", "context_symbols"),
}
_KNOWN_EA_INPUTS: tuple[tuple[str, tuple[str, str] | None, Any], ...] = (
    ("AI_Type", None, 0),
    ("AI_Ensemble", ("model", "ensemble_enabled"), False),
    ("Ensemble_AgreePct", ("model", "ensemble_agree_pct"), 70.0),
    ("Ensemble_ExplorePct", ("model", "ensemble_explore_pct"), 12.0),
    ("Ensemble_ShadowEveryBars", ("model", "ensemble_shadow_every_bars"), 3),
    ("Ensemble_ShadowSamples", ("model", "ensemble_shadow_samples"), 20),
    ("Ensemble_ShadowEpochs", ("model", "ensemble_shadow_epochs"), 1),
    ("AI_BuyThreshold", ("model", "buy_threshold"), 0.60),
    ("AI_SellThreshold", ("model", "sell_threshold"), 0.40),
    ("AI_M1SyncBars", ("features", "m1sync_bars"), 3),
    ("AI_Warmup", ("model", "warmup_enabled"), False),
    ("AI_WarmupSamples", ("model", "warmup_samples"), 10000),
    ("AI_WarmupLoops", ("model", "warmup_loops"), 100),
    ("AI_WarmupFolds", ("model", "warmup_folds"), 3),
    ("AI_WarmupSeed", ("model", "warmup_seed"), 42),
    ("AI_WarmupMinTrades", ("model", "warmup_min_trades"), 120),
    ("TP_USD", ("risk", "tp_usd"), 100.0),
    ("SL_USD", ("risk", "sl_usd"), 5.0),
    ("Lot", ("risk", "lot"), 0.01),
    ("AI_PositionSizing", ("risk", "position_sizing"), 1),
    ("RiskPerTradePct", ("risk", "risk_per_trade_pct"), 0.35),
    ("RiskTargetMovePoints", ("risk", "risk_target_move_points"), 12.0),
    ("MaxPortfolioExposureLots", ("risk", "max_portfolio_exposure_lots"), 0.30),
    ("MaxCorrelatedExposureLots", ("risk", "max_correlated_exposure_lots"), 0.20),
    ("MaxDirectionalClusterLots", ("risk", "max_directional_cluster_lots"), 0.18),
    ("RiskMinConfidence", ("risk", "risk_min_confidence"), 0.52),
    ("RiskMinReliability", ("risk", "risk_min_reliability"), 0.48),
    ("RiskMaxPathRisk", ("risk", "risk_max_path_risk"), 0.72),
    ("RiskMaxFillRisk", ("risk", "risk_max_fill_risk"), 0.68),
    ("RiskMinTradeGate", ("risk", "risk_min_trade_gate"), 0.52),
    ("RiskMinHierarchyScore", ("risk", "risk_min_hierarchy_score"), 0.46),
    ("RiskMinHierarchyConsistency", ("risk", "risk_min_hierarchy_consistency"), 0.40),
    ("RiskMinHierarchyTradability", ("risk", "risk_min_hierarchy_tradability"), 0.38),
    ("RiskMinHierarchyExecution", ("risk", "risk_min_hierarchy_execution"), 0.34),
    ("RiskMinMacroStateQuality", ("risk", "risk_min_macro_state_quality"), 0.24),
    ("RiskMaxPortfolioPressure", ("risk", "risk_max_portfolio_pressure"), 0.78),
    ("RiskKillTradeGate", ("risk", "risk_kill_trade_gate"), 0.24),
    ("RiskKillPathRisk", ("risk", "risk_kill_path_risk"), 0.92),
    ("RiskKillFillRisk", ("risk", "risk_kill_fill_risk"), 0.90),
    ("TradeMagic", ("risk", "trade_magic"), 6206001),
    ("MaxDD", ("risk", "max_dd"), 50.0),
    ("TradeKiller", ("risk", "trade_killer"), 0),
    ("TrailEnabled", ("risk", "trail_enabled"), True),
    ("TrailStartUSD", ("risk", "trail_start_usd"), 5.0),
    ("TrailGivebackPct", ("risk", "trail_giveback_pct"), 30.0),
    ("TrailTPBreathUSD", ("risk", "trail_tp_breath_usd"), 5.0),
    ("AI_Window", ("model", "window"), 60),
    ("AI_Epochs", ("model", "epochs"), 8),
    ("AI_LearningRate", ("model", "learning_rate"), 0.01),
    ("AI_L2", ("model", "l2"), 0.010),
    ("PredictionTargetMinutes", ("features", "horizon"), 5),
    ("AI_MultiHorizon", ("model", "multi_horizon"), True),
    ("AI_Horizons", ("model", "horizons"), [3, 5, 8, 13]),
    ("AI_HorizonPenaltyPerMinute", ("model", "horizon_penalty_per_minute"), 0.0015),
    ("AI_CommissionPerLotSide", ("execution", "commission_per_lot_side"), 0.0),
    ("AI_ExecutionProfile", ("execution", "execution_profile"), "default"),
    ("AI_CostBufferPoints", ("execution", "cost_buffer_points"), 2.0),
    ("AI_ExecutionSlippageOverride", ("execution", "slippage_points"), -1.0),
    ("AI_ExecutionFillPenaltyOverride", ("execution", "fill_penalty_points"), -1.0),
    ("AI_EVThresholdPoints", ("model", "ev_threshold_points"), 0.30),
    ("AI_EVLookbackSamples", ("model", "ev_lookback_samples"), 80),
    ("AI_OnlineSamples", ("model", "online_samples"), 60),
    ("AI_OnlineEpochs", ("model", "online_epochs"), 1),
    ("AI_DebugFlow", ("model", "debug_flow"), False),
    ("AI_ComplianceHarness", ("model", "compliance_harness"), False),
    ("AI_FeatureNormalization", ("features", "normalization"), 0),
    ("SessionFilterEnabled", ("risk", "session_filter_enabled"), False),
    ("SessionMinAfterOpenMinutes", ("risk", "session_min_after_open_minutes"), 60),
    ("SessionMinBeforeCloseMinutes", ("risk", "session_min_before_close_minutes"), 60),
    ("AI_ContextSymbols", ("features", "context_symbols"), ["EURUSD", "USDJPY", "AUDUSD", "EURAUD", "EURGBP", "GBPUSD"]),
)
_KNOWN_AUDIT_FIELDS: tuple[tuple[str, tuple[str, str] | None, Any], ...] = (
    ("bars", ("audit", "bars"), 20000),
    ("horizon", ("features", "horizon"), 5),
    ("m1sync_bars", ("features", "m1sync_bars"), 3),
    ("normalization", ("features", "normalization"), 0),
    ("sequence_bars", ("features", "sequence_bars"), 0),
    ("schema_id", ("features", "schema_id"), 0),
    ("feature_mask", ("features", "feature_mask"), 0),
    ("commission_per_lot_side", ("execution", "commission_per_lot_side"), 0.0),
    ("cost_buffer_points", ("execution", "cost_buffer_points"), 2.0),
    ("slippage_points", ("execution", "slippage_points"), 0.0),
    ("fill_penalty_points", ("execution", "fill_penalty_points"), 0.0),
    ("wf_train_bars", ("audit", "wf_train_bars"), 256),
    ("wf_test_bars", ("audit", "wf_test_bars"), 64),
    ("wf_purge_bars", ("audit", "wf_purge_bars"), 32),
    ("wf_embargo_bars", ("audit", "wf_embargo_bars"), 24),
    ("wf_folds", ("audit", "wf_folds"), 6),
    ("window_start_unix", ("audit", "window_start_unix"), 0),
    ("window_end_unix", ("audit", "window_end_unix"), 0),
    ("seed", ("audit", "seed"), 42),
    ("scenario_list", ("audit", "scenario_list"), []),
    ("execution_profile", ("execution", "execution_profile"), "default"),
)


def _sha256_path(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deep_merge(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = {key: copy.deepcopy(value) for key, value in base.items()}
        for key, value in overlay.items():
            merged[key] = _deep_merge(merged.get(key), value) if key in merged else copy.deepcopy(value)
        return merged
    return copy.deepcopy(overlay)


def _default_sections() -> dict[str, dict[str, Any]]:
    return {section: {} for section in _PROFILE_SECTIONS}


def _normalize_token(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    chars: list[str] = []
    for ch in raw:
        if ch.isalnum():
            chars.append(ch)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-")


def _portable_path(value: str | Path) -> str:
    raw = str(value or "")
    if not raw:
        return raw
    candidate = Path(raw)
    project_root = STRATEGY_PROFILE_CATALOG_PATH.parents[3]
    try:
        relative = candidate.resolve().relative_to(project_root.resolve())
    except Exception:
        return str(candidate)
    return f"<FXAI_ROOT>/{relative.as_posix()}"


def strategy_profile_catalog_path() -> Path:
    return STRATEGY_PROFILE_CATALOG_PATH


def load_strategy_profile_catalog(path: Path | None = None) -> dict[str, Any]:
    catalog_path = path or STRATEGY_PROFILE_CATALOG_PATH
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != STRATEGY_PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"Strategy profile schema mismatch: expected {STRATEGY_PROFILE_SCHEMA_VERSION}, got {schema_version}"
        )
    if not isinstance(payload.get("profiles"), dict):
        raise ValueError("Strategy profile catalog is missing the profiles map")
    if not isinstance(payload.get("defaults"), dict):
        raise ValueError("Strategy profile catalog is missing defaults")
    return payload


def _resolve_profile_id(catalog: dict[str, Any], raw_id: str | None, *, default_id: str) -> str:
    aliases = dict(catalog.get("aliases", {}))
    requested = (raw_id or "").strip() or default_id
    requested = aliases.get(requested, requested)
    profiles = catalog.get("profiles", {})
    if requested not in profiles:
        raise ValueError(f"Unknown strategy profile layer: {requested}")
    return requested


def _collect_profile_payload(
    catalog: dict[str, Any],
    profile_id: str,
    _seen: set[str] | None = None,
) -> dict[str, Any]:
    seen = _seen or set()
    if profile_id in seen:
        raise ValueError(f"Strategy profile inheritance cycle detected at {profile_id}")
    seen.add(profile_id)
    profiles = catalog.get("profiles", {})
    payload = profiles.get(profile_id)
    if not isinstance(payload, dict):
        raise ValueError(f"Strategy profile not found: {profile_id}")
    merged = _default_sections()
    metadata: dict[str, Any] = {}
    for parent in list(payload.get("extends", []) or []):
        parent_payload = _collect_profile_payload(catalog, str(parent), seen)
        for section in _PROFILE_SECTIONS:
            merged[section] = _deep_merge(merged[section], parent_payload.get(section, {}))
        metadata = _deep_merge(metadata, parent_payload.get("metadata", {}))
    for section in _PROFILE_SECTIONS:
        merged[section] = _deep_merge(merged[section], payload.get(section, {}))
    metadata = _deep_merge(metadata, {
        "profile_id": profile_id,
        "profile_version": int(payload.get("profile_version", 1) or 1),
        "description": str(payload.get("description", "") or ""),
        "compatibility": copy.deepcopy(payload.get("compatibility", {})),
    })
    merged["metadata"] = metadata
    seen.remove(profile_id)
    return merged


def profile_overrides_from_parameters(parameters: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return _default_sections()
    overrides = _default_sections()
    for section in _PROFILE_SECTIONS:
        section_payload = parameters.get(section)
        if isinstance(section_payload, dict):
            overrides[section] = _deep_merge(overrides[section], section_payload)
    for key, value in parameters.items():
        if value is None:
            continue
        target = _KNOWN_OVERRIDE_MAP.get(str(key))
        if target is None:
            continue
        section, field = target
        overrides[section][field] = copy.deepcopy(value)
    return overrides


def _resolve_symbol_layers(catalog: dict[str, Any], symbol: str) -> list[str]:
    profiles = catalog.get("profiles", {})
    layers: list[str] = []
    if "symbol/default" in profiles:
        layers.append("symbol/default")
    symbol_id = f"symbol/{(symbol or '').upper()}"
    if symbol_id in profiles and symbol_id not in layers:
        layers.append(symbol_id)
    return layers


def _resolve_broker_layers(catalog: dict[str, Any], *, broker_profile: str | None, server: str | None) -> tuple[list[str], str]:
    profiles = catalog.get("profiles", {})
    requested = _normalize_token(broker_profile or server) or "default"
    layers: list[str] = []
    if "broker/default" in profiles:
        layers.append("broker/default")
    broker_id = f"broker/{requested}"
    if broker_id in profiles and broker_id not in layers:
        layers.append(broker_id)
    return layers, requested


def _resolve_runtime_layers(catalog: dict[str, Any], runtime_mode: str) -> list[str]:
    defaults = dict(catalog.get("defaults", {}))
    runtime_profiles = dict(defaults.get("runtime_profiles", {}))
    profiles = catalog.get("profiles", {})
    requested = str(runtime_mode or "research").strip().lower() or "research"
    runtime_id = str(runtime_profiles.get(requested, f"runtime/{requested}"))
    layers: list[str] = []
    if "runtime/default" in profiles:
        layers.append("runtime/default")
    if runtime_id in profiles and runtime_id not in layers:
        layers.append(runtime_id)
    return layers


def _coerce_position_sizing(value: Any) -> int:
    if isinstance(value, str):
        token = _normalize_token(value).replace("-", "_")
        if token in _POSITION_SIZING_ENUM:
            return int(_POSITION_SIZING_ENUM[token])
    try:
        return int(value)
    except Exception:
        return 1


def _coerce_execution_profile(value: Any) -> int:
    token = _normalize_token(str(value or "default")) or "default"
    return int(_EXECUTION_PROFILE_ENUM.get(token, 0))


def _coerce_horizons(value: Any, *, fallback: int) -> list[int]:
    if isinstance(value, str):
        text = value.strip().replace("{", "").replace("}", "")
        values = [segment.strip() for segment in text.replace(";", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    elif value is None:
        values = [fallback]
    else:
        values = [value]
    horizons: list[int] = []
    for item in values:
        try:
            candidate = int(item)
        except Exception:
            continue
        if candidate > 0 and candidate not in horizons:
            horizons.append(candidate)
    if fallback > 0 and fallback not in horizons:
        horizons.append(fallback)
    horizons.sort()
    return horizons or [fallback]


def _coerce_context_symbols(value: Any, *, symbol: str) -> list[str]:
    if isinstance(value, str):
        text = value.strip().replace("{", "").replace("}", "")
        items = [segment.strip().upper() for segment in text.replace(";", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        items = [str(segment).strip().upper() for segment in value]
    else:
        items = []
    normalized: list[str] = []
    anchor = (symbol or "").upper()
    if anchor:
        normalized.append(anchor)
    for item in items:
        if item and item not in normalized:
            normalized.append(item)
    return normalized


def _section_value(sections: dict[str, Any], location: tuple[str, str] | None, fallback: Any) -> Any:
    if location is None:
        return copy.deepcopy(fallback)
    section_name, field = location
    section_payload = sections.get(section_name, {})
    if isinstance(section_payload, dict) and field in section_payload:
        return copy.deepcopy(section_payload[field])
    return copy.deepcopy(fallback)


def compile_strategy_profile(
    *,
    strategy_profile: str | None = None,
    symbol: str = "EURUSD",
    broker_profile: str | None = None,
    server: str | None = None,
    runtime_mode: str = "research",
    overrides: dict[str, Any] | None = None,
    plugin_name: str = "",
    ai_id: int | None = None,
    catalog: dict[str, Any] | None = None,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    catalog_payload = copy.deepcopy(catalog) if catalog is not None else load_strategy_profile_catalog(catalog_path)
    defaults = dict(catalog_payload.get("defaults", {}))
    base_id = _resolve_profile_id(
        catalog_payload,
        strategy_profile,
        default_id=str(defaults.get("strategy_profile", "strategy/default")),
    )
    layers = [base_id]
    layers.extend(_resolve_symbol_layers(catalog_payload, symbol))
    broker_layers, broker_token = _resolve_broker_layers(
        catalog_payload,
        broker_profile=broker_profile,
        server=server,
    )
    layers.extend(broker_layers)
    layers.extend(_resolve_runtime_layers(catalog_payload, runtime_mode))

    effective_sections = _default_sections()
    lineage: list[dict[str, Any]] = []
    for layer_id in layers:
        payload = _collect_profile_payload(catalog_payload, layer_id)
        for section in _PROFILE_SECTIONS:
            effective_sections[section] = _deep_merge(effective_sections[section], payload.get(section, {}))
        lineage.append(payload["metadata"])

    override_sections = profile_overrides_from_parameters(overrides)
    for section in _PROFILE_SECTIONS:
        effective_sections[section] = _deep_merge(effective_sections[section], override_sections.get(section, {}))

    horizon = int(_section_value(effective_sections, ("features", "horizon"), 5) or 5)
    effective_sections["model"]["horizons"] = _coerce_horizons(
        effective_sections["model"].get("horizons"),
        fallback=horizon,
    )
    effective_sections["features"]["context_symbols"] = _coerce_context_symbols(
        effective_sections["features"].get("context_symbols"),
        symbol=symbol,
    )

    ea_inputs: dict[str, Any] = {}
    for input_name, location, fallback in _KNOWN_EA_INPUTS:
        value = _section_value(effective_sections, location, fallback)
        if input_name == "AI_Type" and ai_id is not None:
            value = int(ai_id)
        elif input_name == "AI_PositionSizing":
            value = _coerce_position_sizing(value)
        elif input_name == "AI_ExecutionProfile":
            value = _coerce_execution_profile(value)
        elif input_name == "AI_Horizons":
            value = effective_sections["model"].get("horizons", [horizon])
        elif input_name == "AI_ContextSymbols":
            value = effective_sections["features"].get("context_symbols", [])
        ea_inputs[input_name] = value
    ea_inputs.update(copy.deepcopy(effective_sections.get("mt5_inputs", {})))
    if ai_id is not None:
        ea_inputs["AI_Type"] = int(ai_id)

    audit_values: dict[str, Any] = {
        "plugin_name": str(plugin_name or ""),
        "plugin_id": int(ai_id) if ai_id is not None else 28,
        "symbol": str(symbol or "EURUSD"),
    }
    for field_name, location, fallback in _KNOWN_AUDIT_FIELDS:
        audit_values[field_name] = _section_value(effective_sections, location, fallback)
    audit_values.update(copy.deepcopy(effective_sections.get("audit_inputs", {})))
    audit_values["execution_profile"] = str(audit_values.get("execution_profile", "default") or "default")

    compatibility = {}
    for item in lineage:
        compatibility = _deep_merge(compatibility, item.get("compatibility", {}))

    catalog_origin = catalog_path or STRATEGY_PROFILE_CATALOG_PATH
    return {
        "schema_version": STRATEGY_PROFILE_SCHEMA_VERSION,
        "catalog_version": str(catalog_payload.get("catalog_version", "") or ""),
        "catalog_path": str(catalog_origin),
        "catalog_sha256": _sha256_path(catalog_origin),
        "strategy_profile_id": base_id,
        "strategy_profile_version": int(lineage[0].get("profile_version", 1) if lineage else 1),
        "layer_ids": layers,
        "layers": {
            "strategy": base_id,
            "symbol": [layer for layer in layers if layer.startswith("symbol/")],
            "broker": [layer for layer in layers if layer.startswith("broker/")],
            "runtime": [layer for layer in layers if layer.startswith("runtime/")],
        },
        "context": {
            "symbol": str(symbol or "EURUSD"),
            "broker_profile": str(broker_profile or ""),
            "resolved_broker": broker_token,
            "server": str(server or ""),
            "runtime_mode": str(runtime_mode or "research"),
            "plugin_name": str(plugin_name or ""),
            "ai_id": (int(ai_id) if ai_id is not None else None),
        },
        "effective": effective_sections,
        "compiled": {
            "ea_inputs": ea_inputs,
            "audit_values": audit_values,
        },
        "compatibility": compatibility,
        "lineage": lineage,
    }


def mt5_input_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return "{" + ", ".join(items) + "}"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".") if "." in f"{value:.6f}" else f"{value:.6f}"
    return str(value)


def render_mt5_set(compiled_profile: dict[str, Any], *, line_defaults: dict[str, tuple[Any, Any, Any, str]] | None = None) -> str:
    defaults = dict(line_defaults or {})
    lines: list[str] = []
    for input_name, value in compiled_profile["compiled"]["ea_inputs"].items():
        min_spec, start_spec, stop_spec, optimize_spec = defaults.get(
            input_name,
            (value, 0, 0, "N"),
        )
        lines.append(
            f"{input_name}={mt5_input_value(value)}||{mt5_input_value(min_spec)}||"
            f"{mt5_input_value(start_spec)}||{mt5_input_value(stop_spec)}||{optimize_spec}"
        )
    return "\n".join(lines) + "\n"


def build_strategy_profile_manifest(
    compiled_profile: dict[str, Any],
    *,
    artifact_kind: str,
    artifact_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": STRATEGY_PROFILE_SCHEMA_VERSION,
        "generated_at": int(time.time()),
        "artifact_kind": str(artifact_kind),
        "artifact_path": _portable_path(artifact_path),
        "catalog_version": str(compiled_profile.get("catalog_version", "") or ""),
        "catalog_path": _portable_path(compiled_profile.get("catalog_path", "")),
        "catalog_sha256": str(compiled_profile.get("catalog_sha256", "") or ""),
        "strategy_profile_id": str(compiled_profile.get("strategy_profile_id", "") or ""),
        "strategy_profile_version": int(compiled_profile.get("strategy_profile_version", 0) or 0),
        "layer_ids": list(compiled_profile.get("layer_ids", [])),
        "layers": copy.deepcopy(compiled_profile.get("layers", {})),
        "context": copy.deepcopy(compiled_profile.get("context", {})),
        "compatibility": copy.deepcopy(compiled_profile.get("compatibility", {})),
        "effective": copy.deepcopy(compiled_profile.get("effective", {})),
        "compiled": copy.deepcopy(compiled_profile.get("compiled", {})),
        "metadata": copy.deepcopy(metadata or {}),
    }
