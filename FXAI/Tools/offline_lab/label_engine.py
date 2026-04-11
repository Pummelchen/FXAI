from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import libsql
from testlab.shared import resolve_execution_profile

from .common import OfflineLabError, commit_db, ensure_dir, now_unix, query_all, safe_token, sha256_text
from .exporter import resolve_dataset_rows
from .label_engine_config import load_config, validate_config_payload
from .label_engine_contracts import (
    LABEL_ENGINE_ARTIFACT_DIR,
    LABEL_ENGINE_CONFIG_PATH,
    LABEL_ENGINE_REPORT_PATH,
    LABEL_ENGINE_REPORT_VERSION,
    LABEL_ENGINE_RUNTIME_SUMMARY_PATH,
    LABEL_ENGINE_SCHEMA_VERSION,
    LABEL_ENGINE_STATUS_PATH,
    ensure_label_engine_dirs,
    isoformat_utc,
    json_dump,
)
from .label_engine_math import (
    BarRecord,
    CostSpec,
    build_horizon_specs,
    build_label_row,
    generate_baseline_candidates,
    infer_point_size,
    load_external_candidates,
    summarize_horizon_rows,
    top_reason_counts,
)


def _load_dataset_bars(conn: libsql.Connection, dataset_id: int) -> list[BarRecord]:
    rows = query_all(
        conn,
        """
        SELECT bar_time_unix, open, high, low, close, spread_points, tick_volume, real_volume
          FROM dataset_bars
         WHERE dataset_id = ?
         ORDER BY bar_time_unix ASC
        """,
        (int(dataset_id),),
    )
    return [
        BarRecord(
            index=index,
            time_unix=int(row["bar_time_unix"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            spread_points=float(row["spread_points"]),
            tick_volume=int(row["tick_volume"]),
            real_volume=int(row["real_volume"]),
        )
        for index, row in enumerate(rows)
    ]


def _build_effective_config(
    *,
    base_config: dict[str, Any],
    execution_profile: str | None = None,
    candidate_mode: str | None = None,
) -> dict[str, Any]:
    effective = json.loads(json.dumps(base_config))
    if execution_profile:
        effective["execution_profile"] = str(execution_profile)
    if candidate_mode:
        effective.setdefault("meta_labeling", {})["candidate_mode"] = str(candidate_mode).upper()
    return validate_config_payload(effective)


def _artifact_dir(dataset_key: str, profile_name: str) -> Path:
    token = safe_token(f"{dataset_key}__{profile_name or 'adhoc'}")
    return LABEL_ENGINE_ARTIFACT_DIR / token


def _cost_spec_for_bar(
    *,
    bar: BarRecord,
    execution_profile: dict[str, Any],
    config: dict[str, Any],
    execution_penalty_points: float = 0.0,
    news_penalty_points: float = 0.0,
) -> CostSpec:
    spread_multiplier = float(config.get("spread_multiplier", 1.0) or 1.0)
    commission_points = float(config.get("commission_points", 0.0) or 0.0)
    safety_margin_points = float(config.get("safety_margin_points", 0.25) or 0.25)
    return CostSpec(
        spread_cost_points=float(bar.spread_points) * spread_multiplier,
        slippage_points=float(execution_profile.get("slippage_points", 0.0) or 0.0),
        fill_penalty_points=float(execution_profile.get("fill_penalty_points", 0.0) or 0.0),
        commission_points=commission_points,
        safety_margin_points=safety_margin_points,
        execution_penalty_points=float(execution_penalty_points or 0.0),
        news_penalty_points=float(news_penalty_points or 0.0),
    )


def _write_ndjson(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _upsert_artifact_record(
    conn: libsql.Connection,
    *,
    dataset_id: int,
    profile_name: str,
    artifact_scope: str,
    artifact_dir: Path,
    bundle_path: Path,
    labels_path: Path,
    meta_labels_path: Path,
    summary_path: Path,
    config_sha256: str,
    summary_payload: dict[str, Any],
    label_version: int,
) -> None:
    conn.execute(
        """
        INSERT INTO label_engine_artifacts(
            dataset_id, profile_name, artifact_scope, artifact_dir, bundle_path, labels_path, meta_labels_path,
            summary_path, config_sha256, label_version, summary_json, created_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_id, profile_name, artifact_scope) DO UPDATE SET
            artifact_dir=excluded.artifact_dir,
            bundle_path=excluded.bundle_path,
            labels_path=excluded.labels_path,
            meta_labels_path=excluded.meta_labels_path,
            summary_path=excluded.summary_path,
            config_sha256=excluded.config_sha256,
            label_version=excluded.label_version,
            summary_json=excluded.summary_json,
            created_at=excluded.created_at
        """,
        (
            int(dataset_id),
            str(profile_name),
            str(artifact_scope),
            str(artifact_dir),
            str(bundle_path),
            str(labels_path),
            str(meta_labels_path),
            str(summary_path),
            str(config_sha256),
            int(label_version),
            json.dumps(summary_payload, sort_keys=True),
            now_unix(),
        ),
    )


def build_label_engine_bundle(
    conn: libsql.Connection,
    *,
    dataset: dict[str, Any],
    profile_name: str = "",
    config: dict[str, Any] | None = None,
    candidate_path: Path | None = None,
) -> dict[str, Any]:
    ensure_label_engine_dirs()
    config_payload = config or load_config()
    label_version = int(config_payload.get("label_version", 1) or 1)
    execution_profile_name = str(config_payload.get("execution_profile", "default") or "default")
    execution_profile = resolve_execution_profile(execution_profile_name)
    dataset_id = int(dataset["id"])
    dataset_key = str(dataset["dataset_key"])
    symbol = str(dataset["symbol"])
    bars = _load_dataset_bars(conn, dataset_id)
    if not bars:
        raise OfflineLabError(f"Dataset {dataset_key} has no dataset_bars rows")

    horizons = build_horizon_specs(config_payload)
    max_horizon_bars = max(item.bars for item in horizons)
    sample_start_bars = int(config_payload.get("sample_start_bars", 0) or 0)
    stride = int(config_payload.get("overlap_stride_bars", 1) or 1)
    if len(bars) <= sample_start_bars + max_horizon_bars:
        raise OfflineLabError(f"Dataset {dataset_key} does not have enough bars for the configured horizons")

    point_size = infer_point_size(symbol, bars[-1].close, config_payload)
    artifact_dir = _artifact_dir(dataset_key, profile_name)
    ensure_dir(artifact_dir)
    labels_path = artifact_dir / "labels.ndjson"
    meta_labels_path = artifact_dir / "meta_labels.ndjson"
    summary_path = artifact_dir / "summary.json"
    bundle_path = artifact_dir / "bundle.json"
    config_snapshot_path = artifact_dir / "config_snapshot.json"

    time_to_index = {bar.time_unix: bar.index for bar in bars}
    label_rows: list[dict[str, Any]] = []
    label_index: dict[tuple[int, str, str], dict[str, Any]] = {}
    for sample_index in range(sample_start_bars, len(bars) - max_horizon_bars, stride):
        bar = bars[sample_index]
        for horizon in horizons:
            future_bars = bars[sample_index + 1: sample_index + horizon.bars + 1]
            if len(future_bars) != horizon.bars:
                continue
            for side in ("LONG", "SHORT"):
                row = build_label_row(
                    dataset_key=dataset_key,
                    symbol=symbol,
                    point_size=point_size,
                    bar=bar,
                    future_bars=future_bars,
                    side=side,
                    horizon=horizon,
                    cost_spec=_cost_spec_for_bar(bar=bar, execution_profile=execution_profile, config=config_payload),
                    config=config_payload,
                    label_version=label_version,
                )
                label_rows.append(row)
                label_index[(sample_index, horizon.horizon_id, side)] = row

    meta_rows: list[dict[str, Any]] = []
    meta_cfg = dict(config_payload.get("meta_labeling", {}))
    meta_enabled = bool(meta_cfg.get("enabled", True))
    candidate_mode = str(meta_cfg.get("candidate_mode", "BASELINE_MOMENTUM") or "BASELINE_MOMENTUM")
    candidates = []
    min_strength = float(meta_cfg.get("min_raw_signal_strength", 0.15) or 0.15)
    if meta_enabled:
        if candidate_path:
            candidates = load_external_candidates(dataset_key=dataset_key, path=candidate_path, time_to_index=time_to_index)
            candidate_mode = "EXTERNAL_FILE"
        elif candidate_mode == "EXTERNAL_FILE":
            raise OfflineLabError(
                "Label engine candidate_mode=EXTERNAL_FILE requires --candidate-path so meta-labels remain reproducible"
            )
        else:
            candidates = generate_baseline_candidates(
                dataset_key=dataset_key,
                bars=bars,
                point_size=point_size,
                config=config_payload,
            )

        for candidate in candidates:
            side = str(candidate.side)
            if side not in {"LONG", "SHORT"}:
                continue
            candidate_failures = ["SIGNAL_TOO_WEAK"] if abs(float(candidate.raw_score or 0.0)) < min_strength else []
            matching_horizons = [item for item in horizons if candidate.horizon_id in (None, "", item.horizon_id)]
            for horizon in matching_horizons:
                label_row = label_index.get((candidate.sample_index, horizon.horizon_id, side))
                if label_row is None:
                    continue
                meta_reason_codes = list(candidate_failures)
                if int(label_row.get("tradeability_label", 0) or 0) <= 0:
                    meta_reason_codes.extend(str(reason) for reason in label_row.get("reason_codes", []))
                if float(candidate.execution_penalty_points or 0.0) > 0.0:
                    meta_reason_codes.append("EXECUTION_STRESS_TOO_HIGH")
                if float(candidate.news_penalty_points or 0.0) > 0.0:
                    meta_reason_codes.append("NEWS_STRESS_CONFLICT")
                meta_rows.append(
                    {
                        "signal_id": candidate.signal_id,
                        "sample_id": str(label_row["sample_id"]),
                        "timestamp_unix": int(candidate.bar_time_unix),
                        "symbol": symbol,
                        "dataset_key": dataset_key,
                        "side": side,
                        "horizon_id": horizon.horizon_id,
                        "candidate_source": candidate.source,
                        "raw_score": round(float(candidate.raw_score or 0.0), 6),
                        "raw_signal_strength": round(abs(float(candidate.raw_score or 0.0)), 6),
                        "tradeability_label": int(label_row.get("tradeability_label", 0) or 0),
                        "meta_label_trade": int(not meta_reason_codes),
                        "reason_codes": meta_reason_codes,
                        "candidate_diagnostics": dict(candidate.diagnostics),
                        "label_version": label_version,
                    }
                )

    horizon_summaries: list[dict[str, Any]] = []
    for horizon in horizons:
        horizon_label_rows = [row for row in label_rows if row["horizon_id"] == horizon.horizon_id]
        horizon_meta_rows = [row for row in meta_rows if row["horizon_id"] == horizon.horizon_id]
        summary = summarize_horizon_rows(label_rows=horizon_label_rows, meta_rows=horizon_meta_rows)
        summary["horizon_id"] = horizon.horizon_id
        summary["bars"] = horizon.bars
        horizon_summaries.append(summary)

    top_reasons = top_reason_counts(label_rows + meta_rows)
    summary_metrics = {
        "label_row_count": len(label_rows),
        "meta_label_count": len(meta_rows),
        "candidate_count": len(candidates),
        "point_size": round(point_size, 10),
        "sample_stride_bars": stride,
        "long_tradeability_rate": round(
            sum(int(row.get("tradeability_label", 0) or 0) for row in label_rows if row["side"] == "LONG")
            / max(sum(1 for row in label_rows if row["side"] == "LONG"), 1),
            6,
        ),
        "short_tradeability_rate": round(
            sum(int(row.get("tradeability_label", 0) or 0) for row in label_rows if row["side"] == "SHORT")
            / max(sum(1 for row in label_rows if row["side"] == "SHORT"), 1),
            6,
        ),
        "meta_acceptance_rate": round(
            sum(int(row.get("meta_label_trade", 0) or 0) for row in meta_rows) / max(len(meta_rows), 1),
            6,
        ),
    }
    meta_summary = {
        "enabled": meta_enabled,
        "candidate_mode": candidate_mode,
        "candidate_count": len(candidates),
        "accepted_count": sum(int(row.get("meta_label_trade", 0) or 0) for row in meta_rows),
        "rejected_count": len(meta_rows) - sum(int(row.get("meta_label_trade", 0) or 0) for row in meta_rows),
        "min_raw_signal_strength": min_strength,
        "candidate_path": str(candidate_path) if candidate_path else "",
    }
    quality_flags = {
        "path_approximation_used": True,
        "execution_profile_used": execution_profile_name,
        "external_candidates_used": bool(candidate_path),
        "partial_cost_model": bool(float(config_payload.get("commission_points", 0.0) or 0.0) <= 0.0),
    }
    effective_config = json.loads(json.dumps(config_payload))
    effective_config.setdefault("meta_labeling", {})["candidate_mode"] = candidate_mode
    effective_config["point_size_resolved"] = point_size
    effective_config["dataset_key"] = dataset_key
    effective_config["symbol"] = symbol
    config_snapshot_path.write_text(json.dumps(effective_config, indent=2, sort_keys=True), encoding="utf-8")

    bundle = {
        "schema_version": LABEL_ENGINE_SCHEMA_VERSION,
        "report_version": LABEL_ENGINE_REPORT_VERSION,
        "generated_at": isoformat_utc(),
        "profile_name": profile_name,
        "dataset_id": dataset_id,
        "dataset_key": dataset_key,
        "symbol": symbol,
        "timeframe": str(dataset.get("timeframe", "M1")),
        "bar_count": len(bars),
        "point_size": point_size,
        "execution_profile": execution_profile_name,
        "label_version": label_version,
        "summary_metrics": summary_metrics,
        "meta_summary": meta_summary,
        "quality_flags": quality_flags,
        "top_reason_codes": top_reasons,
        "horizon_summaries": horizon_summaries,
        "artifact_paths": {
            "labels_ndjson": str(labels_path),
            "meta_labels_ndjson": str(meta_labels_path),
            "summary_json": str(summary_path),
            "bundle_json": str(bundle_path),
            "config_snapshot_json": str(config_snapshot_path),
        },
    }

    _write_ndjson(labels_path, label_rows)
    _write_ndjson(meta_labels_path, meta_rows)
    summary_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")

    config_sha256 = sha256_text(json.dumps(effective_config, sort_keys=True))
    _upsert_artifact_record(
        conn,
        dataset_id=dataset_id,
        profile_name=profile_name,
        artifact_scope="latest",
        artifact_dir=artifact_dir,
        bundle_path=bundle_path,
        labels_path=labels_path,
        meta_labels_path=meta_labels_path,
        summary_path=summary_path,
        config_sha256=config_sha256,
        summary_payload=bundle,
        label_version=label_version,
    )
    commit_db(conn)
    return bundle


def build_label_engine_report(conn: libsql.Connection, *, profile_name: str = "", limit: int = 12) -> dict[str, Any]:
    ensure_label_engine_dirs()
    rows = query_all(
        conn,
        """
        SELECT lea.dataset_id, lea.profile_name, lea.artifact_scope, lea.artifact_dir, lea.bundle_path,
               lea.labels_path, lea.meta_labels_path, lea.summary_path, lea.config_sha256, lea.label_version,
               lea.summary_json, lea.created_at, d.dataset_key, d.symbol, d.timeframe, d.start_unix, d.end_unix, d.bars
          FROM label_engine_artifacts lea
          JOIN datasets d ON d.id = lea.dataset_id
         WHERE (? = '' OR lea.profile_name = ?)
         ORDER BY lea.created_at DESC, lea.dataset_id DESC
         LIMIT ?
        """,
        (profile_name, profile_name, int(limit)),
    )
    builds: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(str(row.get("summary_json", "{}") or "{}"))
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("dataset_id", int(row["dataset_id"]))
        payload.setdefault("dataset_key", str(row["dataset_key"]))
        payload.setdefault("profile_name", str(row["profile_name"] or ""))
        payload.setdefault("symbol", str(row["symbol"]))
        payload.setdefault("timeframe", str(row["timeframe"]))
        payload.setdefault("bar_count", int(row["bars"]))
        payload.setdefault("created_at_unix", int(row["created_at"]))
        payload.setdefault(
            "artifact_paths",
            {
                "bundle_json": str(row["bundle_path"]),
                "labels_ndjson": str(row["labels_path"]),
                "meta_labels_ndjson": str(row["meta_labels_path"]),
                "summary_json": str(row["summary_path"]),
            },
        )
        builds.append(payload)

    latest = builds[0] if builds else None
    report = {
        "schema_version": LABEL_ENGINE_REPORT_VERSION,
        "generated_at": isoformat_utc(),
        "profile_name": profile_name,
        "artifact_count": len(builds),
        "latest_dataset_key": str(latest.get("dataset_key", "")) if isinstance(latest, dict) else "",
        "builds": builds,
    }
    json_dump(LABEL_ENGINE_REPORT_PATH, report)
    runtime_summary = latest if isinstance(latest, dict) else {"generated_at": isoformat_utc(), "artifact_count": 0}
    json_dump(LABEL_ENGINE_RUNTIME_SUMMARY_PATH, runtime_summary)
    json_dump(
        LABEL_ENGINE_STATUS_PATH,
        {
            "generated_at": isoformat_utc(),
            "profile_name": profile_name,
            "artifact_count": len(builds),
            "latest_dataset_key": str(latest.get("dataset_key", "")) if isinstance(latest, dict) else "",
            "artifacts": {
                "report_json": str(LABEL_ENGINE_REPORT_PATH),
                "runtime_summary_json": str(LABEL_ENGINE_RUNTIME_SUMMARY_PATH),
            },
        },
    )
    return report


def validate_label_engine_config() -> dict[str, Any]:
    ensure_label_engine_dirs()
    payload = load_config(LABEL_ENGINE_CONFIG_PATH)
    return {
        "ok": True,
        "config_path": str(LABEL_ENGINE_CONFIG_PATH),
        "config": payload,
        "artifact_dir": str(LABEL_ENGINE_ARTIFACT_DIR),
        "report_path": str(LABEL_ENGINE_REPORT_PATH),
    }


def build_label_engine_artifacts(
    conn: libsql.Connection,
    *,
    args,
) -> dict[str, Any]:
    base_config = load_config()
    effective_config = _build_effective_config(
        base_config=base_config,
        execution_profile=getattr(args, "execution_profile", None),
        candidate_mode=getattr(args, "candidate_mode", None),
    )
    group_key = str(getattr(args, "group_key", "") or "")
    datasets = resolve_dataset_rows(conn, args, False, group_key)
    if getattr(args, "limit_datasets", 0):
        datasets = datasets[: int(getattr(args, "limit_datasets", 0))]
    if not datasets:
        raise OfflineLabError("No datasets matched the label-engine build request")
    candidate_path_text = str(getattr(args, "candidate_path", "") or "").strip()
    candidate_path = Path(candidate_path_text) if candidate_path_text else None
    bundles = [
        build_label_engine_bundle(
            conn,
            dataset=dataset,
            profile_name=str(getattr(args, "profile", "") or ""),
            config=effective_config,
            candidate_path=candidate_path,
        )
        for dataset in datasets
    ]
    report = build_label_engine_report(conn, profile_name=str(getattr(args, "profile", "") or ""))
    return {
        "status": "ok",
        "bundle_count": len(bundles),
        "datasets": [bundle["dataset_key"] for bundle in bundles],
        "report_path": str(LABEL_ENGINE_REPORT_PATH),
        "report": report,
    }
