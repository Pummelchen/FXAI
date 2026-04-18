from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
import libsql

from .common import *
from .shadow_fleet import symbol_shadow_summary


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _mean(values: list[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(sum(values) / float(len(values)))


def _quantile(values: list[float], q: float, default: float = 0.0) -> float:
    if not values:
        return float(default)
    seq = sorted(float(v) for v in values)
    if len(seq) == 1:
        return seq[0]
    pos = _clamp(q, 0.0, 1.0) * float(len(seq) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return seq[lo]
    frac = pos - float(lo)
    return seq[lo] * (1.0 - frac) + seq[hi] * frac


def _norm_entropy(counts: list[int]) -> float:
    total = float(sum(max(int(v), 0) for v in counts))
    if total <= 1e-9:
        return 0.0
    probs = [float(v) / total for v in counts if int(v) > 0]
    if len(probs) <= 1:
        return 0.0
    ent = -sum(p * math.log(p) for p in probs if p > 0.0)
    return _clamp(ent / math.log(float(len(probs))), 0.0, 1.0)


def _session_bucket_from_hour(hour: int) -> str:
    h = int(hour) % 24
    if 7 <= h <= 12:
        return "london"
    if 13 <= h <= 20:
        return "newyork"
    return "asia"


def build_symbol_world_model(conn: libsql.Connection,
                             profile_name: str,
                             symbol: str,
                             dataset_limit: int = 4) -> dict:
    dataset_rows = query_all(
        conn,
        """
        SELECT id, dataset_key, months, bars, start_unix, end_unix
          FROM datasets
         WHERE symbol = ?
         ORDER BY end_unix DESC, created_at DESC
         LIMIT ?
        """,
        (symbol, dataset_limit),
    )
    if not dataset_rows:
        shadow = symbol_shadow_summary(conn, profile_name, symbol)
        return {
            "profile_name": profile_name,
            "symbol": symbol,
            "dataset_count": 0,
            "bar_count": 0,
            "sigma_scale": 1.0,
            "drift_bias": 0.0,
            "spread_scale": 1.0,
            "gap_prob": 0.0,
            "gap_scale": 0.0,
            "flip_prob": 0.0,
            "context_corr_bias": _clamp(
                0.35 * float(shadow.get("mean_portfolio_div", 0.0)) -
                0.30 * float(shadow.get("mean_portfolio_corr", 0.0)),
                -1.0,
                1.0,
            ),
            "liquidity_stress": _clamp(float(shadow.get("mean_portfolio_pressure", 0.0)), 0.0, 3.0),
            "macro_focus": _clamp(float(shadow.get("mean_policy_no_trade_prob", 0.0)), 0.0, 1.5),
            "session_edge_focus": 0.0,
            "trend_persistence": 0.5,
            "shock_memory": 0.0,
            "recovery_bias": 0.0,
            "spread_shock_prob": 0.0,
            "spread_shock_scale": 1.0,
            "shadow_summary": shadow,
            "datasets": [],
        }

    returns: list[float] = []
    spread_values: list[float] = []
    gap_values: list[float] = []
    edge_abs_returns: list[float] = []
    edge_spreads: list[float] = []
    non_edge_abs_returns: list[float] = []
    non_edge_spreads: list[float] = []
    session_abs_returns: dict[str, list[float]] = {"asia": [], "london": [], "newyork": []}
    session_spreads: dict[str, list[float]] = {"asia": [], "london": [], "newyork": []}
    bar_records: list[tuple[float, float, float]] = []
    same_sign = 0
    sign_pairs = 0
    shock_hits = 0
    shock_follow_same = 0
    shock_follow_reverse = 0
    dataset_payloads: list[dict] = []
    total_bars = 0

    for dataset in dataset_rows:
        bars = query_all(
            conn,
            """
            SELECT bar_time_unix, open, high, low, close, spread_points
              FROM dataset_bars
             WHERE dataset_id = ?
             ORDER BY bar_time_unix ASC
            """,
            (int(dataset["id"]),),
        )
        if len(bars) < 8:
            continue
        dataset_payloads.append({
            "dataset_key": str(dataset["dataset_key"]),
            "months": int(dataset["months"]),
            "bars": int(dataset["bars"]),
            "start_unix": int(dataset["start_unix"]),
            "end_unix": int(dataset["end_unix"]),
        })
        total_bars += len(bars)
        local_returns: list[float] = []
        local_spreads: list[float] = []
        for idx in range(1, len(bars)):
            prev = bars[idx - 1]
            cur = bars[idx]
            prev_close = float(prev["close"])
            cur_open = float(cur["open"])
            cur_close = float(cur["close"])
            if prev_close <= 0.0 or cur_open <= 0.0 or cur_close <= 0.0:
                continue
            ret = (cur_close - prev_close) / prev_close
            gap = (cur_open - prev_close) / prev_close
            spread = float(cur["spread_points"])
            returns.append(ret)
            local_returns.append(ret)
            gap_values.append(abs(gap))
            spread_values.append(spread)
            local_spreads.append(spread)

            hour = datetime.fromtimestamp(int(cur["bar_time_unix"]), tz=timezone.utc).hour
            edge_hour = (7 <= hour <= 9) or (15 <= hour <= 17) or hour in (22, 23, 0)
            edge_score = 1.0 if edge_hour else 0.0
            session_key = _session_bucket_from_hour(hour)
            bar_records.append((ret, spread, edge_score))
            session_abs_returns[session_key].append(abs(ret))
            session_spreads[session_key].append(spread)
            if edge_hour:
                edge_abs_returns.append(abs(ret))
                edge_spreads.append(spread)
            else:
                non_edge_abs_returns.append(abs(ret))
                non_edge_spreads.append(spread)

        for idx in range(1, len(local_returns)):
            prev = local_returns[idx - 1]
            cur = local_returns[idx]
            if abs(prev) <= 1e-12 or abs(cur) <= 1e-12:
                continue
            sign_pairs += 1
            if prev * cur > 0.0:
                same_sign += 1

        if local_returns:
            shock_thr = max(_quantile([abs(v) for v in local_returns], 0.92, 0.0), 1e-6)
            for idx in range(len(local_returns) - 1):
                cur_abs = abs(local_returns[idx])
                if cur_abs < shock_thr:
                    continue
                shock_hits += 1
                nxt = local_returns[idx + 1]
                if local_returns[idx] * nxt > 0.0:
                    shock_follow_same += 1
                elif local_returns[idx] * nxt < 0.0:
                    shock_follow_reverse += 1

    abs_returns = [abs(v) for v in returns]
    sigma = _mean(abs_returns, 0.0)
    drift = _mean(returns, 0.0)
    median_spread = max(_quantile(spread_values, 0.50, 1.0), 1.0)
    p90_spread = _quantile(spread_values, 0.90, median_spread)
    p98_spread = _quantile(spread_values, 0.98, p90_spread)
    gap_med = _quantile(gap_values, 0.50, 0.0)
    gap_p95 = _quantile(gap_values, 0.95, gap_med)
    spread_shock_prob = 0.0
    if spread_values:
        shock_threshold = max(p90_spread, median_spread * 1.5)
        spread_shock_prob = sum(1 for value in spread_values if value >= shock_threshold) / float(len(spread_values))

    edge_vol = _mean(edge_abs_returns, sigma)
    non_edge_vol = _mean(non_edge_abs_returns, sigma)
    edge_spread = _mean(edge_spreads, median_spread)
    non_edge_spread = _mean(non_edge_spreads, median_spread)

    shadow = symbol_shadow_summary(conn, profile_name, symbol)
    persistence = (same_sign / float(sign_pairs)) if sign_pairs > 0 else 0.5
    shock_memory = (shock_follow_same / float(shock_hits)) if shock_hits > 0 else 0.0
    shock_reversal = (shock_follow_reverse / float(shock_hits)) if shock_hits > 0 else 0.0
    vol_thr = max(_quantile(abs_returns, 0.65, sigma), sigma)
    spread_thr = max(_quantile(spread_values, 0.75, median_spread), median_spread)
    state_names = ["calm_trend", "calm_revert", "stress_trend", "stress_revert"]
    state_counts = [0, 0, 0, 0]
    state_ret_sum = [0.0, 0.0, 0.0, 0.0]
    state_abs_sum = [0.0, 0.0, 0.0, 0.0]
    state_spread_sum = [0.0, 0.0, 0.0, 0.0]
    transition = [[0, 0, 0, 0] for _ in range(4)]
    prev_state = -1
    prev_ret = 0.0
    state_changes = 0
    stress_stay = 0
    stress_obs = 0
    revert_obs = 0
    for ret, spread, edge_score in bar_records:
        vol_hi = abs(ret) >= vol_thr
        trend_like = (prev_ret == 0.0 or ret * prev_ret >= 0.0)
        state = (2 if vol_hi or spread >= spread_thr else 0) + (0 if trend_like else 1)
        state_counts[state] += 1
        state_ret_sum[state] += ret
        state_abs_sum[state] += abs(ret)
        state_spread_sum[state] += spread
        if state in (1, 3):
            revert_obs += 1
        if prev_state >= 0:
            transition[prev_state][state] += 1
            if state != prev_state:
                state_changes += 1
        if prev_state in (2, 3):
            stress_obs += 1
            if state in (2, 3):
                stress_stay += 1
        prev_state = state
        prev_ret = ret * (1.0 + 0.10 * edge_score)

    total_states = float(max(sum(state_counts), 1))
    regime_transition_burst = state_changes / float(max(len(bar_records) - 1, 1))
    vol_cluster_bias = (stress_stay / float(max(stress_obs, 1))) if stress_obs > 0 else 0.0
    mean_revert_bias = (revert_obs / total_states)
    transition_entropy = 0.0
    for idx in range(4):
        transition_entropy += (state_counts[idx] / total_states) * _norm_entropy(transition[idx])
    shock_decay = _clamp(0.35 + 0.45 * (1.0 - shock_memory) + 0.20 * shock_reversal, 0.0, 1.5)
    state_prototypes = []
    for idx, name in enumerate(state_names):
        count = max(state_counts[idx], 1)
        row_sum = max(sum(transition[idx]), 1)
        state_prototypes.append({
            "state": name,
            "weight": state_counts[idx] / total_states,
            "sigma_scale": _clamp((state_abs_sum[idx] / float(count)) / max(sigma, 1e-6), 0.40, 4.00),
            "drift_bias": _clamp(state_ret_sum[idx] / float(count), -0.001, 0.001),
            "spread_scale": _clamp((state_spread_sum[idx] / float(count)) / max(median_spread, 1.0), 0.50, 6.00),
            "transition": [
                _clamp(transition[idx][j] / float(row_sum), 0.0, 1.0)
                for j in range(4)
            ],
        })

    return {
        "profile_name": profile_name,
        "symbol": symbol,
        "dataset_count": len(dataset_payloads),
        "bar_count": total_bars,
        "sigma_scale": _clamp(
            0.85 + 4200.0 * sigma +
            0.18 * float(shadow.get("mean_route_regret", 0.0)),
            0.50,
            3.00,
        ),
        "drift_bias": _clamp(
            drift + 0.00005 * float(shadow.get("mean_route_value", 0.0)) -
            0.00005 * float(shadow.get("mean_policy_no_trade_prob", 0.0)),
            -0.001,
            0.001,
        ),
        "spread_scale": _clamp(
            (p90_spread / median_spread) +
            0.08 * float(shadow.get("mean_portfolio_pressure", 0.0)),
            0.50,
            4.00,
        ),
        "gap_prob": _clamp(
            sum(1 for value in gap_values if value >= max(gap_p95, gap_med * 2.0, 1e-6)) /
            float(max(len(gap_values), 1)),
            0.0,
            0.30,
        ),
        "gap_scale": _clamp(
            1.0 + (gap_p95 / max(gap_med, 1e-6)) +
            1.2 * float(shadow.get("mean_portfolio_pressure", 0.0)),
            0.0,
            8.0,
        ),
        "flip_prob": _clamp(
            1.0 - persistence + 0.08 * float(shadow.get("mean_route_regret", 0.0)),
            0.0,
            0.50,
        ),
        "context_corr_bias": _clamp(
            0.20 * (persistence - 0.5) +
            0.35 * float(shadow.get("mean_portfolio_div", 0.0)) -
            0.30 * float(shadow.get("mean_portfolio_corr", 0.0)),
            -1.0,
            1.0,
        ),
        "liquidity_stress": _clamp(
            ((edge_spread / max(non_edge_spread, 1.0)) - 1.0) +
            0.25 * float(shadow.get("mean_portfolio_supervisor_score", 0.0)),
            0.0,
            3.0,
        ),
        "macro_focus": _clamp(
            0.18 * ((edge_vol / max(non_edge_vol, 1e-6)) - 1.0) +
            0.20 * float(shadow.get("mean_policy_no_trade_prob", 0.0)),
            0.0,
            1.5,
        ),
        "session_edge_focus": _clamp(
            0.45 * ((edge_vol / max(non_edge_vol, 1e-6)) - 1.0) +
            0.30 * ((edge_spread / max(non_edge_spread, 1.0)) - 1.0),
            0.0,
            1.5,
        ),
        "trend_persistence": _clamp(persistence, 0.0, 1.0),
        "shock_memory": _clamp(shock_memory, 0.0, 1.0),
        "recovery_bias": _clamp(shock_reversal - shock_memory, -1.0, 1.0),
        "spread_shock_prob": _clamp(spread_shock_prob, 0.0, 0.50),
        "spread_shock_scale": _clamp(p98_spread / median_spread, 1.0, 8.0),
        "regime_transition_burst": _clamp(regime_transition_burst, 0.0, 1.0),
        "transition_entropy": _clamp(transition_entropy, 0.0, 1.0),
        "mean_revert_bias": _clamp(mean_revert_bias, 0.0, 1.0),
        "vol_cluster_bias": _clamp(vol_cluster_bias, 0.0, 1.0),
        "shock_decay": shock_decay,
        "asia_sigma_scale": _clamp(_mean(session_abs_returns["asia"], sigma) / max(sigma, 1e-6), 0.50, 3.00),
        "london_sigma_scale": _clamp(_mean(session_abs_returns["london"], sigma) / max(sigma, 1e-6), 0.50, 3.00),
        "newyork_sigma_scale": _clamp(_mean(session_abs_returns["newyork"], sigma) / max(sigma, 1e-6), 0.50, 3.00),
        "asia_spread_scale": _clamp(_mean(session_spreads["asia"], median_spread) / max(median_spread, 1.0), 0.50, 4.00),
        "london_spread_scale": _clamp(_mean(session_spreads["london"], median_spread) / max(median_spread, 1.0), 0.50, 4.00),
        "newyork_spread_scale": _clamp(_mean(session_spreads["newyork"], median_spread) / max(median_spread, 1.0), 0.50, 4.00),
        "state_prototypes": state_prototypes,
        "shadow_summary": shadow,
        "datasets": dataset_payloads,
    }


def write_world_model_artifacts(conn: libsql.Connection,
                                args,
                                symbols: list[str]) -> list[dict]:
    out_dir = RESEARCH_DIR / safe_token(args.profile) / "WorldModels"
    ensure_dir(out_dir)
    artifacts: list[dict] = []
    for symbol in sorted({str(s).upper() for s in symbols if str(s).strip()}):
        payload = build_symbol_world_model(conn, args.profile, symbol)
        artifact_path = out_dir / f"world_model_{safe_token(symbol)}.json"
        artifact_path.write_text(
            json.dumps(portableize_payload_paths(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        artifacts.append({
            "symbol": symbol,
            "artifact_path": str(artifact_path),
            "artifact_sha256": testlab.sha256_path(artifact_path),
            "payload": payload,
        })
    summary_path = out_dir / "world_models.json"
    summary_path.write_text(
        json.dumps(portableize_payload_paths(artifacts), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifacts
