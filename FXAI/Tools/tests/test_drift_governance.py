from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from offline_lab.adaptive_router import write_adaptive_router_profiles
from offline_lab.common import close_db, commit_db, connect_db, now_unix, query_one, query_scalar, sha256_text
from offline_lab.drift_governance import build_drift_governance_report, run_drift_governance_cycle
from offline_lab.drift_governance_config import default_config
from offline_lab.environment import bootstrap_environment
from offline_lab.fixtures import patched_paths, seed_completed_run_fixture, seed_profile_fixture
from offline_lab.student_router import write_student_router_profiles


def _insert_shadow_observation(
    conn,
    *,
    profile_name: str,
    symbol: str,
    plugin_name: str,
    family_id: int,
    captured_at: int,
    source_tag: str,
    meta_weight: float,
    reliability: float,
    global_edge: float,
    context_edge: float,
    context_regret: float,
    portfolio_objective: float,
    portfolio_stability: float,
    route_value: float,
    route_regret: float,
    shadow_score: float,
    regime_id: int,
    policy_no_trade_prob: float,
    portfolio_pressure: float,
    control_plane_score: float,
    portfolio_supervisor_score: float,
) -> None:
    payload_json = json.dumps(
        {
            "captured_from": "test_drift_governance",
            "source_tag": source_tag,
            "plugin_name": plugin_name,
        },
        sort_keys=True,
    )
    conn.execute(
        """
        INSERT INTO shadow_fleet_observations(
            profile_name, symbol, plugin_name, family_id, captured_at, source_path, source_sha256,
            meta_weight, reliability, global_edge, context_edge, context_regret, portfolio_objective,
            portfolio_stability, portfolio_corr, portfolio_div, route_value, route_regret,
            route_counterfactual, shadow_score, regime_id, horizon_minutes, obs_count,
            policy_enter_prob, policy_no_trade_prob, policy_exit_prob, policy_add_prob, policy_reduce_prob,
            policy_timeout_prob, policy_tighten_prob, policy_portfolio_fit, policy_capital_efficiency,
            policy_lifecycle_action, portfolio_pressure, control_plane_score, portfolio_supervisor_score, payload_json
        )
        VALUES(
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.18, 0.62, ?, ?, 0.20, ?, ?, 5, 32,
            0.58, ?, 0.16, 0.24, 0.22, 0.20, 0.18, 0.64, 0.66, 2, ?, ?, ?, ?
        )
        """,
        (
            profile_name,
            symbol,
            plugin_name,
            family_id,
            captured_at,
            f"/fixture/{symbol}_{plugin_name}_{source_tag}.tsv",
            sha256_text(f"{profile_name}:{symbol}:{plugin_name}:{captured_at}:{source_tag}"),
            meta_weight,
            reliability,
            global_edge,
            context_edge,
            context_regret,
            portfolio_objective,
            portfolio_stability,
            route_value,
            route_regret,
            shadow_score,
            regime_id,
            policy_no_trade_prob,
            portfolio_pressure,
            control_plane_score,
            portfolio_supervisor_score,
            payload_json,
        ),
    )


def _seed_drift_windows(
    conn,
    fixture: dict[str, object],
    *,
    plugin_name: str | None = None,
    degraded_recent: bool = True,
) -> None:
    profile_name = str(fixture["profile_name"])
    symbol = str(fixture["symbol"])
    family_id = int(fixture["family_id"])
    plugin = plugin_name or str(fixture["plugin_name"])
    base_now = now_unix()
    conn.execute(
        "DELETE FROM shadow_fleet_observations WHERE profile_name = ? AND symbol = ? AND plugin_name = ?",
        (profile_name, symbol, plugin),
    )
    for idx in range(12):
        _insert_shadow_observation(
            conn,
            profile_name=profile_name,
            symbol=symbol,
            plugin_name=plugin,
            family_id=family_id,
            captured_at=base_now - 518400 - (idx * 7200),
            source_tag=f"reference_{idx}",
            meta_weight=1.10,
            reliability=0.82,
            global_edge=0.34,
            context_edge=0.27,
            context_regret=0.11,
            portfolio_objective=0.42,
            portfolio_stability=0.71,
            route_value=0.26,
            route_regret=0.12,
            shadow_score=0.46,
            regime_id=1,
            policy_no_trade_prob=0.28,
            portfolio_pressure=0.24,
            control_plane_score=0.31,
            portfolio_supervisor_score=0.37,
        )
    for idx in range(6):
        if degraded_recent:
            recent_values = {
                "meta_weight": 1.68,
                "reliability": 0.14,
                "global_edge": -0.28,
                "context_edge": -0.33,
                "context_regret": 0.92,
                "portfolio_objective": -0.36,
                "portfolio_stability": 0.10,
                "route_value": -0.31,
                "route_regret": 0.95,
                "shadow_score": -0.41,
                "regime_id": 5,
                "policy_no_trade_prob": 0.96,
                "portfolio_pressure": 0.88,
                "control_plane_score": 0.94,
                "portfolio_supervisor_score": 0.98,
            }
        else:
            recent_values = {
                "meta_weight": 1.18,
                "reliability": 0.76,
                "global_edge": 0.39,
                "context_edge": 0.32,
                "context_regret": 0.08,
                "portfolio_objective": 0.48,
                "portfolio_stability": 0.73,
                "route_value": 0.31,
                "route_regret": 0.10,
                "shadow_score": 0.42,
                "regime_id": 2,
                "policy_no_trade_prob": 0.24,
                "portfolio_pressure": 0.22,
                "control_plane_score": 0.28,
                "portfolio_supervisor_score": 0.34,
            }
        _insert_shadow_observation(
            conn,
            profile_name=profile_name,
            symbol=symbol,
            plugin_name=plugin,
            family_id=family_id,
            captured_at=base_now - (idx * 900),
            source_tag=f"recent_{idx}",
            meta_weight=recent_values["meta_weight"],
            reliability=recent_values["reliability"],
            global_edge=recent_values["global_edge"],
            context_edge=recent_values["context_edge"],
            context_regret=recent_values["context_regret"],
            portfolio_objective=recent_values["portfolio_objective"],
            portfolio_stability=recent_values["portfolio_stability"],
            route_value=recent_values["route_value"],
            route_regret=recent_values["route_regret"],
            shadow_score=recent_values["shadow_score"],
            regime_id=int(recent_values["regime_id"]),
            policy_no_trade_prob=recent_values["policy_no_trade_prob"],
            portfolio_pressure=recent_values["portfolio_pressure"],
            control_plane_score=recent_values["control_plane_score"],
            portfolio_supervisor_score=recent_values["portfolio_supervisor_score"],
        )
    commit_db(conn)


def test_drift_governance_cycle_updates_state_and_router_artifacts():
    with tempfile.TemporaryDirectory(prefix="fxai_drift_governance_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            fixture = seed_profile_fixture(conn, profile_name="driftgov")
            _seed_drift_windows(conn, fixture)
            args = argparse.Namespace(
                profile=fixture["profile_name"],
                db=str(paths["default_db"]),
                group_key="drift_cycle",
                runtime_mode="research",
            )

            payload = run_drift_governance_cycle(conn, args, "drift_cycle")
            router_payload = write_student_router_profiles(conn, args)
            adaptive_payload = write_adaptive_router_profiles(conn, args)

            assert payload["actions"]
            state = query_one(
                conn,
                """
                SELECT governance_state, action_recommendation, action_applied, weight_multiplier, restrict_live
                  FROM plugin_governance_states
                 WHERE profile_name = ? AND symbol = ? AND plugin_name = ?
                """,
                (fixture["profile_name"], fixture["symbol"], fixture["plugin_name"]),
            )
            assert state is not None
            assert str(state["action_recommendation"]) != "NONE"
            assert int(state["action_applied"]) == 1
            assert float(state["weight_multiplier"]) <= 0.58
            router_json = json.loads(
                (paths["research_dir"] / fixture["profile_name"] / f"student_router_{fixture['symbol']}.json").read_text(encoding="utf-8")
            )
            override = dict(router_json["governance_overrides"][fixture["plugin_name"]])
            assert override["action_applied"] is True
            assert float(router_json["plugin_weights"][fixture["plugin_name"]]) <= 0.58
            adaptive_json = json.loads(
                (paths["research_dir"] / fixture["profile_name"] / f"adaptive_router_{fixture['symbol']}.json").read_text(encoding="utf-8")
            )
            assert fixture["plugin_name"] in adaptive_json["governance_overrides"]
            assert len(router_payload) == 1
            assert len(adaptive_payload) == 1
            close_db(conn)


def test_drift_governance_recommend_only_preserves_operational_state():
    with tempfile.TemporaryDirectory(prefix="fxai_drift_recommend_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            config = default_config()
            config["action_mode"] = "RECOMMEND_ONLY"
            (paths["drift_governance_dir"] / "drift_governance_config.json").write_text(
                json.dumps(config, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            conn = connect_db(paths["default_db"])
            fixture = seed_profile_fixture(conn, profile_name="driftrecommend")
            _seed_drift_windows(conn, fixture)
            args = argparse.Namespace(
                profile=fixture["profile_name"],
                db=str(paths["default_db"]),
                group_key="recommend_only",
                runtime_mode="research",
            )

            payload = run_drift_governance_cycle(conn, args, "recommend_only")

            state = query_one(
                conn,
                """
                SELECT governance_state, action_recommendation, action_applied
                  FROM plugin_governance_states
                 WHERE profile_name = ? AND symbol = ? AND plugin_name = ?
                """,
                (fixture["profile_name"], fixture["symbol"], fixture["plugin_name"]),
            )
            assert payload["actions"]
            assert state is not None
            assert str(state["action_recommendation"]) != "NONE"
            assert int(state["action_applied"]) == 0
            assert str(state["governance_state"]) == "CHAMPION"
            close_db(conn)


def test_drift_governance_tracks_challenger_eligibility():
    with tempfile.TemporaryDirectory(prefix="fxai_drift_challenger_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            fixture = seed_profile_fixture(conn, profile_name="driftchallenger")
            _seed_drift_windows(conn, fixture)

            challenger_fixture = dict(fixture)
            challenger_fixture["plugin_name"] = "ai_tft"
            challenger_fixture["family_id"] = 4
            params_json = json.dumps({"PredictionTargetMinutes": 5, "M1SyncBars": 3, "FeatureSchema": 6, "variant": "challenger"}, sort_keys=True)
            conn.execute(
                """
                INSERT OR REPLACE INTO best_configs(
                    dataset_scope, dataset_id, profile_name, symbol, plugin_name, ai_id, family_id, run_id,
                    promoted_at, score, ranking_score, support_count, parameters_json, audit_set_path, ea_set_path, support_json
                )
                VALUES('aggregate', ?, ?, ?, ?, 7, ?, NULL, ?, 89.2, 88.7, 3, ?, ?, ?, '[]')
                """,
                (
                    int(query_scalar(conn, "SELECT id FROM datasets WHERE dataset_key = ?", (fixture["dataset_key"],), 0)),
                    fixture["profile_name"],
                    fixture["symbol"],
                    challenger_fixture["plugin_name"],
                    challenger_fixture["family_id"],
                    now_unix(),
                    params_json,
                    f"/fixture/{fixture['symbol']}_ai_tft_audit.set",
                    f"/fixture/{fixture['symbol']}_ai_tft_ea.set",
                ),
            )
            for variant in range(3):
                params = {
                    "bars": 1000,
                    "horizon": 5,
                    "m1sync_bars": 3,
                    "normalization": 14 + variant,
                    "sequence_bars": 16,
                    "schema_id": 6,
                    "feature_mask": 0,
                    "execution_profile": "default",
                    "wf_train_bars": 256,
                    "wf_test_bars": 64,
                    "wf_purge_bars": 32,
                    "wf_embargo_bars": 24,
                    "variant": variant,
                }
                seed_completed_run_fixture(conn, challenger_fixture, parameters=params, score=89.0 + variant)
            _seed_drift_windows(conn, challenger_fixture, plugin_name="ai_tft", degraded_recent=False)

            args = argparse.Namespace(
                profile=fixture["profile_name"],
                db=str(paths["default_db"]),
                group_key="challenger_cycle",
                runtime_mode="research",
            )

            payload = run_drift_governance_cycle(conn, args, "challenger_cycle")
            report = build_drift_governance_report(conn, fixture["profile_name"])

            eval_row = query_one(
                conn,
                """
                SELECT eligibility_state, qualifies, support_count, promotion_margin
                  FROM plugin_challenger_evaluations
                 WHERE profile_name = ? AND symbol = ? AND plugin_name = ?
                 ORDER BY created_at DESC, id DESC
                 LIMIT 1
                """,
                (fixture["profile_name"], fixture["symbol"], "ai_tft"),
            )
            assert payload["actions"]
            assert eval_row is not None
            assert str(eval_row["eligibility_state"]) == "QUALIFIED"
            assert int(eval_row["qualifies"]) == 1
            assert int(eval_row["support_count"]) >= 3
            assert float(eval_row["promotion_margin"]) > 0.0
            symbol_report = next(item for item in report["symbols"] if item["symbol"] == fixture["symbol"])
            challenger_plugins = [item for item in symbol_report["plugins"] if item["plugin_name"] == "ai_tft"]
            assert challenger_plugins
            assert challenger_plugins[0]["challenger_evaluation"]["qualifies"] is True
            close_db(conn)
