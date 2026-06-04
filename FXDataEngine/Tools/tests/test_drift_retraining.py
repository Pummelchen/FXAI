from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from offline_lab.common import close_db, commit_db, connect_db, now_unix, query_all, query_one
from offline_lab.drift_retraining import (
    build_drift_retraining_report,
    build_retraining_campaign_plan,
    load_retraining_requests_for_execution,
    mark_retraining_execution,
    queue_drift_retraining_requests,
)
from offline_lab.environment import bootstrap_environment
from offline_lab.fixtures import patched_paths, seed_profile_fixture
from offline_lab.promotion import SERIOUS_SCENARIOS


def _args(profile: str, **overrides):
    values = {
        "profile": profile,
        "symbol": "",
        "plugin": "",
        "min_risk_score": 0.56,
        "include_low_support": False,
        "data_days": 90,
        "months_list": "",
        "wf_train_years": "1,2,3,5",
        "scenario_list": SERIOUS_SCENARIOS,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _insert_governance_state(conn, fixture: dict[str, object], *, low_support: bool = False, risk: float = 0.72) -> None:
    now = now_unix()
    support = {
        "sample_count_recent": 6 if not low_support else 1,
        "sample_count_reference": 12 if not low_support else 2,
        "reference_scope": "SYMBOL_PLUGIN",
    }
    payload = {
        "timestamp": "2026-06-05T00:00:00Z",
        "plugin_id": fixture["plugin_name"],
        "pair_scope": fixture["symbol"],
        "health_state": "DEGRADED",
        "governance_state": "DEGRADED",
        "recommended_governance_state": "DEGRADED",
        "drift_scores": {
            "feature_drift_score": 0.68,
            "calibration_drift_score": 0.62,
            "performance_drift_score": 0.74,
        },
        "drift_details": {},
        "support": support,
        "action_recommendation": "DOWNWEIGHT",
        "action_applied": True,
        "weight_multiplier": 0.58,
        "reason_codes": ["FEATURE_DRIFT_ELEVATED", "RECENT_COST_ADJUSTED_UTILITY_WEAK"],
        "quality_flags": {"low_support": low_support, "reference_fallback_scope": "SYMBOL_PLUGIN"},
        "metadata": {
            "reference_window": {"start": now - 900000, "end": now - 500000},
            "live_window": {"start": now - 7200, "end": now},
            "policy_version": 1,
            "cycle_group_key": "test_cycle",
        },
    }
    conn.execute(
        """
        INSERT INTO plugin_governance_states(
            profile_name, symbol, plugin_name, family_id, base_registry_status, health_state, governance_state,
            action_recommendation, action_applied, weight_multiplier, restrict_live, shadow_only, disabled,
            candidate_eligible, champion_eligible, aggregate_risk_score, reason_codes_json, quality_flags_json,
            payload_json, policy_version, updated_at
        )
        VALUES(?, ?, ?, ?, 'champion', 'DEGRADED', 'DEGRADED', 'DOWNWEIGHT', 1, 0.58, 0, 0, 0, 0, 0, ?, ?, ?, ?, 1, ?)
        ON CONFLICT(profile_name, symbol, plugin_name) DO UPDATE SET
            aggregate_risk_score=excluded.aggregate_risk_score,
            reason_codes_json=excluded.reason_codes_json,
            quality_flags_json=excluded.quality_flags_json,
            payload_json=excluded.payload_json,
            updated_at=excluded.updated_at
        """,
        (
            fixture["profile_name"],
            fixture["symbol"],
            fixture["plugin_name"],
            fixture["family_id"],
            risk,
            json.dumps(payload["reason_codes"], sort_keys=True),
            json.dumps(payload["quality_flags"], sort_keys=True),
            json.dumps(payload, sort_keys=True),
            now,
        ),
    )
    commit_db(conn)


def test_drift_retraining_queue_is_idempotent_and_emits_one_alert():
    with tempfile.TemporaryDirectory(prefix="fxai_drift_retrain_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            fixture = seed_profile_fixture(conn, profile_name="driftretrain")
            _insert_governance_state(conn, fixture)

            first = queue_drift_retraining_requests(conn, _args(str(fixture["profile_name"])))
            second = queue_drift_retraining_requests(conn, _args(str(fixture["profile_name"])))

            rows = query_all(conn, "SELECT * FROM drift_retraining_requests WHERE profile_name = ?", (fixture["profile_name"],))
            assert len(rows) == 1
            assert first["queued_or_updated_count"] == 1
            assert first["new_alert_count"] == 1
            assert second["queued_or_updated_count"] == 1
            assert second["new_alert_count"] == 0
            alert_lines = (paths["drift_retraining_dir"] / "drift_retraining_alerts.jsonl").read_text(encoding="utf-8").strip().splitlines()
            assert len(alert_lines) == 1
            alert = json.loads(alert_lines[0])
            assert alert["component"] == "drift_retraining"
            assert alert["request"]["symbol"] == fixture["symbol"]
            close_db(conn)


def test_drift_retraining_low_support_requires_explicit_opt_in():
    with tempfile.TemporaryDirectory(prefix="fxai_drift_retrain_low_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            fixture = seed_profile_fixture(conn, profile_name="driftlow")
            _insert_governance_state(conn, fixture, low_support=True, risk=0.92)

            skipped = queue_drift_retraining_requests(conn, _args(str(fixture["profile_name"])))
            queued = queue_drift_retraining_requests(conn, _args(str(fixture["profile_name"]), include_low_support=True))

            assert skipped["queued_or_updated_count"] == 0
            assert skipped["skipped"][0]["reason"] == "low_support"
            assert queued["queued_or_updated_count"] == 1
            row = query_one(conn, "SELECT * FROM drift_retraining_requests WHERE profile_name = ?", (fixture["profile_name"],))
            assert row is not None
            assert int(row["priority"]) == 100
            close_db(conn)


def test_drift_retraining_execution_plan_keeps_promotion_manual():
    with tempfile.TemporaryDirectory(prefix="fxai_drift_retrain_plan_") as tmp_dir:
        with patched_paths(Path(tmp_dir)) as paths:
            bootstrap_environment(paths["default_db"], init_db=True)
            conn = connect_db(paths["default_db"])
            fixture = seed_profile_fixture(conn, profile_name="driftplan")
            _insert_governance_state(conn, fixture)
            queue_drift_retraining_requests(conn, _args(str(fixture["profile_name"]), data_days=120, wf_train_years="1,2,3,5"))

            exec_args = argparse.Namespace(
                profile=fixture["profile_name"],
                request_key="",
                symbol="",
                plugin="",
                status="queued",
                limit=1,
                skip_compile=True,
                limit_experiments=2,
                limit_runs=3,
                top_plugins=1,
            )
            requests = load_retraining_requests_for_execution(conn, exec_args)
            assert len(requests) == 1
            plan = build_retraining_campaign_plan(requests[0], exec_args, "manual_group")
            assert plan["auto_export"] is True
            assert plan["auto_promote"] is False
            assert plan["plugin_filter"] == fixture["plugin_name"]
            assert plan["promotion_command"] == "manual_best_params_only"
            assert plan["months_list"] == "4"
            assert plan["wf_year_presets"] == "1,2,3,5"

            mark_retraining_execution(
                conn,
                int(requests[0]["id"]),
                status="queued",
                payload={"dry_run": True, "plan": plan, "promotion_skipped": True},
            )
            commit_db(conn)
            report = build_drift_retraining_report(conn, str(fixture["profile_name"]))
            assert report["safety"]["auto_promote"] is False
            assert report["requests"][0]["execution"]["promotion_skipped"] is True
            close_db(conn)
