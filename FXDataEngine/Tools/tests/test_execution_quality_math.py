from __future__ import annotations

from offline_lab.execution_quality_math import compute_execution_quality_forecast


def test_execution_quality_partial_fill_risk_blends_probability_and_shortfall():
    config = {
        "state_thresholds": {
            "stressed_min": 0.36,
            "caution_min": 0.54,
        },
        "lot_scales": {},
        "enter_prob_buffers": {},
        "forecast_caps": {},
        "weights": {},
        "support_soft_floor": 64,
        "allow_block_state": True,
        "block_on_unknown": True,
    }
    tier = {
        "kind": "GLOBAL",
        "support": 400,
        "quality": 0.68,
        "price_cost_mult": 1.02,
        "slippage_mult": 0.96,
        "fill_quality_bias": 0.05,
        "latency_mult": 1.02,
        "fragility_mult": 0.92,
        "deviation_mult": 1.0,
        "fallback_used": False,
    }

    result = compute_execution_quality_forecast(
        config=config,
        tier=tier,
        symbol="EURUSD",
        session_label="LONDON_NY_OVERLAP",
        regime_label="TREND_PERSISTENT",
        current_price_cost_points=0.8,
        broker_slippage_points=0.2,
        broker_latency_points=0.4,
        broker_reject_prob=0.02,
        broker_partial_fill_prob=0.05,
        broker_fill_ratio_mean=0.50,
        broker_event_burst_penalty=0.02,
        micro_price_cost_zscore=0.30,
        micro_hostile_execution=0.16,
        micro_liquidity_stress=0.18,
        micro_vol_burst=0.50,
        micro_tick_rate_zscore=0.40,
        micro_tick_imbalance=0.22,
        news_risk_score=0.08,
        rates_risk_score=0.10,
        stale_context_count=0,
        news_window_active=False,
        rates_repricing_active=False,
        handoff_flag=False,
        support_usable=True,
        data_stale=False,
        base_allowed_deviation_points=4.0,
    )

    assert "BROKER_PARTIAL_FILL_RISK_ELEVATED" not in result["reason_codes"]
    assert result["execution_state"] in {"NORMAL", "CAUTION"}
