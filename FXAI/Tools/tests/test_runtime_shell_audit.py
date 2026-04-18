from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(relpath: str) -> str:
    return (PROJECT_ROOT / relpath).read_text(encoding="utf-8")


def test_cycle_recovery_stays_symbol_scoped():
    fxai = _read("FXAI.mq5")
    assert "FXAI_RecoverManagedCycleState(_Symbol);" in fxai
    assert "CycleStartTime = FXAI_GetOldestPositionTime(managed_symbol);" in fxai
    assert re.search(r"datetime FXAI_GetOldestPositionTime\(const string symbol = \"\"\)", fxai)


def test_last_order_request_tracks_symbol_and_pending_order_lifecycle():
    fxai = _read("FXAI.mq5")
    execution = _read("Engine/Runtime/Trade/runtime_trade_execution.mqh")
    health = _read("Engine/Runtime/runtime_system_health.mqh")

    assert 'string   g_last_order_request_symbol = "";' in fxai
    assert "g_last_order_request_order_ticket = result.order;" in execution
    assert "g_last_order_request_uses_pending_order = plan.use_pending;" in execution
    assert "deal_symbol != g_last_order_request_symbol" in fxai
    assert "g_last_order_request_uses_pending_order" in health
    assert "cleared_completed_pending_request" in health
