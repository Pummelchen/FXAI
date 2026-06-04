from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = ROOT / "FXPlugins" / "API" / "Backends" / "Python"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _load_module_backend():
    path = BACKEND_DIR / "fxai_plugin_module_backend.py"
    spec = importlib.util.spec_from_file_location("fxai_plugin_module_backend_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_financial_utility_loss_penalizes_wrong_adverse_trade_more_than_correct_trade():
    import torch

    from fxai_financial_loss import financial_utility_loss_torch

    targets = {
        "labelClass": 0,
        "movePoints": -12.0,
        "sampleWeight": 2.0,
        "mfePoints": 14.0,
        "maePoints": 5.0,
        "pathRisk": 0.45,
        "fillRisk": 0.25,
        "priceCostPoints": 1.2,
        "minMovePoints": 2.0,
    }
    good_logits = torch.tensor([[6.0, -3.0, -2.0]], dtype=torch.float32, requires_grad=True)
    bad_logits = torch.tensor([[-3.0, 6.0, -2.0]], dtype=torch.float32, requires_grad=True)
    move = torch.tensor([12.0], dtype=torch.float32, requires_grad=True)
    quantiles = torch.tensor([[8.0, 10.0, 12.0, 14.0, 16.0]], dtype=torch.float32, requires_grad=True)

    good_loss = financial_utility_loss_torch(good_logits, [0], move, [12.0], quantiles, targets)
    bad_loss = financial_utility_loss_torch(bad_logits, [0], move, [12.0], quantiles, targets)

    assert math.isfinite(float(good_loss.detach()))
    assert math.isfinite(float(bad_loss.detach()))
    assert float(bad_loss.detach()) > float(good_loss.detach()) * 4.0
    bad_loss.backward()
    assert bad_logits.grad is not None


def test_module_backend_forwards_financial_targets_only_to_opted_in_train_steps():
    backend = _load_module_backend()

    class OptedInModule:
        @staticmethod
        def train_step(
            batch,
            labels,
            moves,
            state=None,
            data_has_volume=True,
            financial_targets=None,
            financial_loss_config=None,
        ):
            assert batch == [[[0.1, 0.2]]]
            assert labels == [1]
            assert moves == [3.0]
            assert data_has_volume is True
            state["targets"] = financial_targets
            state["config"] = financial_loss_config
            return state

    state = {}
    result = backend._call_train_step(
        OptedInModule,
        [[0.1, 0.2]],
        1,
        3.0,
        state,
        True,
        financial_targets={"movePoints": 3.0, "maePoints": 0.7},
        financial_loss_config={"version": "fxai-financial-loss-v1", "adverseTailWeight": 0.2},
    )

    assert result["targets"]["maePoints"] == 0.7
    assert result["config"]["adverseTailWeight"] == 0.2

    class LegacyModule:
        @staticmethod
        def train_step(batch, labels, moves, state=None):
            state["legacy"] = (batch, labels, moves)
            return state

    legacy_state = {}
    legacy_result = backend._call_train_step(
        LegacyModule,
        [[0.1]],
        2,
        0.0,
        legacy_state,
        False,
        financial_targets={"movePoints": 0.0},
        financial_loss_config={"version": "fxai-financial-loss-v1"},
    )
    assert legacy_result["legacy"] == ([[[0.1]]], [2], [0.0])
