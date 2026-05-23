"""PyTorch/MPS backend for ai_mlp.

The Swift CPU implementation is the deterministic online path. This module is
plugin-local acceleration code for batched sequence inference/training on Apple
Silicon MPS, with CPU fallback when MPS is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import torch

FEATURES = 24
HIDDEN = 16
CLASSES = 3
ARCH_ID = 9
VOLUME_INDEXES = [6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83]


def preferred_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def prepare_features(features: Iterable[Iterable[float]] | torch.Tensor, data_has_volume: bool = True) -> torch.Tensor:
    x = torch.as_tensor(features, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    x = torch.clamp(x, -50.0, 50.0)
    if not data_has_volume:
        valid = [idx for idx in VOLUME_INDEXES if idx < x.shape[-1]]
        if valid:
            x = x.clone()
            x[..., valid] = 0.0
    return x


def _col(x: torch.Tensor, index: int) -> torch.Tensor:
    if index >= x.shape[-1]:
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
    return x[..., index]


def build_features(x: torch.Tensor, horizon_minutes: int = 30, sequence_bars: int = 32, session_bucket: int = 0) -> torch.Tensor:
    arch = torch.sin(_col(x, 1) * (0.35 + 0.03 * ARCH_ID))
    values = [
        torch.ones_like(_col(x, 1)), _col(x, 1), _col(x, 2), _col(x, 3), _col(x, 4), _col(x, 7), _col(x, 12),
        torch.clamp(0.65 * _col(x, 40) + 0.35 * _col(x, 6), -8.0, 8.0),
        _col(x, 1), torch.abs(_col(x, 4)), torch.abs(_col(x, 3) - _col(x, 1)), _col(x, 7),
        _col(x, 1), _col(x, 2), _col(x, 64 + 14), _col(x, 64 + 19),
        torch.full_like(_col(x, 1), max(0.0, min(float(horizon_minutes) / 60.0, 2.0))),
        torch.full_like(_col(x, 1), max(0.0, min(float(session_bucket) / 5.0, 1.0))),
        torch.full_like(_col(x, 1), max(0.0, min(float(sequence_bars) / 128.0, 2.0))),
        arch, torch.tanh(_col(x, 2) + arch), torch.tanh(_col(x, 3) - arch), _col(x, 1) - _col(x, 2), _col(x, 2) - _col(x, 3)
    ]
    return torch.clamp(torch.stack(values, dim=-1), -8.0, 8.0)


@dataclass
class SequenceTorchState:
    w1: torch.Tensor
    b1: torch.Tensor
    head: torch.Tensor
    move: torch.Tensor

    @classmethod
    def seeded(cls, device: torch.device | None = None) -> "SequenceTorchState":
        device = device or preferred_device()
        gen = torch.Generator(device="cpu").manual_seed(10_000 + ARCH_ID)
        w1 = torch.randn((HIDDEN, FEATURES), generator=gen, dtype=torch.float32) * 0.08
        b1 = torch.randn((HIDDEN,), generator=gen, dtype=torch.float32) * 0.03
        head = torch.randn((CLASSES, HIDDEN + 1), generator=gen, dtype=torch.float32) * 0.04
        move = torch.abs(torch.randn((HIDDEN + 1,), generator=gen, dtype=torch.float32)) * 0.02
        return cls(w1.to(device), b1.to(device), head.to(device), move.to(device))


def predict_batch(features: Iterable[Iterable[float]] | torch.Tensor, *, data_has_volume: bool = True, horizon_minutes: int = 30, sequence_bars: int = 32, session_bucket: int = 0, state: SequenceTorchState | None = None) -> dict[str, torch.Tensor]:
    device = state.w1.device if state is not None else preferred_device()
    state = state or SequenceTorchState.seeded(device)
    x = prepare_features(features, data_has_volume=data_has_volume).to(device)
    z = build_features(x, horizon_minutes, sequence_bars, session_bucket)
    hidden = torch.tanh(z @ state.w1.T + state.b1)
    hidden_bias = torch.cat([torch.ones((hidden.shape[0], 1), dtype=hidden.dtype, device=device), hidden], dim=-1)
    probabilities = torch.softmax(torch.clamp(hidden_bias @ state.head.T, -30.0, 30.0), dim=-1)
    move = torch.clamp(hidden_bias @ state.move, min=0.0)
    return {"class_probabilities": probabilities, "expected_move_points": move, "hidden": hidden}


def train_step(features: Iterable[Iterable[float]] | torch.Tensor, labels: Iterable[int] | torch.Tensor, *, learning_rate: float = 0.01, state: SequenceTorchState | None = None) -> SequenceTorchState:
    device = state.w1.device if state is not None else preferred_device()
    state = state or SequenceTorchState.seeded(device)
    x = prepare_features(features).to(device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device).clamp(0, CLASSES - 1)
    z = build_features(x)
    hidden = torch.tanh(z @ state.w1.T + state.b1)
    hidden_bias = torch.cat([torch.ones((hidden.shape[0], 1), dtype=hidden.dtype, device=device), hidden], dim=-1)
    probs = torch.softmax(hidden_bias @ state.head.T, dim=-1)
    target = torch.nn.functional.one_hot(y, CLASSES).to(torch.float32)
    grad = (target - probs).T @ hidden_bias / max(1, x.shape[0])
    state.head = state.head + float(learning_rate) * grad
    return state
