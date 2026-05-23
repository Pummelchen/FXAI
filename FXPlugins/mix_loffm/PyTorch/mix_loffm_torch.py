"""PyTorch/MPS inference helper for the FXAI mix_loffm plugin.

The Swift CPU implementation is the deterministic trading path. This module is
plugin-local research/acceleration code for batched gating and expert inference
on Apple Silicon MPS, with CPU fallback when MPS is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


EXPERTS = 4
DERIVED = 10
LATENT = 12


def preferred_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_gate_weights(device: torch.device) -> torch.Tensor:
    weights = torch.zeros((EXPERTS, DERIVED), dtype=torch.float32, device=device)
    weights[0, :10] = torch.tensor([1.20, 0.80, -0.25, 0.40, 0.15, 0.30, -0.20, 0.25, 0.10, 0.10], device=device)
    weights[1, :10] = torch.tensor([-0.85, 0.25, 0.70, 0.10, -0.25, 0.05, 0.25, -0.10, 0.15, -0.10], device=device)
    weights[2, :10] = torch.tensor([0.45, 1.10, -0.20, 0.90, 0.50, 0.10, -0.35, 0.20, 0.05, 0.05], device=device)
    weights[3, :10] = torch.tensor([-0.20, -0.35, 0.25, 0.10, 0.15, 0.55, 0.80, -0.10, 0.20, 0.0], device=device)
    return weights


def _seed_direction_weights(device: torch.device) -> torch.Tensor:
    weights = torch.zeros((EXPERTS, LATENT), dtype=torch.float32, device=device)
    for expert in range(EXPERTS):
        weights[expert, 0] = 0.05
        weights[expert, 1] = [0.35, -0.30, 0.22, 0.05][expert]
        weights[expert, 2] = 0.22 if expert == 2 else 0.10
        weights[expert, 3] = 0.18 if expert == 1 else 0.08
        weights[expert, 4] = -0.22 if expert == 3 else -0.06
        weights[expert, 5] = [0.16, -0.16, 0.08, 0.08][expert]
        weights[expert, 6] = [0.14, -0.08, 0.10, -0.08][expert]
        weights[expert, 7] = 0.12 if expert == 1 else 0.04
        weights[expert, 8] = -0.18 if expert == 3 else -0.04
        weights[expert, 9] = 0.06
        weights[expert, 10] = 0.04
        weights[expert, 11] = 0.18 if expert == 2 else 0.06
    return weights


@dataclass
class LoffmTorchState:
    gate_weights: torch.Tensor
    direction_weights: torch.Tensor
    skip_weights: torch.Tensor
    usage_ema: torch.Tensor
    hit_ema: torch.Tensor
    confidence_ema: torch.Tensor

    @classmethod
    def seeded(cls, device: torch.device | None = None) -> "LoffmTorchState":
        device = device or preferred_device()
        skip = torch.zeros((EXPERTS, DERIVED), dtype=torch.float32, device=device)
        skip[:, 0] = torch.tensor([-0.10, -0.10, -0.10, 0.25], device=device)
        skip[:, 5] = torch.tensor([0.05, 0.05, 0.05, 0.18], device=device)
        skip[:, 8] = torch.tensor([0.08, 0.08, 0.08, 0.20], device=device)
        return cls(
            gate_weights=_seed_gate_weights(device),
            direction_weights=_seed_direction_weights(device),
            skip_weights=skip,
            usage_ema=torch.full((EXPERTS,), 1.0 / EXPERTS, dtype=torch.float32, device=device),
            hit_ema=torch.full((EXPERTS,), 0.50, dtype=torch.float32, device=device),
            confidence_ema=torch.full((EXPERTS,), 0.50, dtype=torch.float32, device=device),
        )


def build_derived(features: torch.Tensor, session_bucket: int = 0) -> torch.Tensor:
    x = torch.clamp(features, -8.0, 8.0)

    def col(index: int) -> torch.Tensor:
        if index >= x.shape[-1]:
            return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        return x[..., index]

    def avg_abs(start: int, end: int) -> torch.Tensor:
        stop = min(end + 1, x.shape[-1])
        if start >= stop:
            return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        return torch.mean(torch.abs(x[..., start:stop]), dim=-1)

    f1, f2, f3, f4 = col(1), col(2), col(3), col(4)
    f5, f6, f7, f8 = col(5), col(6), col(7), col(8)
    f9, f10, f11, f12 = col(9), col(10), col(11), col(12)
    g1, g2, g3, g4 = avg_abs(13, 20), avg_abs(21, 32), avg_abs(33, 48), avg_abs(49, 62)
    session_bias = {-1: -0.10, 0: -0.10, 1: 0.20, 2: 0.35, 3: 0.15}.get(session_bucket, -0.10)
    return torch.stack(
        [
            torch.ones_like(f1),
            torch.clamp(0.48 * f1 + 0.34 * f2 + 0.20 * f3 - 0.10 * f4 + 0.08 * f12, -6.0, 6.0),
            torch.clamp(0.70 * torch.abs(f6) + 0.45 * torch.abs(f7) + 0.25 * g1, -6.0, 6.0),
            torch.clamp(-0.45 * f1 + 0.35 * f5 - 0.20 * f9 + 0.10 * f10, -6.0, 6.0),
            torch.clamp(0.55 * torch.abs(f2 - f5) + 0.35 * torch.abs(f3 - f4) + 0.15 * g2, -6.0, 6.0),
            torch.clamp(0.90 * torch.abs(f7) + 0.35 * torch.abs(f8) + 0.10 * g3, -6.0, 6.0),
            torch.clamp(0.45 * f10 - 0.35 * f11 + 0.25 * f12, -6.0, 6.0),
            torch.clamp(0.65 * f1 + 0.25 * f2 - 0.12 * f5 + 0.08 * g4, -6.0, 6.0),
            torch.clamp(0.35 * torch.abs(f3 - f2) + 0.35 * torch.abs(f5 - f4) + 0.20 * g1 + 0.10 * g4, -6.0, 6.0),
            torch.clamp(0.40 * f8 + 0.25 * f11 + 0.12 * g2 - 0.10 * g3, -6.0, 6.0) + session_bias,
        ],
        dim=-1,
    )


def predict_batch(
    features: Iterable[Iterable[float]] | torch.Tensor,
    session_bucket: int = 0,
    data_has_volume: bool = True,
    state: LoffmTorchState | None = None,
) -> dict[str, torch.Tensor]:
    device = state.gate_weights.device if state is not None else preferred_device()
    state = state or LoffmTorchState.seeded(device)
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim == 3:
        x = x[:, -1, :]
    if not data_has_volume:
        volume_indexes = [6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83]
        valid_indexes = [index for index in volume_indexes if index < x.shape[-1]]
        if valid_indexes:
            x = x.clone()
            x[..., valid_indexes] = 0.0
    derived = build_derived(x, session_bucket=session_bucket)
    logits = derived @ state.gate_weights.T
    logits = logits - 0.35 * (state.usage_ema - 1.0 / EXPERTS)
    logits = logits + 0.15 * state.hit_ema - 0.10 * state.confidence_ema
    gates = torch.softmax(torch.clamp(logits, -20.0, 20.0), dim=-1)
    latent_input = torch.zeros((x.shape[0], EXPERTS, LATENT), dtype=torch.float32, device=device)
    latent_input[..., 0] = 1.0
    latent_input[..., 1:9] = derived[:, None, 1:9]
    expert_up = torch.sigmoid(torch.sum(latent_input * state.direction_weights[None, :, :], dim=-1))
    skip_logits = derived @ state.skip_weights.T + 0.22 * torch.abs(derived[:, 5:6]) + 0.14 * torch.abs(derived[:, 8:9])
    expert_skip = torch.sigmoid(torch.clamp(skip_logits, -20.0, 20.0))
    buy = torch.sum(gates * (1.0 - expert_skip) * expert_up, dim=-1)
    sell = torch.sum(gates * (1.0 - expert_skip) * (1.0 - expert_up), dim=-1)
    skip = torch.sum(gates * expert_skip, dim=-1)
    probs = torch.stack([sell, buy, skip], dim=-1)
    probs = torch.clamp(probs, 0.0005, 0.9995)
    probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    return {"class_probabilities": probs, "expert_gates": gates}


def train_step(
    features: Iterable[Iterable[float]] | torch.Tensor,
    labels: Iterable[int],
    moves: Iterable[float],
    state: LoffmTorchState | None = None,
    lr: float = 0.03,
    session_bucket: int = 0,
    data_has_volume: bool = True,
) -> LoffmTorchState:
    state = state or LoffmTorchState.seeded()
    device = state.gate_weights.device
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim == 3:
        x = x[:, -1, :]
    if not data_has_volume:
        volume_indexes = [6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83]
        valid_indexes = [index for index in volume_indexes if index < x.shape[-1]]
        if valid_indexes:
            x = x.clone()
            x[..., valid_indexes] = 0.0
    derived = build_derived(x, session_bucket=session_bucket)
    prediction = predict_batch(x, session_bucket=session_bucket, data_has_volume=data_has_volume, state=state)
    probabilities = prediction["class_probabilities"]
    gates = prediction["expert_gates"]
    raw_labels = list(labels)
    if not raw_labels:
        raw_labels = [2] * x.shape[0]
    target = torch.tensor(raw_labels[: x.shape[0]], dtype=torch.long, device=device).clamp(0, 2)
    moves_tensor = torch.tensor(list(moves) or [1.0] * x.shape[0], dtype=torch.float32, device=device).abs().clamp_min(0.10)
    if moves_tensor.numel() < x.shape[0]:
        moves_tensor = torch.cat([moves_tensor, moves_tensor[-1:].repeat(x.shape[0] - moves_tensor.numel())])
    one_hot = torch.nn.functional.one_hot(target, num_classes=3).to(dtype=torch.float32)
    error = one_hot - probabilities
    signed_direction = error[:, 1] - error[:, 0]
    trade_pressure = error[:, 0] + error[:, 1] - error[:, 2]
    skip_pressure = error[:, 2] - 0.5 * (error[:, 0] + error[:, 1])
    expert_credit = gates * (0.5 + 0.5 * torch.abs(signed_direction).unsqueeze(-1))
    latent_train = torch.zeros((x.shape[0], EXPERTS, LATENT), dtype=torch.float32, device=device)
    latent_train[..., 0] = 1.0
    latent_train[..., 1:9] = derived[:, None, 1:9]
    state.direction_weights += lr * torch.mean(
        expert_credit.unsqueeze(-1)
        * signed_direction.view(-1, 1, 1)
        * moves_tensor.clamp(max=10.0).view(-1, 1, 1)
        * latent_train,
        dim=0,
    )
    state.gate_weights += lr * torch.mean(expert_credit.unsqueeze(-1) * trade_pressure.view(-1, 1, 1) * derived[:, None, :], dim=0)
    state.skip_weights += lr * torch.mean(gates.unsqueeze(-1) * skip_pressure.view(-1, 1, 1) * derived[:, None, :], dim=0)
    state.usage_ema = 0.98 * state.usage_ema + 0.02 * torch.mean(gates, dim=0)
    hit = torch.sum(probabilities * one_hot, dim=-1)
    state.hit_ema = 0.98 * state.hit_ema + 0.02 * torch.mean(gates * hit.unsqueeze(-1), dim=0).clamp(0.0, 1.0)
    state.confidence_ema = 0.98 * state.confidence_ema + 0.02 * torch.mean(gates * torch.max(probabilities, dim=-1).values.unsqueeze(-1), dim=0)
    return state
