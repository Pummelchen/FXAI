"""PyTorch/MPS inference helper for FXAI mix_moe_conformal.

The Swift CPU implementation is the deterministic trading path. This module is
plugin-local acceleration code for batched MoE routing and conformal gating on
Apple Silicon MPS. Live accelerator routing requires MPS; CPU execution is
reserved for explicit contract-test fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


EXPERTS = 4
REGIME = 11
FEATURES = 32
WEIGHTS = FEATURES + 1
BUCKETS = 12
CLASSES = 3


def preferred_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MoeConformalModule(nn.Module):
    """Modern MoE router with differentiable conformal calibration heads."""

    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Sequential(
            nn.LayerNorm(REGIME),
            nn.Linear(REGIME, 32),
            nn.GELU(),
            nn.Linear(32, EXPERTS),
        )
        self.expert_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(FEATURES),
                    nn.Linear(FEATURES, 32),
                    nn.GELU(),
                    nn.Linear(32, CLASSES),
                )
                for _ in range(EXPERTS)
            ]
        )
        self.move_heads = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(FEATURES), nn.Linear(FEATURES, 1), nn.Softplus())
                for _ in range(EXPERTS)
            ]
        )
        self.temperature = nn.Parameter(torch.ones(()))

    def forward(self, regime: torch.Tensor, features: torch.Tensor) -> dict[str, torch.Tensor]:
        gates = torch.softmax(self.router(regime) / self.temperature.clamp_min(0.20), dim=-1)
        expert_logits = torch.stack([head(features) for head in self.expert_heads], dim=1)
        logits = torch.sum(gates.unsqueeze(-1) * expert_logits, dim=1)
        moves = torch.cat([head(features) for head in self.move_heads], dim=-1)
        return {
            "class_probabilities": torch.softmax(logits, dim=-1),
            "expected_move_points": torch.sum(gates * moves, dim=-1),
            "expert_gates": gates,
        }


def split_conformal_cutoff(scores: torch.Tensor, alpha: float = 0.10) -> torch.Tensor:
    """Return the conservative split-conformal quantile for nonconformity scores."""
    if scores.numel() == 0:
        return torch.ones((), dtype=torch.float32, device=scores.device)
    sorted_scores = torch.sort(scores.flatten()).values
    rank = int(torch.ceil(torch.tensor((sorted_scores.numel() + 1) * (1.0 - alpha))).item()) - 1
    if rank >= sorted_scores.numel():
        return torch.ones((), dtype=sorted_scores.dtype, device=sorted_scores.device)
    return sorted_scores[max(rank, 0)]


def load_balance_loss(gates: torch.Tensor) -> torch.Tensor:
    """Keep MoE routing from collapsing into a single expert."""
    target = torch.full((EXPERTS,), 1.0 / EXPERTS, dtype=gates.dtype, device=gates.device)
    return torch.mean((gates.mean(dim=0) - target).pow(2))


def prepare_features(features: Iterable[Iterable[float]] | torch.Tensor, data_has_volume: bool = True) -> torch.Tensor:
    x = torch.as_tensor(features, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim == 3:
        x = x[:, -1, :]
    x = torch.clamp(x, -50.0, 50.0)
    if not data_has_volume:
        volume_indexes = [6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83]
        valid_indexes = [index for index in volume_indexes if index < x.shape[-1]]
        if valid_indexes:
            x = x.clone()
            x[..., valid_indexes] = 0.0
    return x


def _seed_router(device: torch.device) -> torch.Tensor:
    router = torch.zeros((EXPERTS, REGIME), dtype=torch.float32, device=device)
    for expert in range(EXPERTS):
        router[expert, min(expert + 1, REGIME - 1)] = 0.10
    return router


@dataclass
class MoeConformalTorchState:
    router: torch.Tensor
    gate_weights: torch.Tensor
    direction_weights: torch.Tensor
    move_weights: torch.Tensor
    usage_ema: torch.Tensor
    calibration_weights: torch.Tensor
    bucket_quantiles90: torch.Tensor
    model: MoeConformalModule | None = None
    optimizer: torch.optim.Optimizer | None = None

    @classmethod
    def seeded(cls, device: torch.device | None = None) -> "MoeConformalTorchState":
        device = device or preferred_device()
        model = MoeConformalModule().to(device)
        return cls(
            router=_seed_router(device),
            gate_weights=torch.zeros((EXPERTS, WEIGHTS), dtype=torch.float32, device=device),
            direction_weights=torch.zeros((EXPERTS, WEIGHTS), dtype=torch.float32, device=device),
            move_weights=torch.zeros((EXPERTS, WEIGHTS), dtype=torch.float32, device=device),
            usage_ema=torch.full((EXPERTS,), 1.0 / EXPERTS, dtype=torch.float32, device=device),
            calibration_weights=torch.zeros((CLASSES, 5), dtype=torch.float32, device=device),
            bucket_quantiles90=torch.full((BUCKETS,), 0.40, dtype=torch.float32, device=device),
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4),
        )


def _column(x: torch.Tensor, index: int) -> torch.Tensor:
    if index >= x.shape[-1]:
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
    return x[..., index]


def build_regime(features: torch.Tensor, session_bucket: int = 0) -> torch.Tensor:
    x = torch.clamp(features, -10.0, 10.0)
    r1, r5, r15, r60 = _column(x, 0), _column(x, 1), _column(x, 2), _column(x, 3)
    volatility = torch.abs(_column(x, 4))
    return torch.stack(
        [
            torch.ones_like(r1),
            torch.clamp(r1, -10.0, 10.0),
            torch.clamp(r5, -10.0, 10.0),
            torch.clamp(r15, -10.0, 10.0),
            torch.clamp(r60, -10.0, 10.0),
            torch.clamp(volatility, 0.0, 10.0),
            torch.clamp(r1 - r5, -10.0, 10.0),
            torch.clamp(r5 - r15, -10.0, 10.0),
            torch.clamp((r1 + r5 + r15) / torch.clamp(volatility, min=1.0e-6), -10.0, 10.0),
            torch.clamp(_column(x, 5), -10.0, 10.0),
            torch.clamp(_column(x, 6), -10.0, 10.0),
        ],
        dim=-1,
    )


def build_model_features(features: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(features, -10.0, 10.0)
    if x.shape[-1] >= FEATURES:
        return x[..., :FEATURES]
    padding = torch.zeros((*x.shape[:-1], FEATURES - x.shape[-1]), dtype=x.dtype, device=x.device)
    return torch.cat([x, padding], dim=-1)


def bucket_index(features: torch.Tensor, session_bucket: int = 0) -> torch.Tensor:
    session = max(0, min(3, int(session_bucket)))
    volatility = torch.abs(_column(features, 4))
    regime = torch.where(volatility < 0.75, 0, torch.where(volatility < 1.75, 1, 2))
    return torch.clamp(3 * session + regime, 0, BUCKETS - 1).long()


def predict_batch(
    features: Iterable[Iterable[float]] | torch.Tensor,
    session_bucket: int = 0,
    data_has_volume: bool = True,
    min_move_points: float = 0.10,
    price_cost_points: float = 0.0,
    state: MoeConformalTorchState | None = None,
) -> dict[str, torch.Tensor]:
    device = state.router.device if state is not None else preferred_device()
    state = state or MoeConformalTorchState.seeded(device)
    x = prepare_features(features, data_has_volume=data_has_volume).to(device)
    regime = build_regime(x, session_bucket=session_bucket)
    model_features = build_model_features(x)
    logits = regime @ state.router.T
    logits = logits - 0.35 * (state.usage_ema - 1.0 / EXPERTS)
    gates = torch.softmax(torch.clamp(logits, -30.0, 30.0), dim=-1)
    head_features = torch.cat(
        [torch.ones((x.shape[0], 1), dtype=torch.float32, device=device), model_features],
        dim=-1,
    )
    trade = torch.sigmoid(head_features @ state.gate_weights.T)
    up = torch.sigmoid(head_features @ state.direction_weights.T)
    move = torch.abs(head_features @ state.move_weights.T)
    p_trade = torch.sum(gates * torch.clamp(trade, 0.001, 0.999), dim=-1)
    p_up = torch.sum(gates * torch.clamp(up, 0.001, 0.999), dim=-1)
    expected_move = torch.sum(gates * move, dim=-1)
    buy = p_trade * p_up
    sell = p_trade * (1.0 - p_up)
    skip = 1.0 - p_trade
    q90 = state.bucket_quantiles90[bucket_index(x, session_bucket=session_bucket)]
    allow_buy = (1.0 - buy) <= q90
    allow_sell = (1.0 - sell) <= q90
    ambiguous = allow_buy == allow_sell
    buy = torch.where(ambiguous, buy * 0.50, buy)
    sell = torch.where(ambiguous, sell * 0.50, sell)
    skip = torch.where(ambiguous, torch.maximum(skip, torch.full_like(skip, 0.55)), skip)
    raw = torch.stack([sell, buy, skip], dim=-1)
    raw = torch.clamp(raw, 0.0005, 0.9990)
    raw = raw / torch.sum(raw, dim=-1, keepdim=True)
    minimum_move = max(float(min_move_points), 0.10)
    calibrator_features = torch.stack(
        [
            torch.ones_like(expected_move),
            torch.clamp(raw[:, 1] - raw[:, 0], -1.0, 1.0),
            torch.clamp(raw[:, 2], 0.0, 1.0),
            torch.clamp(expected_move / minimum_move, 0.0, 12.0),
            torch.clamp(torch.full_like(expected_move, max(float(price_cost_points), 0.0) / minimum_move), 0.0, 4.0),
        ],
        dim=-1,
    )
    calibrated_logits = torch.log(torch.clamp(raw, min=1.0e-6)) + calibrator_features @ state.calibration_weights.T
    probabilities = torch.softmax(calibrated_logits, dim=-1)
    if state.model is not None:
        state.model.eval()
        with torch.no_grad():
            neural = state.model(regime, model_features)
        probabilities = torch.clamp(0.72 * probabilities + 0.28 * neural["class_probabilities"], 0.0005, 0.9990)
        probabilities = probabilities / torch.sum(probabilities, dim=-1, keepdim=True)
        expected_move = 0.72 * expected_move + 0.28 * neural["expected_move_points"]
        gates = torch.clamp(0.72 * gates + 0.28 * neural["expert_gates"], 0.0005, 0.9995)
        gates = gates / torch.sum(gates, dim=-1, keepdim=True)
    return {
        "class_probabilities": probabilities,
        "raw_probabilities": raw,
        "expected_move_points": expected_move,
        "expert_gates": gates,
    }


def train_step(
    features: Iterable[Iterable[float]] | torch.Tensor,
    labels: Iterable[int],
    moves: Iterable[float],
    state: MoeConformalTorchState | None = None,
    lr: float = 0.025,
    session_bucket: int = 0,
    data_has_volume: bool = True,
) -> MoeConformalTorchState:
    state = state or MoeConformalTorchState.seeded()
    device = state.router.device
    x = prepare_features(features, data_has_volume=data_has_volume).to(device)
    regime = build_regime(x, session_bucket=session_bucket)
    model_features = build_model_features(x)
    head_features = torch.cat(
        [torch.ones((x.shape[0], 1), dtype=torch.float32, device=device), model_features],
        dim=-1,
    )
    prediction = predict_batch(x, session_bucket=session_bucket, data_has_volume=data_has_volume, state=state)
    probabilities = prediction["class_probabilities"]
    gates = prediction["expert_gates"]
    raw_labels = list(labels)
    if not raw_labels:
        raw_labels = [2] * x.shape[0]
    target = torch.tensor(raw_labels[: x.shape[0]], dtype=torch.long, device=device).clamp(0, CLASSES - 1)
    moves_tensor = torch.tensor(list(moves) or [1.0] * x.shape[0], dtype=torch.float32, device=device).abs().clamp_min(0.10)
    if moves_tensor.numel() < x.shape[0]:
        moves_tensor = torch.cat([moves_tensor, moves_tensor[-1:].repeat(x.shape[0] - moves_tensor.numel())])
    one_hot = torch.nn.functional.one_hot(target, num_classes=CLASSES).to(dtype=torch.float32)
    error = one_hot - probabilities
    trade_target = 1.0 - one_hot[:, 2]
    trade_prediction = torch.clamp(1.0 - probabilities[:, 2], 0.001, 0.999)
    direction_target = torch.where(one_hot[:, 0] > 0.5, torch.zeros_like(trade_target), torch.ones_like(trade_target))
    direction_error = torch.where(trade_target > 0.0, direction_target - probabilities[:, 1], torch.zeros_like(trade_target))
    trade_error = trade_target - trade_prediction
    expert_credit = gates * (0.5 + 0.5 * torch.abs(trade_error).unsqueeze(-1))
    state.router += lr * torch.mean(expert_credit.unsqueeze(-1) * trade_error.view(-1, 1, 1) * regime[:, None, :], dim=0)
    state.gate_weights += lr * torch.mean(expert_credit.unsqueeze(-1) * trade_error.view(-1, 1, 1) * head_features[:, None, :], dim=0)
    state.direction_weights += lr * torch.mean(
        expert_credit.unsqueeze(-1)
        * direction_error.view(-1, 1, 1)
        * moves_tensor.clamp(max=10.0).view(-1, 1, 1)
        * head_features[:, None, :],
        dim=0,
    )
    expected_move = prediction["expected_move_points"]
    move_error = (moves_tensor - expected_move).clamp(-10.0, 10.0)
    state.move_weights += lr * 0.10 * torch.mean(expert_credit.unsqueeze(-1) * move_error.view(-1, 1, 1) * head_features[:, None, :], dim=0)
    calibrator_features = torch.stack(
        [
            torch.ones_like(expected_move),
            torch.clamp(probabilities[:, 1] - probabilities[:, 0], -1.0, 1.0),
            torch.clamp(probabilities[:, 2], 0.0, 1.0),
            torch.clamp(expected_move / torch.clamp(moves_tensor, min=0.10), 0.0, 12.0),
            torch.ones_like(expected_move),
        ],
        dim=-1,
    )
    state.calibration_weights += lr * 0.20 * torch.mean(error.unsqueeze(-1) * calibrator_features[:, None, :], dim=0)
    state.usage_ema = 0.98 * state.usage_ema + 0.02 * torch.mean(gates, dim=0)
    buckets = bucket_index(x, session_bucket=session_bucket)
    nonconformity = 1.0 - torch.sum(probabilities * one_hot, dim=-1)
    for bucket in torch.unique(buckets):
        mask = buckets == bucket
        if torch.any(mask):
            observed = torch.quantile(nonconformity[mask].detach(), 0.90)
            state.bucket_quantiles90[bucket] = 0.98 * state.bucket_quantiles90[bucket] + 0.02 * observed
    if state.model is not None and state.optimizer is not None:
        state.model.train()
        neural = state.model(regime.detach(), model_features.detach())
        neural_probs = neural["class_probabilities"].clamp_min(1.0e-6)
        neural_nonconformity = 1.0 - torch.sum(neural_probs * one_hot, dim=-1)
        cutoff = split_conformal_cutoff(neural_nonconformity.detach(), alpha=0.10)
        loss = F.nll_loss(torch.log(neural_probs), target)
        loss = loss + 0.05 * F.smooth_l1_loss(neural["expected_move_points"], moves_tensor)
        loss = loss + 0.03 * torch.relu(neural_nonconformity.mean() - cutoff)
        loss = loss + 0.02 * load_balance_loss(neural["expert_gates"])
        state.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), 1.0)
        state.optimizer.step()
    return state
