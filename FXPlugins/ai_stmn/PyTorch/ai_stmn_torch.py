from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

PLUGIN_NAME = "ai_stmn"
ARCHITECTURE_MODE = "stmn"
FEATURE_COUNT = 32
HIDDEN_COUNT = 18
CLASS_COUNT = 3


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _features(batch: Iterable[Iterable[float]], device: Optional[torch.device] = None) -> torch.Tensor:
    rows = []
    for row in batch:
        values = list(row)[:FEATURE_COUNT]
        if len(values) < FEATURE_COUNT:
            values.extend([0.0] * (FEATURE_COUNT - len(values)))
        rows.append(values)
    if not rows:
        rows = [[0.0] * FEATURE_COUNT]
    return torch.tensor(rows, dtype=torch.float32, device=device or _device()).clamp(-50.0, 50.0)


@dataclass
class AIStmnTorchState:
    input_weight: torch.Tensor
    class_weight: torch.Tensor
    move_weight: torch.Tensor

    @classmethod
    def create(cls, device: Optional[torch.device] = None) -> "AIStmnTorchState":
        dev = device or _device()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(10_000 + 15)
        input_weight = torch.randn(FEATURE_COUNT, HIDDEN_COUNT, generator=generator, dtype=torch.float32).to(dev) * 0.05
        class_weight = torch.randn(HIDDEN_COUNT, CLASS_COUNT, generator=generator, dtype=torch.float32).to(dev) * 0.04
        move_weight = torch.rand(HIDDEN_COUNT, 1, generator=generator, dtype=torch.float32).to(dev) * 0.03
        return cls(input_weight=input_weight, class_weight=class_weight, move_weight=move_weight)


def encode(batch: Iterable[Iterable[float]], state: Optional[AIStmnTorchState] = None) -> torch.Tensor:
    state = state or AIStmnTorchState.create()
    x = _features(batch, state.input_weight.device)
    hidden = torch.tanh(x @ state.input_weight)
    if ARCHITECTURE_MODE in {"tcn", "cnnLSTM", "attentionCNNBiLSTM", "lstmTCN"}:
        hidden = 0.7 * hidden + 0.3 * torch.roll(hidden, shifts=1, dims=1)
    elif ARCHITECTURE_MODE in {"transformer", "temporalFusionTransformer", "autoformer", "patchTransformer", "causalTokenForecaster", "foundationForecaster", "geodesicAttention", "mythosRDT"}:
        attn = torch.softmax(hidden, dim=1)
        hidden = torch.tanh(hidden + attn * hidden.mean(dim=1, keepdim=True))
    elif ARCHITECTURE_MODE in {"s4", "stmn", "fewc", "gha", "tensorTesseract"}:
        hidden = torch.tanh(0.85 * hidden + 0.15 * torch.cumsum(hidden, dim=1) / max(HIDDEN_COUNT, 1))
    return hidden


def predict_batch(batch: Iterable[Iterable[float]], state: Optional[AIStmnTorchState] = None) -> dict[str, list[list[float]] | list[float]]:
    state = state or AIStmnTorchState.create()
    with torch.no_grad():
        hidden = encode(batch, state)
        logits = hidden @ state.class_weight
        probabilities = torch.softmax(logits, dim=1)
        move = torch.relu(hidden @ state.move_weight).squeeze(-1)
    return {
        "plugin": PLUGIN_NAME,
        "device": str(state.input_weight.device),
        "class_probabilities": probabilities.detach().cpu().tolist(),
        "move_mean_points": move.detach().cpu().tolist(),
    }


def train_step(batch: Iterable[Iterable[float]], labels: Iterable[int], moves: Iterable[float], state: Optional[AIStmnTorchState] = None, lr: float = 0.01) -> AIStmnTorchState:
    state = state or AIStmnTorchState.create()
    x = _features(batch, state.input_weight.device)
    y = torch.tensor(list(labels), dtype=torch.long, device=state.input_weight.device).clamp(0, CLASS_COUNT - 1)
    m = torch.tensor(list(moves), dtype=torch.float32, device=state.input_weight.device).abs()
    state.input_weight.requires_grad_(True)
    state.class_weight.requires_grad_(True)
    state.move_weight.requires_grad_(True)
    hidden = torch.tanh(x @ state.input_weight)
    logits = hidden @ state.class_weight
    move_pred = torch.relu(hidden @ state.move_weight).squeeze(-1)
    loss = torch.nn.functional.cross_entropy(logits, y) + 0.05 * torch.nn.functional.smooth_l1_loss(move_pred, m)
    loss.backward()
    with torch.no_grad():
        for tensor in (state.input_weight, state.class_weight, state.move_weight):
            tensor -= lr * tensor.grad.clamp(-1.0, 1.0)
            tensor.grad = None
    return state
