from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn

FEATURE_COUNT = 16
VOLUME_FEATURE_INDEXES = (6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83)


def preferred_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DemoPluginTemplateTorch(nn.Module):
    """PyTorch/MPS template module with no trade logic."""

    def __init__(self, feature_count: int = FEATURE_COUNT) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Linear(feature_count, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
        )
        self.head = nn.Linear(16, 3)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(features)
        logits = self.head(encoded)
        template_skip_bias = torch.tensor([0.0, 0.0, 8.0], dtype=logits.dtype, device=logits.device)
        return {"class_probabilities": torch.softmax(logits * 0.0 + template_skip_bias, dim=-1)}


@dataclass
class DemoPluginTemplateTorchState:
    model: DemoPluginTemplateTorch
    optimizer: torch.optim.Optimizer

    @classmethod
    def seeded(cls, feature_count: int = FEATURE_COUNT, device: torch.device | None = None) -> "DemoPluginTemplateTorchState":
        resolved = device or preferred_device()
        model = DemoPluginTemplateTorch(feature_count=feature_count).to(resolved)
        return cls(model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4))

    @classmethod
    def create(cls, device: torch.device | None = None, lr: float = 3.0e-4) -> "DemoPluginTemplateTorchState":
        resolved = device or preferred_device()
        model = DemoPluginTemplateTorch(feature_count=FEATURE_COUNT).to(resolved)
        return cls(model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4))


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for values in optimizer.state.values():
        for key, value in list(values.items()):
            if isinstance(value, torch.Tensor):
                values[key] = value.to(device)


def _state_on_preferred_device(state: DemoPluginTemplateTorchState) -> DemoPluginTemplateTorchState:
    device = preferred_device()
    state.model.to(device)
    _move_optimizer_state(state.optimizer, device)
    return state


def prepare_features(
    features: Iterable[Iterable[float]] | torch.Tensor,
    feature_count: int = FEATURE_COUNT,
    data_has_volume: bool = True,
) -> torch.Tensor:
    x = torch.as_tensor(features, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim == 3:
        x = x[:, -1, :]
    if x.shape[-1] < feature_count:
        x = torch.nn.functional.pad(x, (0, feature_count - x.shape[-1]))
    x = torch.nan_to_num(x[:, :feature_count], nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
    if not data_has_volume:
        valid = [index for index in VOLUME_FEATURE_INDEXES if index < feature_count]
        if valid:
            x = x.clone()
            x[..., valid] = 0.0
    return x


def predict_batch(
    features: Iterable[Iterable[float]] | torch.Tensor,
    state: Optional[DemoPluginTemplateTorchState] = None,
    data_has_volume: bool = True,
) -> dict[str, list[list[float]] | list[float]]:
    state = _state_on_preferred_device(state or DemoPluginTemplateTorchState.create())
    feature_count = int(state.model.encoder[0].normalized_shape[0])
    x = prepare_features(features, feature_count=feature_count, data_has_volume=data_has_volume)
    x = x.to(next(state.model.parameters()).device)
    state.model.eval()
    with torch.no_grad():
        probabilities = state.model(x)["class_probabilities"]
    batch_size = int(probabilities.shape[0])
    return {
        "class_probabilities": probabilities.detach().cpu().tolist(),
        "move_mean_points": [0.0] * batch_size,
        "move_quantiles": [[0.0, 0.0, 0.0] for _ in range(batch_size)],
    }


def train_step(
    features: Iterable[Iterable[float]] | torch.Tensor,
    labels: Iterable[int] | torch.Tensor,
    moves: Iterable[float] | torch.Tensor,
    state: Optional[DemoPluginTemplateTorchState] = None,
    lr: float = 3.0e-4,
    data_has_volume: bool = True,
) -> DemoPluginTemplateTorchState:
    del labels, moves, lr
    state = _state_on_preferred_device(state or DemoPluginTemplateTorchState.create())
    _ = predict_batch(features, state=state, data_has_volume=data_has_volume)
    return state
