from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


def preferred_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DemoPluginTemplateTorch(nn.Module):
    """PyTorch/MPS template module with no trade logic."""

    def __init__(self, feature_count: int = 16) -> None:
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
    def seeded(cls, feature_count: int = 16, device: torch.device | None = None) -> "DemoPluginTemplateTorchState":
        resolved = device or preferred_device()
        model = DemoPluginTemplateTorch(feature_count=feature_count).to(resolved)
        return cls(model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4))


def prepare_features(features: Iterable[Iterable[float]] | torch.Tensor, feature_count: int = 16) -> torch.Tensor:
    x = torch.as_tensor(features, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.shape[-1] < feature_count:
        x = torch.nn.functional.pad(x, (0, feature_count - x.shape[-1]))
    return x[:, :feature_count]


def predict_batch(
    state: DemoPluginTemplateTorchState,
    features: Iterable[Iterable[float]] | torch.Tensor,
) -> dict[str, torch.Tensor]:
    x = prepare_features(features, feature_count=state.model.encoder[0].normalized_shape[0]).to(next(state.model.parameters()).device)
    state.model.eval()
    with torch.no_grad():
        return state.model(x)


def train_step(
    state: DemoPluginTemplateTorchState,
    features: Iterable[Iterable[float]] | torch.Tensor,
    labels: Iterable[int] | torch.Tensor,
) -> DemoPluginTemplateTorchState:
    x = prepare_features(features, feature_count=state.model.encoder[0].normalized_shape[0]).to(next(state.model.parameters()).device)
    target = torch.as_tensor(labels, dtype=torch.long, device=x.device).clamp(0, 2)
    state.model.train()
    probabilities = state.model(x)["class_probabilities"].clamp_min(1.0e-6)
    loss = torch.nn.functional.nll_loss(torch.log(probabilities), target)
    state.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(state.model.parameters(), 1.0)
    state.optimizer.step()
    return state
