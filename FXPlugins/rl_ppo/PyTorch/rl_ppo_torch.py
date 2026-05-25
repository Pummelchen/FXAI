"""PyTorch/MPS reference PPO backend for FXAI rl_ppo."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F

PLUGIN_NAME = "rl_ppo"
ARCHITECTURE_MODE = "proximalPolicyOptimization"
FEATURE_COUNT = 32
ACTION_COUNT = 3
HIDDEN_COUNT = 64
VOLUME_FEATURE_INDEXES = (6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83)


def preferred_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _features(
    batch: Iterable[Iterable[float]] | torch.Tensor,
    device: torch.device,
    data_has_volume: bool = True,
) -> torch.Tensor:
    x = (
        batch.detach().to(device=device, dtype=torch.float32)
        if isinstance(batch, torch.Tensor)
        else torch.tensor(list(batch), dtype=torch.float32, device=device)
    )
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim == 3:
        x = x[:, -1, :]
    if x.shape[-1] < FEATURE_COUNT:
        x = torch.cat(
            [x, torch.zeros((x.shape[0], FEATURE_COUNT - x.shape[-1]), dtype=x.dtype, device=x.device)],
            dim=-1,
        )
    x = x[..., :FEATURE_COUNT].clamp(-8.0, 8.0)
    if not data_has_volume:
        valid = [index for index in VOLUME_FEATURE_INDEXES if index < x.shape[-1]]
        if valid:
            x = x.clone()
            x[..., valid] = 0.0
    return x


class ActorCriticPPO(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(FEATURE_COUNT),
            nn.Linear(FEATURE_COUNT, HIDDEN_COUNT),
            nn.Tanh(),
            nn.Linear(HIDDEN_COUNT, HIDDEN_COUNT),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(HIDDEN_COUNT, ACTION_COUNT)
        self.value_head = nn.Linear(HIDDEN_COUNT, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        encoded = torch.nan_to_num(self.encoder(observations), nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
        logits = torch.nan_to_num(self.policy_head(encoded), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        value = torch.nan_to_num(self.value_head(encoded).squeeze(-1), nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        return torch.distributions.Categorical(logits=logits), value

    def act(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.forward(observations)
        action = distribution.sample()
        return action, distribution.log_prob(action), value


@dataclass
class RolloutBuffer:
    observations: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    old_log_probs: list[torch.Tensor] = field(default_factory=list)
    rewards: list[torch.Tensor] = field(default_factory=list)
    dones: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)

    def append(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        old_log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.observations.append(observation.detach())
        self.actions.append(action.detach())
        self.old_log_probs.append(old_log_prob.detach())
        self.rewards.append(reward.detach())
        self.dones.append(done.detach())
        self.values.append(value.detach())

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.cat(self.observations),
            torch.cat(self.actions).long(),
            torch.cat(self.old_log_probs),
            torch.cat(self.rewards),
            torch.cat(self.dones),
            torch.cat(self.values),
        )

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.old_log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    @property
    def is_empty(self) -> bool:
        return not self.observations


class OfflineFXRolloutEnvironment:
    """Minimal offline FX reward model for PPO rollouts.

    The action convention follows FXDataEngine labels: sell=0, buy=1, skip=2.
    Signed move points are positive for upward price moves and negative for
    downward price moves. The reward is net of a symmetric transaction cost.
    """

    def __init__(self, transaction_cost_points: float = 0.0, reward_scale: float = 1.0) -> None:
        self.transaction_cost_points = max(0.0, float(transaction_cost_points))
        self.reward_scale = max(1.0e-9, float(reward_scale))

    def reward(self, signed_move_points: float, action: int) -> float:
        if action == 1:
            return (float(signed_move_points) - self.transaction_cost_points) / self.reward_scale
        if action == 0:
            return (-float(signed_move_points) - self.transaction_cost_points) / self.reward_scale
        return 0.0


def append_offline_rollout(
    state: "RlPPOReferenceState",
    batch: Iterable[Iterable[float]] | torch.Tensor,
    signed_moves: Iterable[float],
    data_has_volume: bool = True,
    transaction_cost_points: float = 0.0,
) -> RlPPOReferenceState:
    device = next(state.model.parameters()).device
    observations = _features(batch, device, data_has_volume=data_has_volume)
    environment = OfflineFXRolloutEnvironment(transaction_cost_points=transaction_cost_points)
    moves = list(signed_moves)
    if len(moves) < observations.shape[0]:
        moves.extend([0.0] * (observations.shape[0] - len(moves)))
    with torch.no_grad():
        distribution, values = state.model(observations)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
    rewards = torch.tensor(
        [environment.reward(move, int(action.item())) for move, action in zip(moves[: observations.shape[0]], actions)],
        dtype=torch.float32,
        device=device,
    )
    dones = torch.zeros_like(rewards)
    if dones.numel() > 0:
        dones[-1] = 1.0
    state.rollout.append(observations, actions, log_probs, rewards, dones, values)
    return state


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    next_value = torch.zeros((), dtype=values.dtype, device=values.device)
    for step in reversed(range(rewards.shape[0])):
        mask = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * mask - values[step]
        gae = delta + gamma * lam * mask * gae
        advantages[step] = gae
        next_value = values[step]
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)
    return advantages, returns


def ppo_clipped_loss(
    model: ActorCriticPPO,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float = 0.2,
    entropy_weight: float = 0.01,
    value_weight: float = 0.5,
) -> torch.Tensor:
    distribution, values = model(observations)
    log_probs = distribution.log_prob(actions)
    ratio = torch.exp(log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = F.smooth_l1_loss(values, returns)
    entropy_loss = -distribution.entropy().mean()
    return policy_loss + value_weight * value_loss + entropy_weight * entropy_loss


def _sanitize_gradients_and_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad = torch.nan_to_num(parameter.grad, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.nan_to_num(parameter, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0))


@dataclass
class RlPPOReferenceState:
    model: ActorCriticPPO
    optimizer: torch.optim.Optimizer
    rollout: RolloutBuffer

    @classmethod
    def create(cls, device: Optional[torch.device] = None, lr: float = 3.0e-4) -> "RlPPOReferenceState":
        dev = device or preferred_device()
        model = ActorCriticPPO().to(dev)
        return cls(model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=lr), rollout=RolloutBuffer())


def predict_batch(
    batch: Iterable[Iterable[float]] | torch.Tensor,
    state: Optional[RlPPOReferenceState] = None,
    data_has_volume: bool = True,
) -> dict[str, list[list[float]] | list[float] | str]:
    state = state or RlPPOReferenceState.create()
    state.model.eval()
    device = next(state.model.parameters()).device
    observations = _features(batch, device, data_has_volume=data_has_volume)
    with torch.no_grad():
        distribution, value = state.model(observations)
    return {
        "plugin": PLUGIN_NAME,
        "architecture": ARCHITECTURE_MODE,
        "device": str(device),
        "class_probabilities": distribution.probs.cpu().tolist(),
        "state_value": value.cpu().tolist(),
    }


def train_step(
    batch: Iterable[Iterable[float]] | torch.Tensor,
    labels: Iterable[int],
    moves: Iterable[float],
    state: Optional[RlPPOReferenceState] = None,
    lr: float = 3.0e-4,
    data_has_volume: bool = True,
) -> RlPPOReferenceState:
    state = state or RlPPOReferenceState.create(lr=lr)
    device = next(state.model.parameters()).device
    observations = _features(batch, device, data_has_volume=data_has_volume)
    label_values = list(labels) or [2] * observations.shape[0]
    move_values = list(moves) or [0.0] * observations.shape[0]
    actions = torch.tensor(label_values[: observations.shape[0]], dtype=torch.long, device=device).clamp(0, ACTION_COUNT - 1)
    rewards = torch.tensor([abs(float(value)) for value in move_values[: observations.shape[0]]], dtype=torch.float32, device=device)
    old_distribution, old_values = state.model(observations)
    old_log_probs = old_distribution.log_prob(actions).detach()
    dones = torch.zeros_like(rewards)
    if dones.numel() > 0:
        dones[-1] = 1.0
    state.rollout.append(observations, actions, old_log_probs, rewards, dones, old_values.detach())
    rollout_observations, rollout_actions, rollout_log_probs, rollout_rewards, rollout_dones, rollout_values = state.rollout.tensors()
    advantages, returns = compute_gae(rollout_rewards, rollout_values, rollout_dones)
    state.optimizer.zero_grad(set_to_none=True)
    loss = ppo_clipped_loss(state.model, rollout_observations, rollout_actions, rollout_log_probs, advantages, returns)
    loss.backward()
    _sanitize_gradients_and_parameters(state.model)
    torch.nn.utils.clip_grad_norm_(state.model.parameters(), 1.0)
    state.optimizer.step()
    _sanitize_gradients_and_parameters(state.model)
    state.rollout.clear()
    return state
