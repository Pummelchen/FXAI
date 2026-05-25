"""PyTorch/MPS reference backend for FXAI ai_tst.

This plugin-local backend implements a time-series transformer encoder with real PyTorch layer
families. Swift CPU remains the deterministic offline fallback; this module is
the reference accelerator path for training and batched inference on Apple Silicon.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F

PLUGIN_NAME = "ai_tst"
ARCHITECTURE_MODE = "TST"
FEATURE_COUNT = 32
CLASS_COUNT = 3
HIDDEN_COUNT = 48
HEAD_COUNT = 4
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
VOLUME_FEATURE_INDEXES = (6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83)


def preferred_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_sequence(batch: Iterable[Iterable[float]] | torch.Tensor, device: Optional[torch.device] = None, data_has_volume: bool = True) -> torch.Tensor:
    dev = device or preferred_device()
    if isinstance(batch, torch.Tensor):
        x = batch.detach().to(device=dev, dtype=torch.float32)
    else:
        x = torch.tensor(list(batch), dtype=torch.float32, device=dev)
    if x.ndim == 1:
        x = x.view(1, 1, -1)
    elif x.ndim == 2:
        x = x.unsqueeze(1)
    elif x.ndim != 3:
        raise ValueError("batch must be a vector, [batch, feature], or [batch, sequence, feature]")
    if x.shape[-1] < FEATURE_COUNT:
        padding = torch.zeros((*x.shape[:-1], FEATURE_COUNT - x.shape[-1]), dtype=x.dtype, device=x.device)
        x = torch.cat([x, padding], dim=-1)
    x = torch.nan_to_num(x[..., :FEATURE_COUNT], nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
    if not data_has_volume:
        valid = [idx for idx in VOLUME_FEATURE_INDEXES if idx < x.shape[-1]]
        if valid:
            x = x.clone()
            x[..., valid] = 0.0
    if x.device.type == "mps" and x.shape[1] < 3:
        prefix = x[:, :1, :].expand(-1, 3 - x.shape[1], -1).clone()
        x = torch.cat([prefix, x], dim=1).contiguous()
    return x


def _targets(labels: Iterable[int], batch_size: int, device: torch.device) -> torch.Tensor:
    raw = list(labels)
    if not raw:
        raw = [2] * batch_size
    if len(raw) < batch_size:
        raw.extend([raw[-1]] * (batch_size - len(raw)))
    return torch.tensor(raw[:batch_size], dtype=torch.long, device=device).clamp(0, CLASS_COUNT - 1)


def _moves(moves: Iterable[float], batch_size: int, device: torch.device) -> torch.Tensor:
    raw = [abs(float(value)) for value in moves]
    if not raw:
        raw = [1.0] * batch_size
    if len(raw) < batch_size:
        raw.extend([raw[-1]] * (batch_size - len(raw)))
    return torch.tensor(raw[:batch_size], dtype=torch.float32, device=device).clamp_min(0.0)


def _pinball_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    quantiles = torch.tensor(QUANTILES, dtype=prediction.dtype, device=prediction.device).view(1, -1)
    error = target.view(-1, 1) - prediction
    return torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()


class PredictionHeads(nn.Module):
    def __init__(self, hidden_count: int = HIDDEN_COUNT) -> None:
        super().__init__()
        self.class_head = nn.Sequential(nn.LayerNorm(hidden_count), nn.Linear(hidden_count, hidden_count), nn.GELU(), nn.Linear(hidden_count, CLASS_COUNT))
        self.move_head = nn.Sequential(nn.LayerNorm(hidden_count), nn.Linear(hidden_count, 1), nn.Softplus())
        self.quantile_head = nn.Sequential(nn.LayerNorm(hidden_count), nn.Linear(hidden_count, len(QUANTILES)), nn.Softplus())

    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.class_head(encoded)
        move = self.move_head(encoded).squeeze(-1)
        quantiles = torch.cumsum(self.quantile_head(encoded), dim=-1)
        return logits, move, quantiles


class CausalTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=self.padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=self.padding, dilation=dilation)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(0.05)

    def _chomp(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :-self.padding] if self.padding > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._chomp(self.conv1(x))
        y = self.dropout(F.gelu(self.norm(y)))
        y = self._chomp(self.conv2(y))
        return F.gelu(x + y)


class GatedResidualNetwork(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.gate = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        candidate = self.fc2(F.elu(self.fc1(x)))
        return self.norm(x + torch.sigmoid(self.gate(x)) * candidate)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, feature_count: int = FEATURE_COUNT, hidden_count: int = HIDDEN_COUNT) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(1, hidden_count)
        self.weight_projection = nn.Linear(feature_count, feature_count)
        self.grn = GatedResidualNetwork(hidden_count)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.weight_projection(x), dim=-1)
        projected = self.feature_projection(x.unsqueeze(-1))
        return self.grn(torch.sum(projected * weights.unsqueeze(-1), dim=-2)), weights


class AutoCorrelation(nn.Module):
    def __init__(self, width: int = HIDDEN_COUNT, max_lag: int = 12) -> None:
        super().__init__()
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.max_lag = max_lag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        lag_count = max(1, min(self.max_lag, x.shape[1]))
        correlations = []
        shifted_values = []
        for lag in range(lag_count):
            shifted_k = torch.roll(k, shifts=-lag, dims=1)
            shifted_v = torch.roll(v, shifts=-lag, dims=1)
            correlations.append((q * shifted_k).mean(dim=(1, 2)))
            shifted_values.append(shifted_v)
        weights = torch.softmax(torch.stack(correlations, dim=-1), dim=-1)
        values = torch.stack(shifted_values, dim=1)
        return torch.sum(values * weights.view(x.shape[0], lag_count, 1, 1), dim=1)


class S4DLayer(nn.Module):
    def __init__(self, width: int = HIDDEN_COUNT, max_kernel: int = 128) -> None:
        super().__init__()
        self.log_a = nn.Parameter(torch.linspace(-2.5, -0.1, width))
        self.c = nn.Parameter(torch.randn(width) * 0.05)
        self.d = nn.Parameter(torch.ones(width) * 0.05)
        self.max_kernel = max_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, width = x.shape
        length = min(steps, self.max_kernel)
        time = torch.arange(length, dtype=x.dtype, device=x.device)
        kernel = torch.exp(-torch.exp(self.log_a).to(x.device).unsqueeze(-1) * time.unsqueeze(0)) * self.c.to(x.device).unsqueeze(-1)
        y = F.conv1d(x.transpose(1, 2), kernel.flip(-1).unsqueeze(1), padding=length - 1, groups=width)
        return y[..., :steps].transpose(1, 2) + self.d.to(x.device) * x


class GraphMessagePassingLayer(nn.Module):
    def __init__(self, node_count: int, width: int) -> None:
        super().__init__()
        self.edge_logits = nn.Parameter(torch.zeros(node_count, node_count))
        self.message = nn.Linear(width, width)
        self.update = nn.GRUCell(width, width)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        adjacency = torch.softmax(self.edge_logits, dim=-1)
        message = adjacency @ self.message(nodes)
        return self.update(message.reshape(-1, message.shape[-1]), nodes.reshape(-1, nodes.shape[-1])).view_as(nodes)


class AITSTReferenceModel(nn.Module):
    """Reference-grade time-series transformer encoder."""
    def __init__(self) -> None:
        super().__init__()
        self.architecture = ARCHITECTURE_MODE
        self.heads = PredictionHeads()
        if self.architecture == "MLP":
            self.encoder = nn.Sequential(nn.LayerNorm(FEATURE_COUNT), nn.Linear(FEATURE_COUNT, HIDDEN_COUNT), nn.GELU(), nn.Dropout(0.05), nn.Linear(HIDDEN_COUNT, HIDDEN_COUNT), nn.GELU(), nn.LayerNorm(HIDDEN_COUNT))
        elif self.architecture in {"LSTM", "LSTMG"}:
            self.input_norm = nn.LayerNorm(FEATURE_COUNT)
            self.lstm = nn.LSTM(FEATURE_COUNT, HIDDEN_COUNT, num_layers=2, batch_first=True, dropout=0.05)
            self.feature_residual = nn.Linear(FEATURE_COUNT, HIDDEN_COUNT)
            self.gate = nn.Sequential(nn.Linear(FEATURE_COUNT + HIDDEN_COUNT, HIDDEN_COUNT), nn.Sigmoid())
        elif self.architecture == "GRU":
            self.input_norm = nn.LayerNorm(FEATURE_COUNT)
            self.gru = nn.GRU(FEATURE_COUNT, HIDDEN_COUNT, num_layers=2, batch_first=True, dropout=0.05)
        elif self.architecture == "BILSTM":
            self.input_norm = nn.LayerNorm(FEATURE_COUNT)
            self.bilstm = nn.LSTM(FEATURE_COUNT, HIDDEN_COUNT // 2, num_layers=2, batch_first=True, dropout=0.05, bidirectional=True)
            self.merge = GatedResidualNetwork(HIDDEN_COUNT)
        elif self.architecture == "LSTM_TCN":
            self.input_norm = nn.LayerNorm(FEATURE_COUNT)
            self.lstm = nn.LSTM(FEATURE_COUNT, HIDDEN_COUNT, batch_first=True)
            self.tcn = nn.Sequential(CausalTCNBlock(HIDDEN_COUNT, 1), CausalTCNBlock(HIDDEN_COUNT, 2), CausalTCNBlock(HIDDEN_COUNT, 4))
        elif self.architecture == "CNN_LSTM":
            self.conv1 = nn.Conv1d(FEATURE_COUNT, HIDDEN_COUNT, 3, padding=1)
            self.conv2 = nn.Conv1d(HIDDEN_COUNT, HIDDEN_COUNT, 3, padding=1)
            self.lstm = nn.LSTM(HIDDEN_COUNT, HIDDEN_COUNT, batch_first=True)
        elif self.architecture == "ATTN_CNN_BILSTM":
            self.conv1 = nn.Conv1d(FEATURE_COUNT, HIDDEN_COUNT, 3, padding=1)
            self.conv2 = nn.Conv1d(HIDDEN_COUNT, HIDDEN_COUNT, 3, padding=1)
            self.bilstm = nn.LSTM(HIDDEN_COUNT, HIDDEN_COUNT // 2, batch_first=True, bidirectional=True)
            self.attention = nn.MultiheadAttention(HIDDEN_COUNT, HEAD_COUNT, batch_first=True)
        elif self.architecture == "TCN":
            self.input_projection = nn.Conv1d(FEATURE_COUNT, HIDDEN_COUNT, 1)
            self.tcn = nn.Sequential(CausalTCNBlock(HIDDEN_COUNT, 1), CausalTCNBlock(HIDDEN_COUNT, 2), CausalTCNBlock(HIDDEN_COUNT, 4), CausalTCNBlock(HIDDEN_COUNT, 8))
        elif self.architecture == "TST":
            self.input_projection = nn.Linear(FEATURE_COUNT, HIDDEN_COUNT)
            self.position = nn.Parameter(torch.zeros(1, 128, HIDDEN_COUNT))
            layer = nn.TransformerEncoderLayer(d_model=HIDDEN_COUNT, nhead=HEAD_COUNT, dim_feedforward=HIDDEN_COUNT * 4, dropout=0.05, batch_first=True, activation="gelu")
            self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        elif self.architecture == "TFT":
            self.variable_selection = VariableSelectionNetwork()
            self.static_grn = GatedResidualNetwork(HIDDEN_COUNT)
            self.lstm_encoder = nn.LSTM(HIDDEN_COUNT, HIDDEN_COUNT, batch_first=True)
            self.interpretable_attention = nn.MultiheadAttention(HIDDEN_COUNT, HEAD_COUNT, batch_first=True)
            self.post_attention_grn = GatedResidualNetwork(HIDDEN_COUNT)
        elif self.architecture == "AUTOFORMER":
            self.input_projection = nn.Linear(FEATURE_COUNT, HIDDEN_COUNT)
            self.autocorrelation = AutoCorrelation(HIDDEN_COUNT)
            self.trend_projection = nn.Linear(FEATURE_COUNT, HIDDEN_COUNT)
            self.grn = GatedResidualNetwork(HIDDEN_COUNT)
        elif self.architecture in {"PATCHTST", "TIMESFM"}:
            self.patch_size = 16 if self.architecture == "TIMESFM" else 8
            self.stride = 8 if self.architecture == "TIMESFM" else 4
            self.patch_embedding = nn.Linear(FEATURE_COUNT * self.patch_size, HIDDEN_COUNT)
            self.position = nn.Parameter(torch.zeros(1, 128, HIDDEN_COUNT))
            layer = nn.TransformerEncoderLayer(d_model=HIDDEN_COUNT, nhead=HEAD_COUNT, dim_feedforward=HIDDEN_COUNT * 4, batch_first=True, activation="gelu")
            self.foundation_encoder = nn.TransformerEncoder(layer, num_layers=5 if self.architecture == "TIMESFM" else 3)
            self.horizon_quantiles = nn.Linear(HIDDEN_COUNT, 8 * len(QUANTILES))
        elif self.architecture == "S4":
            self.input_projection = nn.Linear(FEATURE_COUNT, HIDDEN_COUNT)
            self.s4_layers = nn.ModuleList([S4DLayer(HIDDEN_COUNT), S4DLayer(HIDDEN_COUNT)])
            self.norm = nn.LayerNorm(HIDDEN_COUNT)
        elif self.architecture == "STMN":
            self.input_projection = nn.Linear(FEATURE_COUNT, HIDDEN_COUNT)
            self.memory_slots = nn.Parameter(torch.randn(12, HIDDEN_COUNT) * 0.03)
            self.read_projection = nn.Linear(HIDDEN_COUNT * 2, HIDDEN_COUNT)
            self.write_cell = nn.GRUCell(HIDDEN_COUNT, HIDDEN_COUNT)
        elif self.architecture == "CHRONOS":
            self.register_buffer("bin_edges", torch.linspace(-4.0, 4.0, 255))
            self.token_embedding = nn.Embedding(256, HIDDEN_COUNT)
            self.position = nn.Parameter(torch.zeros(1, 128, HIDDEN_COUNT))
            layer = nn.TransformerEncoderLayer(d_model=HIDDEN_COUNT, nhead=HEAD_COUNT, dim_feedforward=HIDDEN_COUNT * 4, batch_first=True, activation="gelu")
            self.causal_transformer = nn.TransformerEncoder(layer, num_layers=4)
        elif self.architecture == "FEWC":
            self.experts = nn.ModuleList([nn.Sequential(nn.LayerNorm(FEATURE_COUNT), nn.Linear(FEATURE_COUNT, HIDDEN_COUNT), nn.GELU()) for _ in range(4)])
            self.router = nn.Linear(FEATURE_COUNT, 4)
            self.consolidation_anchor = nn.Parameter(torch.zeros(HIDDEN_COUNT), requires_grad=False)
            self.fisher_diagonal = nn.Parameter(torch.ones(HIDDEN_COUNT) * 0.01, requires_grad=False)
        elif self.architecture == "GEODESIC":
            self.landmarks = nn.Parameter(torch.randn(16, FEATURE_COUNT) * 0.05)
            self.value_projection = nn.Linear(FEATURE_COUNT, HIDDEN_COUNT)
            self.geodesic_projection = nn.Linear(16, HIDDEN_COUNT)
        elif self.architecture == "GHA":
            self.component_logits = nn.Parameter(torch.randn(12, FEATURE_COUNT) * 0.04)
            self.projection = nn.Linear(12, HIDDEN_COUNT)
        elif self.architecture == "QCEW":
            self.encoder = nn.GRU(FEATURE_COUNT, HIDDEN_COUNT, num_layers=2, batch_first=True, dropout=0.05)
            self.context_gate = nn.Linear(HIDDEN_COUNT, HIDDEN_COUNT)
        elif self.architecture == "TESSERACT":
            self.rank = 12
            self.feature_factor = nn.Parameter(torch.randn(FEATURE_COUNT, self.rank) * 0.04)
            self.time_factor = nn.Parameter(torch.randn(128, self.rank) * 0.04)
            self.core = nn.Parameter(torch.randn(self.rank, self.rank, HIDDEN_COUNT) * 0.03)
        elif self.architecture == "TRR":
            self.gru = nn.GRU(FEATURE_COUNT, HIDDEN_COUNT, num_layers=2, batch_first=True, dropout=0.05)
            self.regime_head = nn.Linear(HIDDEN_COUNT, 3)
            self.transition_logits = nn.Parameter(torch.zeros(3, 3))
            self.trend_head = nn.Linear(HIDDEN_COUNT, HIDDEN_COUNT)
            self.reversal_head = nn.Linear(HIDDEN_COUNT, HIDDEN_COUNT)
        elif self.architecture == "MYTHOS_RDT":
            self.state_embedding = nn.Linear(FEATURE_COUNT, HIDDEN_COUNT)
            self.return_embedding = nn.Linear(1, HIDDEN_COUNT)
            self.position = nn.Parameter(torch.zeros(1, 128, HIDDEN_COUNT))
            layer = nn.TransformerEncoderLayer(d_model=HIDDEN_COUNT, nhead=HEAD_COUNT, dim_feedforward=HIDDEN_COUNT * 4, batch_first=True, activation="gelu")
            self.decision_transformer = nn.TransformerEncoder(layer, num_layers=4)
            self.recursive_memory = nn.GRUCell(HIDDEN_COUNT, HIDDEN_COUNT)
        elif self.architecture == "WM_CFX":
            self.currencies = 8
            self.factors = 6
            self.currency_exposure = nn.Parameter(torch.randn(self.currencies, self.factors) * 0.04)
            self.feature_to_currency = nn.Linear(FEATURE_COUNT, self.currencies)
            self.factor_gru = nn.GRU(self.factors, HIDDEN_COUNT, batch_first=True)
            self.cross_rate_decoder = nn.Linear(HIDDEN_COUNT, self.currencies * self.currencies)
        elif self.architecture == "WM_GRAPH":
            self.nodes = 8
            self.node_projection = nn.Linear(FEATURE_COUNT, self.nodes * HIDDEN_COUNT)
            self.message_passing = nn.ModuleList([GraphMessagePassingLayer(self.nodes, HIDDEN_COUNT) for _ in range(3)])
            self.readout = nn.Linear(self.nodes * HIDDEN_COUNT, HIDDEN_COUNT)
        else:
            raise ValueError(f"unsupported architecture {self.architecture}")

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.patch_size:
            pad = torch.zeros((x.shape[0], self.patch_size - x.shape[1], x.shape[2]), dtype=x.dtype, device=x.device)
            x = torch.cat([pad, x], dim=1)
        return x.unfold(1, self.patch_size, self.stride).contiguous().view(x.shape[0], -1, FEATURE_COUNT * self.patch_size)

    def _moving_average(self, x: torch.Tensor, kernel: int = 5) -> torch.Tensor:
        return F.avg_pool1d(x.transpose(1, 2), kernel_size=kernel, stride=1, padding=kernel // 2).transpose(1, 2)

    def _orthonormal_components(self) -> torch.Tensor:
        components = []
        for raw in self.component_logits:
            vec = raw
            for basis in components:
                vec = vec - torch.sum(vec * basis) * basis
            components.append(F.normalize(vec, dim=0))
        return torch.stack(components, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        arch = self.architecture
        if arch == "MLP":
            return self.encoder(x[:, -1, :])
        if arch in {"LSTM", "LSTMG"}:
            output, (hidden, cell) = self.lstm(self.input_norm(x))
            self.last_hidden_state = hidden.detach()
            self.last_cell_state = cell.detach()
            recurrent = output[:, -1, :]
            if arch == "LSTMG":
                current = x[:, -1, :]
                gate = self.gate(torch.cat([current, recurrent], dim=-1))
                self.last_gate = gate.detach()
                return gate * recurrent + (1.0 - gate) * torch.tanh(self.feature_residual(current))
            return recurrent
        if arch == "GRU":
            output, hidden = self.gru(self.input_norm(x))
            self.last_hidden_state = hidden.detach()
            return output[:, -1, :]
        if arch == "BILSTM":
            output, _ = self.bilstm(self.input_norm(x))
            return self.merge(output[:, -1, :])
        if arch == "LSTM_TCN":
            recurrent, _ = self.lstm(self.input_norm(x))
            return self.tcn(recurrent.transpose(1, 2)).transpose(1, 2)[:, -1, :]
        if arch == "CNN_LSTM":
            conv = F.gelu(self.conv2(F.gelu(self.conv1(x.transpose(1, 2))))).transpose(1, 2)
            output, _ = self.lstm(conv)
            return output[:, -1, :]
        if arch == "ATTN_CNN_BILSTM":
            conv = F.gelu(self.conv2(F.gelu(self.conv1(x.transpose(1, 2))))).transpose(1, 2)
            recurrent, _ = self.bilstm(conv)
            attended, weights = self.attention(recurrent, recurrent, recurrent, need_weights=True)
            self.last_attention_weights = weights.detach()
            return attended[:, -1, :]
        if arch == "TCN":
            return self.tcn(self.input_projection(x.transpose(1, 2))).transpose(1, 2)[:, -1, :]
        if arch == "TST":
            encoded = self.transformer(self.input_projection(x) + self.position[:, : x.shape[1], :])
            return encoded[:, -1, :]
        if arch == "TFT":
            selected_steps, weights = [], []
            for step in range(x.shape[1]):
                selected, w = self.variable_selection(x[:, step, :])
                selected_steps.append(selected)
                weights.append(w)
            selected = torch.stack(selected_steps, dim=1)
            self.last_variable_selection = torch.stack(weights, dim=1).detach()
            lstm_out, _ = self.lstm_encoder(self.static_grn(selected))
            attended, attn_weights = self.interpretable_attention(lstm_out, lstm_out, lstm_out, need_weights=True)
            self.last_attention_weights = attn_weights.detach()
            return self.post_attention_grn(attended[:, -1, :])
        if arch == "AUTOFORMER":
            trend = self._moving_average(x)
            seasonal = x - trend
            seasonal_encoded = self.autocorrelation(self.input_projection(seasonal))
            return self.grn(seasonal_encoded[:, -1, :] + self.trend_projection(trend[:, -1, :]))
        if arch in {"PATCHTST", "TIMESFM"}:
            patches = self._patches(x)
            encoded = self.foundation_encoder(self.patch_embedding(patches) + self.position[:, : patches.shape[1], :])
            self.last_horizon_quantiles = self.horizon_quantiles(encoded[:, -1, :]).view(x.shape[0], 8, len(QUANTILES)).detach()
            return encoded[:, -1, :]
        if arch == "S4":
            state = self.input_projection(x)
            for layer in self.s4_layers:
                state = self.norm(F.gelu(layer(state)) + state)
            return state[:, -1, :]
        if arch == "STMN":
            projected = self.input_projection(x)
            state = projected[:, 0, :]
            attention = None
            for step in range(projected.shape[1]):
                query = projected[:, step, :]
                attention = torch.softmax(query @ self.memory_slots.T, dim=-1)
                read = attention @ self.memory_slots
                state = self.write_cell(torch.tanh(self.read_projection(torch.cat([query, read], dim=-1))), state)
            self.last_memory_attention = attention.detach() if attention is not None else None
            return state
        if arch == "CHRONOS":
            signal = x.mean(dim=-1)
            tokens = torch.bucketize(signal.contiguous(), self.bin_edges.to(x.device)).clamp(0, self.token_embedding.num_embeddings - 1)
            embedded = self.token_embedding(tokens) + self.position[:, : tokens.shape[1], :]
            mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=x.device), diagonal=1)
            return self.causal_transformer(embedded, mask=mask)[:, -1, :]
        if arch == "FEWC":
            current = x[:, -1, :]
            gates = torch.softmax(self.router(current), dim=-1)
            expert_outputs = torch.stack([expert(current) for expert in self.experts], dim=1)
            self.last_expert_gates = gates.detach()
            return torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)
        if arch == "GEODESIC":
            current = x[:, -1, :]
            weights = torch.softmax(-torch.cdist(current, self.landmarks.to(current.device), p=2), dim=-1)
            self.last_geodesic_attention = weights.detach()
            return torch.tanh(self.value_projection(current) + self.geodesic_projection(weights))
        if arch == "GHA":
            current = x[:, -1, :]
            components = self._orthonormal_components().to(current.device)
            projected = current @ components.T
            reconstructed = projected @ components
            self.last_reconstruction_error = torch.mean((current - reconstructed).pow(2), dim=-1).detach()
            return torch.tanh(self.projection(projected))
        if arch == "QCEW":
            output, _ = self.encoder(x)
            return torch.tanh(self.context_gate(output[:, -1, :]))
        if arch == "TESSERACT":
            time = self.time_factor[: x.shape[1], :].to(x.device)
            feature_projection = torch.einsum("bsf,fr->bsr", x, self.feature_factor.to(x.device))
            pooled = (feature_projection * time.unsqueeze(0)).mean(dim=1)
            return torch.tanh(torch.einsum("br,rsd,bs->bd", pooled, self.core.to(x.device), pooled))
        if arch == "TRR":
            output, _ = self.gru(x)
            h = output[:, -1, :]
            regimes = torch.softmax(self.regime_head(h), dim=-1)
            next_regime = regimes @ torch.softmax(self.transition_logits, dim=-1)
            self.last_regime_probabilities = next_regime.detach()
            return next_regime[:, :1] * torch.tanh(self.trend_head(h)) + (1.0 - next_regime[:, :1]) * torch.tanh(self.reversal_head(h))
        if arch == "MYTHOS_RDT":
            returns_to_go = torch.flip(torch.cumsum(torch.flip(x[..., 0:1], dims=[1]), dim=1), dims=[1])
            tokens = self.state_embedding(x) + self.return_embedding(returns_to_go) + self.position[:, : x.shape[1], :]
            mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), diagonal=1)
            encoded = self.decision_transformer(tokens, mask=mask)
            memory = torch.zeros((x.shape[0], HIDDEN_COUNT), dtype=x.dtype, device=x.device)
            for step in range(encoded.shape[1]):
                memory = self.recursive_memory(encoded[:, step, :], memory)
            self.last_recursive_memory = memory.detach()
            return memory
        if arch == "WM_CFX":
            currency_state = self.feature_to_currency(x)
            factor_state = currency_state @ self.currency_exposure.to(x.device)
            output, _ = self.factor_gru(factor_state)
            decoded = self.cross_rate_decoder(output[:, -1, :]).view(x.shape[0], self.currencies, self.currencies)
            self.last_cross_rate_consistency = (decoded + decoded.transpose(1, 2)).abs().mean(dim=(1, 2)).detach()
            return output[:, -1, :]
        if arch == "WM_GRAPH":
            current = x[:, -1, :]
            nodes = self.node_projection(current).view(current.shape[0], self.nodes, HIDDEN_COUNT)
            for layer in self.message_passing:
                nodes = layer(nodes)
            self.last_adjacency = torch.softmax(self.message_passing[-1].edge_logits, dim=-1).detach()
            return torch.tanh(self.readout(nodes.reshape(current.shape[0], -1)))
        raise ValueError(f"unsupported architecture {arch}")

    def ewc_penalty(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.architecture != "FEWC":
            return torch.zeros((), dtype=encoded.dtype, device=encoded.device)
        return torch.mean(self.fisher_diagonal.to(encoded.device) * (encoded - self.consolidation_anchor.to(encoded.device)).pow(2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.heads(self.encode(x))


@dataclass
class AITSTReferenceState:
    model: AITSTReferenceModel
    optimizer: torch.optim.Optimizer

    @classmethod
    def create(cls, device: Optional[torch.device] = None, lr: float = 3.0e-4) -> "AITSTReferenceState":
        dev = device or preferred_device()
        model = AITSTReferenceModel().to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
        return cls(model=model, optimizer=optimizer)


def predict_batch(batch: Iterable[Iterable[float]] | torch.Tensor, state: Optional[AITSTReferenceState] = None, data_has_volume: bool = True) -> dict[str, list[list[float]] | list[float] | str]:
    state = state or AITSTReferenceState.create()
    state.model.eval()
    x = _to_sequence(batch, device=next(state.model.parameters()).device, data_has_volume=data_has_volume)
    with torch.no_grad():
        logits, move, quantiles = state.model(x)
        probabilities = torch.softmax(logits, dim=-1)
    return {
        "plugin": PLUGIN_NAME,
        "architecture": ARCHITECTURE_MODE,
        "device": str(next(state.model.parameters()).device),
        "class_probabilities": probabilities.detach().cpu().tolist(),
        "move_mean_points": move.detach().cpu().tolist(),
        "move_quantiles": quantiles.detach().cpu().tolist(),
    }


def train_step(batch: Iterable[Iterable[float]] | torch.Tensor, labels: Iterable[int], moves: Iterable[float], state: Optional[AITSTReferenceState] = None, lr: float = 3.0e-4, data_has_volume: bool = True) -> AITSTReferenceState:
    state = state or AITSTReferenceState.create(lr=lr)
    state.model.train()
    device = next(state.model.parameters()).device
    x = _to_sequence(batch, device=device, data_has_volume=data_has_volume)
    y = _targets(labels, x.shape[0], device)
    m = _moves(moves, x.shape[0], device)
    state.optimizer.zero_grad(set_to_none=True)
    encoded = state.model.encode(x)
    logits, move, quantiles = state.model.heads(encoded)
    loss = F.cross_entropy(logits, y) + 0.05 * F.smooth_l1_loss(move, m) + 0.02 * _pinball_loss(quantiles, m)
    loss = loss + 0.01 * state.model.ewc_penalty(encoded)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(state.model.parameters(), 1.0)
    state.optimizer.step()
    return state
