"""TensorFlow/Keras reference backend for FXAI ai_mlp."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable, Optional

import tensorflow as tf

try:
    from fxai_financial_loss import financial_utility_loss_tensorflow
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2] / "API" / "Backends" / "Python"))
    from fxai_financial_loss import financial_utility_loss_tensorflow

PLUGIN_NAME = "ai_mlp"
ARCHITECTURE_MODE = "MLP"
FEATURE_COUNT = 32
CLASS_COUNT = 3
HIDDEN_COUNT = 48
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
VOLUME_FEATURE_INDEXES = (6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83)


def _to_sequence(batch: Iterable[Iterable[float]], data_has_volume: bool = True) -> tf.Tensor:
    x = tf.convert_to_tensor(list(batch), dtype=tf.float32)
    if len(x.shape) == 1:
        x = tf.reshape(x, (1, 1, -1))
    elif len(x.shape) == 2:
        x = tf.expand_dims(x, axis=1)
    if int(x.shape[-1]) < FEATURE_COUNT:
        x = tf.pad(x, [[0, 0], [0, 0], [0, FEATURE_COUNT - int(x.shape[-1])]])
    x = tf.clip_by_value(x[..., :FEATURE_COUNT], -8.0, 8.0)
    if not data_has_volume:
        indexes = [idx for idx in VOLUME_FEATURE_INDEXES if idx < FEATURE_COUNT]
        mask = tf.ones((FEATURE_COUNT,), dtype=tf.float32)
        mask = tf.tensor_scatter_nd_update(mask, [[idx] for idx in indexes], tf.zeros((len(indexes),), dtype=tf.float32))
        x = x * mask
    return x


class PredictionHeads(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.class_head = tf.keras.Sequential([tf.keras.layers.LayerNormalization(), tf.keras.layers.Dense(HIDDEN_COUNT, activation="gelu"), tf.keras.layers.Dense(CLASS_COUNT)])
        self.move_head = tf.keras.Sequential([tf.keras.layers.LayerNormalization(), tf.keras.layers.Dense(1, activation="softplus")])
        self.quantile_head = tf.keras.Sequential([tf.keras.layers.LayerNormalization(), tf.keras.layers.Dense(len(QUANTILES), activation="softplus")])

    def call(self, encoded: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return self.class_head(encoded), tf.squeeze(self.move_head(encoded), axis=-1), tf.cumsum(self.quantile_head(encoded), axis=-1)


class CausalTCNBlock(tf.keras.layers.Layer):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(HIDDEN_COUNT, 3, padding="causal", dilation_rate=dilation, activation="gelu")
        self.conv2 = tf.keras.layers.Conv1D(HIDDEN_COUNT, 3, padding="causal", dilation_rate=dilation)
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.nn.gelu(self.norm(x + self.conv2(self.conv1(x))))


class AIMLPTensorFlowModel(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.architecture = ARCHITECTURE_MODE
        self.heads = PredictionHeads()
        self.norm = tf.keras.layers.LayerNormalization()
        self.encoder = tf.keras.Sequential([tf.keras.layers.LayerNormalization(), tf.keras.layers.Dense(HIDDEN_COUNT, activation="gelu"), tf.keras.layers.Dropout(0.05), tf.keras.layers.Dense(HIDDEN_COUNT, activation="gelu")])
        self.lstm = tf.keras.layers.LSTM(HIDDEN_COUNT, return_sequences=True, return_state=True, dropout=0.05)
        self.gru = tf.keras.layers.GRU(HIDDEN_COUNT, return_sequences=True, return_state=True, dropout=0.05)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(HIDDEN_COUNT // 2, return_sequences=True, dropout=0.05))
        self.conv1 = tf.keras.layers.Conv1D(HIDDEN_COUNT, 3, padding="same", activation="gelu")
        self.conv2 = tf.keras.layers.Conv1D(HIDDEN_COUNT, 3, padding="same", activation="gelu")
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=HIDDEN_COUNT // 4)
        self.tcn_blocks = [CausalTCNBlock(1), CausalTCNBlock(2), CausalTCNBlock(4), CausalTCNBlock(8)]
        self.gate = tf.keras.layers.Dense(HIDDEN_COUNT, activation="sigmoid")
        self.residual = tf.keras.layers.Dense(HIDDEN_COUNT, activation="tanh")

    def call(self, x: tf.Tensor, training: bool = False) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        arch = self.architecture
        if arch == "MLP":
            encoded = self.encoder(x[:, -1, :], training=training)
        elif arch == "LSTM":
            output, hidden, cell = self.lstm(self.norm(x), training=training)
            self.last_hidden_state = hidden
            self.last_cell_state = cell
            encoded = output[:, -1, :]
        elif arch == "LSTMG":
            output, hidden, cell = self.lstm(self.norm(x), training=training)
            current = x[:, -1, :]
            gate = self.gate(tf.concat([current, output[:, -1, :]], axis=-1))
            self.last_gate = gate
            encoded = gate * output[:, -1, :] + (1.0 - gate) * self.residual(current)
        elif arch == "GRU":
            output, hidden = self.gru(self.norm(x), training=training)
            self.last_hidden_state = hidden
            encoded = output[:, -1, :]
        elif arch == "BILSTM":
            encoded = self.bilstm(self.norm(x), training=training)[:, -1, :]
        elif arch == "LSTM_TCN":
            y, _, _ = self.lstm(self.norm(x), training=training)
            for block in self.tcn_blocks[:3]:
                y = block(y)
            encoded = y[:, -1, :]
        elif arch == "CNN_LSTM":
            y = self.conv2(self.conv1(x))
            y, _, _ = self.lstm(y, training=training)
            encoded = y[:, -1, :]
        elif arch == "ATTN_CNN_BILSTM":
            y = self.bilstm(self.conv2(self.conv1(x)), training=training)
            encoded = self.attention(y, y, training=training)[:, -1, :]
        elif arch == "TCN":
            y = self.conv1(x)
            for block in self.tcn_blocks:
                y = block(y)
            encoded = y[:, -1, :]
        else:
            raise ValueError(f"unsupported architecture {arch}")
        return self.heads(encoded)


@dataclass
class AIMLPTensorFlowState:
    model: AIMLPTensorFlowModel
    optimizer: tf.keras.optimizers.Optimizer

    @classmethod
    def create(cls, lr: float = 3.0e-4) -> "AIMLPTensorFlowState":
        return cls(model=AIMLPTensorFlowModel(), optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1.0e-4))


def predict_batch(batch: Iterable[Iterable[float]], state: Optional[AIMLPTensorFlowState] = None, data_has_volume: bool = True) -> dict[str, list[list[float]] | list[float] | str]:
    state = state or AIMLPTensorFlowState.create()
    logits, move, quantiles = state.model(_to_sequence(batch, data_has_volume=data_has_volume), training=False)
    return {"plugin": PLUGIN_NAME, "architecture": ARCHITECTURE_MODE, "class_probabilities": tf.nn.softmax(logits, axis=-1).numpy().tolist(), "move_mean_points": move.numpy().tolist(), "move_quantiles": quantiles.numpy().tolist()}


def train_step(
    batch: Iterable[Iterable[float]],
    labels: Iterable[int],
    moves: Iterable[float],
    state: Optional[AIMLPTensorFlowState] = None,
    lr: float = 3.0e-4,
    data_has_volume: bool = True,
    financial_targets: Optional[dict[str, Any]] = None,
    financial_loss_config: Optional[dict[str, Any]] = None,
) -> AIMLPTensorFlowState:
    state = state or AIMLPTensorFlowState.create(lr=lr)
    x = _to_sequence(batch, data_has_volume=data_has_volume)
    with tf.GradientTape() as tape:
        logits, move, quantiles = state.model(x, training=True)
        loss = financial_utility_loss_tensorflow(
            logits,
            labels,
            move,
            moves,
            quantile_prediction=quantiles,
            financial_targets=financial_targets,
            financial_loss_config=financial_loss_config,
            quantiles=QUANTILES,
        )
    grads = tape.gradient(loss, state.model.trainable_variables)
    state.optimizer.apply_gradients(zip(grads, state.model.trainable_variables))
    return state
