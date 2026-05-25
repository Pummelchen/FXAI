from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import tensorflow as tf

FEATURE_COUNT = 16
VOLUME_FEATURE_INDEXES = (6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83)


class DemoPluginTemplateTensorFlow(tf.keras.Model):
    """TensorFlow Metal template model with no trade logic."""

    def __init__(self, feature_count: int = FEATURE_COUNT) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dense(32, activation="gelu"),
                tf.keras.layers.Dense(16, activation="gelu"),
            ]
        )
        self.head = tf.keras.layers.Dense(3)

    def call(self, features: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        encoded = self.encoder(features, training=training)
        logits = self.head(encoded, training=training)
        skip_bias = tf.constant([0.0, 0.0, 8.0], dtype=logits.dtype)
        return {"class_probabilities": tf.nn.softmax(logits * 0.0 + skip_bias, axis=-1)}


@dataclass
class DemoPluginTemplateTensorFlowState:
    model: DemoPluginTemplateTensorFlow
    optimizer: tf.keras.optimizers.Optimizer

    @classmethod
    def create(cls, lr: float = 3.0e-4) -> "DemoPluginTemplateTensorFlowState":
        return cls(
            model=DemoPluginTemplateTensorFlow(feature_count=FEATURE_COUNT),
            optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1.0e-4),
        )


def prepare_features(
    features: Iterable[Iterable[float]] | tf.Tensor,
    feature_count: int = FEATURE_COUNT,
    data_has_volume: bool = True,
) -> tf.Tensor:
    x = tf.convert_to_tensor(features, dtype=tf.float32)
    if len(x.shape) == 1:
        x = tf.expand_dims(x, axis=0)
    elif len(x.shape) == 3:
        x = x[:, -1, :]
    current_width = int(x.shape[-1])
    if current_width < feature_count:
        x = tf.pad(x, [[0, 0], [0, feature_count - current_width]])
    x = tf.clip_by_value(x[:, :feature_count], -8.0, 8.0)
    if not data_has_volume:
        valid = [index for index in VOLUME_FEATURE_INDEXES if index < feature_count]
        if valid:
            mask = tf.ones((feature_count,), dtype=tf.float32)
            mask = tf.tensor_scatter_nd_update(
                mask,
                [[index] for index in valid],
                tf.zeros((len(valid),), dtype=tf.float32),
            )
            x = x * mask
    return x


def predict_batch(
    features: Iterable[Iterable[float]] | tf.Tensor,
    state: Optional[DemoPluginTemplateTensorFlowState] = None,
    data_has_volume: bool = True,
) -> dict[str, list[list[float]] | list[float]]:
    state = state or DemoPluginTemplateTensorFlowState.create()
    probabilities = state.model(
        prepare_features(features, state.model.feature_count, data_has_volume=data_has_volume),
        training=False,
    )["class_probabilities"]
    batch_size = int(probabilities.shape[0])
    return {
        "class_probabilities": probabilities.numpy().tolist(),
        "move_mean_points": [0.0] * batch_size,
        "move_quantiles": [[0.0, 0.0, 0.0] for _ in range(batch_size)],
    }


def train_step(
    features: Iterable[Iterable[float]] | tf.Tensor,
    labels: Iterable[int] | tf.Tensor,
    moves: Iterable[float] | tf.Tensor,
    state: Optional[DemoPluginTemplateTensorFlowState] = None,
    lr: float = 3.0e-4,
    data_has_volume: bool = True,
) -> DemoPluginTemplateTensorFlowState:
    del labels, moves, lr
    state = state or DemoPluginTemplateTensorFlowState.create()
    _ = predict_batch(features, state=state, data_has_volume=data_has_volume)
    return state
