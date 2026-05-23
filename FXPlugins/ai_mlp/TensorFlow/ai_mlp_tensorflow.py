"""TensorFlow/Metal backend for ai_mlp.

This plugin-local module mirrors the Swift CPU feature contract for batched
sequence inference. It uses TensorFlow on CPU or tensorflow-metal when the
runtime is installed on Apple Silicon.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import tensorflow as tf

FEATURES = 24
HIDDEN = 16
CLASSES = 3
ARCH_ID = 9
VOLUME_INDEXES = [6, 68, 69, 70, 71, 74, 75, 76, 77, 78, 80, 81, 82, 83]


def prepare_features(features: Iterable[Iterable[float]] | tf.Tensor, data_has_volume: bool = True) -> tf.Tensor:
    x = tf.convert_to_tensor(features, dtype=tf.float32)
    if len(x.shape) == 1:
        x = tf.expand_dims(x, 0)
    x = tf.clip_by_value(x, -50.0, 50.0)
    if not data_has_volume:
        updates = []
        for idx in VOLUME_INDEXES:
            if idx < x.shape[-1]:
                updates.append(idx)
        if updates:
            mask = tf.ones_like(x)
            for idx in updates:
                mask = tf.tensor_scatter_nd_update(mask, [[0, idx]], [0.0]) if x.shape[0] == 1 else mask
            cols = [tf.zeros_like(x[:, i:i+1]) if i in updates else x[:, i:i+1] for i in range(x.shape[-1])]
            x = tf.concat(cols, axis=-1)
    return x


def _col(x: tf.Tensor, index: int) -> tf.Tensor:
    if index >= x.shape[-1]:
        return tf.zeros(tf.shape(x)[:-1], dtype=x.dtype)
    return x[..., index]


def build_features(x: tf.Tensor, horizon_minutes: int = 30, sequence_bars: int = 32, session_bucket: int = 0) -> tf.Tensor:
    arch = tf.sin(_col(x, 1) * (0.35 + 0.03 * ARCH_ID))
    values = [
        tf.ones_like(_col(x, 1)), _col(x, 1), _col(x, 2), _col(x, 3), _col(x, 4), _col(x, 7), _col(x, 12),
        tf.clip_by_value(0.65 * _col(x, 40) + 0.35 * _col(x, 6), -8.0, 8.0),
        _col(x, 1), tf.abs(_col(x, 4)), tf.abs(_col(x, 3) - _col(x, 1)), _col(x, 7),
        _col(x, 1), _col(x, 2), _col(x, 78), _col(x, 83),
        tf.fill(tf.shape(_col(x, 1)), max(0.0, min(float(horizon_minutes) / 60.0, 2.0))),
        tf.fill(tf.shape(_col(x, 1)), max(0.0, min(float(session_bucket) / 5.0, 1.0))),
        tf.fill(tf.shape(_col(x, 1)), max(0.0, min(float(sequence_bars) / 128.0, 2.0))),
        arch, tf.tanh(_col(x, 2) + arch), tf.tanh(_col(x, 3) - arch), _col(x, 1) - _col(x, 2), _col(x, 2) - _col(x, 3)
    ]
    return tf.clip_by_value(tf.stack(values, axis=-1), -8.0, 8.0)


@dataclass
class SequenceTensorFlowState:
    w1: tf.Tensor
    b1: tf.Tensor
    head: tf.Tensor
    move: tf.Tensor

    @classmethod
    def seeded(cls) -> "SequenceTensorFlowState":
        tf.random.set_seed(20_000 + ARCH_ID)
        return cls(
            tf.random.normal((HIDDEN, FEATURES), stddev=0.08),
            tf.random.normal((HIDDEN,), stddev=0.03),
            tf.random.normal((CLASSES, HIDDEN + 1), stddev=0.04),
            tf.abs(tf.random.normal((HIDDEN + 1,), stddev=0.02)),
        )


def predict_batch(features: Iterable[Iterable[float]] | tf.Tensor, *, data_has_volume: bool = True, horizon_minutes: int = 30, sequence_bars: int = 32, session_bucket: int = 0, state: SequenceTensorFlowState | None = None) -> dict[str, tf.Tensor]:
    state = state or SequenceTensorFlowState.seeded()
    x = prepare_features(features, data_has_volume=data_has_volume)
    z = build_features(x, horizon_minutes, sequence_bars, session_bucket)
    hidden = tf.tanh(tf.linalg.matmul(z, state.w1, transpose_b=True) + state.b1)
    hidden_bias = tf.concat([tf.ones((tf.shape(hidden)[0], 1), dtype=hidden.dtype), hidden], axis=-1)
    probabilities = tf.nn.softmax(tf.clip_by_value(tf.linalg.matmul(hidden_bias, state.head, transpose_b=True), -30.0, 30.0), axis=-1)
    move = tf.maximum(tf.linalg.matvec(hidden_bias, state.move), 0.0)
    return {"class_probabilities": probabilities, "expected_move_points": move, "hidden": hidden}
