from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import tensorflow as tf

PLUGIN_NAME = "ai_bilstm"
ARCHITECTURE_MODE = "bidirectional"
FEATURE_COUNT = 32
HIDDEN_COUNT = 20
CLASS_COUNT = 3


def _features(batch: Iterable[Iterable[float]]) -> tf.Tensor:
    rows = []
    for row in batch:
        values = list(row)[:FEATURE_COUNT]
        if len(values) < FEATURE_COUNT:
            values.extend([0.0] * (FEATURE_COUNT - len(values)))
        rows.append(values)
    if not rows:
        rows = [[0.0] * FEATURE_COUNT]
    return tf.clip_by_value(tf.constant(rows, dtype=tf.float32), -50.0, 50.0)


@dataclass
class AIBiLSTMTensorFlowState:
    input_weight: tf.Variable
    class_weight: tf.Variable
    move_weight: tf.Variable

    @classmethod
    def create(cls) -> "AIBiLSTMTensorFlowState":
        generator = tf.random.Generator.from_seed(20_000 + 59)
        return cls(
            input_weight=tf.Variable(generator.normal([FEATURE_COUNT, HIDDEN_COUNT]) * 0.05),
            class_weight=tf.Variable(generator.normal([HIDDEN_COUNT, CLASS_COUNT]) * 0.04),
            move_weight=tf.Variable(generator.uniform([HIDDEN_COUNT, 1]) * 0.03),
        )


def encode(batch: Iterable[Iterable[float]], state: Optional[AIBiLSTMTensorFlowState] = None) -> tf.Tensor:
    state = state or AIBiLSTMTensorFlowState.create()
    hidden = tf.tanh(tf.matmul(_features(batch), state.input_weight))
    if ARCHITECTURE_MODE in {"tcn", "cnnLSTM", "attentionCNNBiLSTM", "lstmTCN"}:
        hidden = 0.7 * hidden + 0.3 * tf.roll(hidden, shift=1, axis=1)
    elif ARCHITECTURE_MODE in {"recurrent", "gatedRecurrent", "gru", "bidirectional"}:
        hidden = tf.tanh(hidden + 0.1 * tf.cumsum(hidden, axis=1) / float(max(HIDDEN_COUNT, 1)))
    return hidden


def predict_batch(batch: Iterable[Iterable[float]], state: Optional[AIBiLSTMTensorFlowState] = None) -> dict[str, object]:
    state = state or AIBiLSTMTensorFlowState.create()
    hidden = encode(batch, state)
    probabilities = tf.nn.softmax(tf.matmul(hidden, state.class_weight), axis=1)
    move = tf.nn.relu(tf.matmul(hidden, state.move_weight))[:, 0]
    devices = [device.name for device in tf.config.list_logical_devices()]
    return {
        "plugin": PLUGIN_NAME,
        "devices": devices,
        "class_probabilities": probabilities.numpy().tolist(),
        "move_mean_points": move.numpy().tolist(),
    }


def train_step(batch: Iterable[Iterable[float]], labels: Iterable[int], moves: Iterable[float], state: Optional[AIBiLSTMTensorFlowState] = None, lr: float = 0.01) -> AIBiLSTMTensorFlowState:
    state = state or AIBiLSTMTensorFlowState.create()
    labels_tensor = tf.clip_by_value(tf.constant(list(labels), dtype=tf.int32), 0, CLASS_COUNT - 1)
    moves_tensor = tf.abs(tf.constant(list(moves), dtype=tf.float32))
    with tf.GradientTape() as tape:
        hidden = encode(batch, state)
        logits = tf.matmul(hidden, state.class_weight)
        move_pred = tf.nn.relu(tf.matmul(hidden, state.move_weight))[:, 0]
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_tensor, logits=logits))
        loss += 0.05 * tf.reduce_mean(tf.keras.losses.huber(moves_tensor, move_pred))
    grads = tape.gradient(loss, [state.input_weight, state.class_weight, state.move_weight])
    for variable, grad in zip([state.input_weight, state.class_weight, state.move_weight], grads):
        if grad is not None:
            variable.assign_sub(lr * tf.clip_by_value(grad, -1.0, 1.0))
    return state
