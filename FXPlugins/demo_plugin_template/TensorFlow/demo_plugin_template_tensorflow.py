from __future__ import annotations

from typing import Iterable

import tensorflow as tf


class DemoPluginTemplateTensorFlow(tf.keras.Model):
    """TensorFlow Metal template model with no trade logic."""

    def __init__(self, feature_count: int = 16) -> None:
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


def prepare_features(features: Iterable[Iterable[float]] | tf.Tensor, feature_count: int = 16) -> tf.Tensor:
    x = tf.convert_to_tensor(features, dtype=tf.float32)
    if len(x.shape) == 1:
        x = tf.expand_dims(x, axis=0)
    current_width = int(x.shape[-1])
    if current_width < feature_count:
        x = tf.pad(x, [[0, 0], [0, feature_count - current_width]])
    return x[:, :feature_count]


def predict_batch(model: DemoPluginTemplateTensorFlow, features: Iterable[Iterable[float]] | tf.Tensor) -> dict[str, tf.Tensor]:
    return model(prepare_features(features, model.feature_count), training=False)


def train_step(
    model: DemoPluginTemplateTensorFlow,
    optimizer: tf.keras.optimizers.Optimizer,
    features: Iterable[Iterable[float]] | tf.Tensor,
    labels: Iterable[int] | tf.Tensor,
) -> DemoPluginTemplateTensorFlow:
    x = prepare_features(features, model.feature_count)
    y = tf.clip_by_value(tf.convert_to_tensor(labels, dtype=tf.int32), 0, 2)
    with tf.GradientTape() as tape:
        probabilities = tf.clip_by_value(model(x, training=True)["class_probabilities"], 1.0e-6, 1.0)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, probabilities))
    gradients = tape.gradient(loss, model.trainable_variables)
    gradient_pairs = [
        (gradient, variable)
        for gradient, variable in zip(gradients, model.trainable_variables)
        if gradient is not None
    ]
    if gradient_pairs:
        optimizer.apply_gradients(gradient_pairs)
    return model
