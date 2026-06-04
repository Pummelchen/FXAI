"""Shared financial utility loss helpers for FXAI plugin-local backends."""

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from collections.abc import Mapping as MappingABC
import math
from typing import Any, Iterable, Mapping


DEFAULT_FINANCIAL_LOSS_CONFIG: dict[str, float | str] = {
    "version": "fxai-financial-loss-v1",
    "classificationWeight": 1.0,
    "moveWeight": 0.05,
    "quantileWeight": 0.02,
    "adverseTailWeight": 0.18,
    "costRiskWeight": 0.10,
    "activityWeight": 0.04,
    "downsideUtilityWeight": 0.03,
    "maxTailMultiplier": 4.0,
    "targetTradeProbability": 0.35,
    "utilityEpsilon": 1.0e-6,
}


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _config_value(config: Mapping[str, Any] | None, key: str) -> float:
    raw = (config or {}).get(key, DEFAULT_FINANCIAL_LOSS_CONFIG[key])
    return max(0.0, _safe_float(raw, float(DEFAULT_FINANCIAL_LOSS_CONFIG[key])))


def _validate_config_version(config: Mapping[str, Any] | None) -> None:
    version = str((config or {}).get("version", DEFAULT_FINANCIAL_LOSS_CONFIG["version"]))
    if version != DEFAULT_FINANCIAL_LOSS_CONFIG["version"]:
        raise ValueError(f"unsupported FXAI financial loss version {version}")


def _target_value(targets: Any, key: str, index: int, default: float) -> float:
    if targets is None:
        return default
    if isinstance(targets, MappingABC):
        value = targets.get(key, default)
    elif isinstance(targets, IterableABC) and not isinstance(targets, (str, bytes)):
        rows = list(targets)
        if not rows:
            return default
        row = rows[min(index, len(rows) - 1)]
        value = row.get(key, default) if isinstance(row, MappingABC) else default
    else:
        value = default
    if isinstance(value, IterableABC) and not isinstance(value, (str, bytes)):
        values = list(value)
        value = values[min(index, len(values) - 1)] if values else default
    return _safe_float(value, default)


def _torch_vector(targets: Any, key: str, batch_size: int, device: Any, default: float):
    import torch

    values = [_target_value(targets, key, index, default) for index in range(batch_size)]
    return torch.tensor(values, dtype=torch.float32, device=device)


def _torch_labels(labels: Iterable[int] | Any, batch_size: int, device: Any):
    import torch

    if isinstance(labels, torch.Tensor):
        y = labels.detach().to(device=device, dtype=torch.long).view(-1)
        if y.numel() >= batch_size:
            return y[:batch_size].clamp(0, 2)
        values = y.detach().cpu().tolist()
    else:
        values = list(labels or [])
    if not values:
        values = [2] * batch_size
    if len(values) < batch_size:
        values.extend([values[-1]] * (batch_size - len(values)))
    return torch.tensor(values[:batch_size], dtype=torch.long, device=device).clamp(0, 2)


def _torch_moves(moves: Iterable[float] | Any, targets: Any, batch_size: int, device: Any):
    import torch

    if isinstance(moves, torch.Tensor):
        signed = moves.detach().to(device=device, dtype=torch.float32).view(-1)
        if signed.numel() < batch_size:
            pad = signed[-1:].expand(batch_size - signed.numel()) if signed.numel() else torch.zeros(batch_size, device=device)
            signed = torch.cat([signed, pad], dim=0)
        signed = signed[:batch_size]
    else:
        values = list(moves or [])
        if not values:
            values = [_target_value(targets, "movePoints", index, 0.0) for index in range(batch_size)]
        if len(values) < batch_size:
            values.extend([values[-1]] * (batch_size - len(values)))
        signed = torch.tensor(values[:batch_size], dtype=torch.float32, device=device)
    target_signed = _torch_vector(targets, "movePoints", batch_size, device, 0.0)
    return torch.where(target_signed.abs() > 0.0, target_signed, signed)


def _resolved_quantiles(count: int, quantiles: tuple[float, ...]) -> tuple[float, ...]:
    if count <= 0:
        return ()
    if len(quantiles) == count:
        return quantiles
    if count == 3:
        return (0.25, 0.50, 0.75)
    if count == 5:
        return (0.10, 0.25, 0.50, 0.75, 0.90)
    step = 1.0 / float(count + 1)
    return tuple(step * float(index + 1) for index in range(count))


def financial_utility_loss_torch(
    logits: Any,
    labels: Iterable[int] | Any,
    move_prediction: Any,
    move_targets: Iterable[float] | Any,
    quantile_prediction: Any | None = None,
    financial_targets: Any | None = None,
    financial_loss_config: Mapping[str, Any] | None = None,
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> Any:
    """Return FXAI's hybrid financial utility loss for PyTorch models."""

    import torch
    import torch.nn.functional as F

    _validate_config_version(financial_loss_config)
    batch_size = int(logits.shape[0])
    device = logits.device
    y = _torch_labels(labels, batch_size, device)
    signed_move = _torch_moves(move_targets, financial_targets, batch_size, device)
    move_abs = signed_move.abs().clamp_min(0.0)
    move_pred = move_prediction.view(-1)[:batch_size].clamp_min(0.0)

    sample_weight = _torch_vector(financial_targets, "sampleWeight", batch_size, device, 1.0).clamp_min(0.0)
    mfe = _torch_vector(financial_targets, "mfePoints", batch_size, device, 0.0).clamp_min(0.0)
    mae = _torch_vector(financial_targets, "maePoints", batch_size, device, 0.0).clamp_min(0.0)
    path_risk = _torch_vector(financial_targets, "pathRisk", batch_size, device, 0.0).clamp(0.0, 1.0)
    fill_risk = _torch_vector(financial_targets, "fillRisk", batch_size, device, 0.0).clamp(0.0, 1.0)
    price_cost = _torch_vector(financial_targets, "priceCostPoints", batch_size, device, 0.0).clamp_min(0.0)
    min_move = _torch_vector(financial_targets, "minMovePoints", batch_size, device, 0.0).clamp_min(0.0)

    epsilon = max(_config_value(financial_loss_config, "utilityEpsilon"), 1.0e-12)
    move_scale = torch.maximum(torch.maximum(move_abs, min_move), price_cost).clamp_min(1.0)
    risk = (path_risk + fill_risk).clamp(0.0, 2.0)
    tail_multiplier = (move_abs / torch.maximum(min_move + price_cost, torch.full_like(move_abs, 0.25))).clamp(
        0.0,
        _config_value(financial_loss_config, "maxTailMultiplier"),
    )
    ce_weight = sample_weight * (1.0 + _config_value(financial_loss_config, "adverseTailWeight") * tail_multiplier * (0.5 + risk))
    ce = F.cross_entropy(logits, y, reduction="none")
    classification_loss = (ce * ce_weight).sum() / ce_weight.sum().clamp_min(epsilon)

    move_loss = F.smooth_l1_loss(move_pred, move_abs, reduction="none")
    move_loss = (move_loss * sample_weight).sum() / sample_weight.sum().clamp_min(epsilon)

    if quantile_prediction is not None:
        q_values = _resolved_quantiles(int(quantile_prediction.shape[-1]), quantiles)
        q = torch.tensor(q_values, dtype=quantile_prediction.dtype, device=device).view(1, -1)
        error = move_abs.view(-1, 1) - quantile_prediction[:batch_size, :len(q_values)]
        pinball = torch.maximum(q * error, (q - 1.0) * error).mean(dim=-1)
        quantile_loss = (pinball * sample_weight).sum() / sample_weight.sum().clamp_min(epsilon)
    else:
        quantile_loss = torch.zeros((), dtype=logits.dtype, device=device)

    probabilities = torch.softmax(logits, dim=-1)
    sell_probability = probabilities[:, 0]
    buy_probability = probabilities[:, 1]
    trade_probability = (sell_probability + buy_probability).clamp(0.0, 1.0)

    wrong_buy = buy_probability * torch.relu(-signed_move - price_cost) / move_scale
    wrong_sell = sell_probability * torch.relu(signed_move - price_cost) / move_scale
    adverse_shape = 1.0 + risk + (mae / torch.maximum(mfe, move_scale)).clamp(0.0, 3.0)
    adverse_tail_loss = ((wrong_buy + wrong_sell) * adverse_shape * sample_weight).sum() / sample_weight.sum().clamp_min(epsilon)

    risk_cost = (price_cost + 0.35 * mae + risk * torch.maximum(min_move, price_cost)).clamp_min(0.0)
    cost_risk_loss = (trade_probability * risk_cost / move_scale * sample_weight).sum() / sample_weight.sum().clamp_min(epsilon)

    opportunity = torch.sigmoid((move_abs - price_cost - min_move) / move_scale)
    safe_fraction = (1.0 - 0.25 * risk).clamp(0.0, 1.0)
    target_activity = _config_value(financial_loss_config, "targetTradeProbability") * opportunity * safe_fraction
    activity_loss = ((trade_probability - target_activity).pow(2) * sample_weight).sum() / sample_weight.sum().clamp_min(epsilon)

    expected_net = buy_probability * (signed_move - price_cost) + sell_probability * (-signed_move - price_cost)
    expected_net = expected_net - trade_probability * risk_cost
    downside_loss = (torch.relu(-expected_net / move_scale) * sample_weight).sum() / sample_weight.sum().clamp_min(epsilon)

    return (
        _config_value(financial_loss_config, "classificationWeight") * classification_loss
        + _config_value(financial_loss_config, "moveWeight") * move_loss
        + _config_value(financial_loss_config, "quantileWeight") * quantile_loss
        + _config_value(financial_loss_config, "adverseTailWeight") * adverse_tail_loss
        + _config_value(financial_loss_config, "costRiskWeight") * cost_risk_loss
        + _config_value(financial_loss_config, "activityWeight") * activity_loss
        + _config_value(financial_loss_config, "downsideUtilityWeight") * downside_loss
    )


def _tf_vector(targets: Any, key: str, batch_size: int, default: float):
    import tensorflow as tf

    values = [_target_value(targets, key, index, default) for index in range(batch_size)]
    return tf.convert_to_tensor(values, dtype=tf.float32)


def _tf_labels(labels: Iterable[int] | Any, batch_size: int):
    import tensorflow as tf

    if isinstance(labels, tf.Tensor):
        y = tf.reshape(tf.cast(labels, tf.int32), (-1,))
        count = int(y.shape[0])
        if count >= batch_size:
            return tf.clip_by_value(y[:batch_size], 0, 2)
        pad_value = y[-1] if count > 0 else tf.constant(2, dtype=tf.int32)
        y = tf.concat([y, tf.fill((batch_size - count,), pad_value)], axis=0)
        return tf.clip_by_value(y[:batch_size], 0, 2)
    values = list(labels or [])
    if not values:
        values = [2] * batch_size
    if len(values) < batch_size:
        values.extend([values[-1]] * (batch_size - len(values)))
    return tf.clip_by_value(tf.convert_to_tensor(values[:batch_size], dtype=tf.int32), 0, 2)


def financial_utility_loss_tensorflow(
    logits: Any,
    labels: Iterable[int] | Any,
    move_prediction: Any,
    move_targets: Iterable[float] | Any,
    quantile_prediction: Any | None = None,
    financial_targets: Any | None = None,
    financial_loss_config: Mapping[str, Any] | None = None,
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> Any:
    """Return FXAI's hybrid financial utility loss for TensorFlow/Keras models."""

    import tensorflow as tf

    _validate_config_version(financial_loss_config)
    batch_size = int(logits.shape[0])
    y = _tf_labels(labels, batch_size)
    if isinstance(move_targets, tf.Tensor):
        signed_move = tf.reshape(tf.cast(move_targets, tf.float32), (-1,))
        count = int(signed_move.shape[0])
        if count < batch_size:
            pad_value = signed_move[-1] if count > 0 else tf.constant(0.0, dtype=tf.float32)
            signed_move = tf.concat([signed_move, tf.fill((batch_size - count,), pad_value)], axis=0)
        signed_move = signed_move[:batch_size]
    else:
        raw_moves = list(move_targets or [])
        if not raw_moves:
            raw_moves = [_target_value(financial_targets, "movePoints", index, 0.0) for index in range(batch_size)]
        if len(raw_moves) < batch_size:
            raw_moves.extend([raw_moves[-1]] * (batch_size - len(raw_moves)))
        signed_move = tf.convert_to_tensor(raw_moves[:batch_size], dtype=tf.float32)
    target_signed = _tf_vector(financial_targets, "movePoints", batch_size, 0.0)
    signed_move = tf.where(tf.abs(target_signed) > 0.0, target_signed, signed_move)
    move_abs = tf.maximum(tf.abs(signed_move), 0.0)
    move_pred = tf.maximum(tf.reshape(move_prediction, (-1,))[:batch_size], 0.0)

    sample_weight = tf.maximum(_tf_vector(financial_targets, "sampleWeight", batch_size, 1.0), 0.0)
    mfe = tf.maximum(_tf_vector(financial_targets, "mfePoints", batch_size, 0.0), 0.0)
    mae = tf.maximum(_tf_vector(financial_targets, "maePoints", batch_size, 0.0), 0.0)
    path_risk = tf.clip_by_value(_tf_vector(financial_targets, "pathRisk", batch_size, 0.0), 0.0, 1.0)
    fill_risk = tf.clip_by_value(_tf_vector(financial_targets, "fillRisk", batch_size, 0.0), 0.0, 1.0)
    price_cost = tf.maximum(_tf_vector(financial_targets, "priceCostPoints", batch_size, 0.0), 0.0)
    min_move = tf.maximum(_tf_vector(financial_targets, "minMovePoints", batch_size, 0.0), 0.0)

    epsilon = max(_config_value(financial_loss_config, "utilityEpsilon"), 1.0e-12)
    move_scale = tf.maximum(tf.maximum(tf.maximum(move_abs, min_move), price_cost), 1.0)
    risk = tf.clip_by_value(path_risk + fill_risk, 0.0, 2.0)
    tail_base = tf.maximum(min_move + price_cost, tf.fill(tf.shape(move_abs), 0.25))
    tail_multiplier = tf.clip_by_value(
        move_abs / tail_base,
        0.0,
        _config_value(financial_loss_config, "maxTailMultiplier"),
    )
    ce_weight = sample_weight * (1.0 + _config_value(financial_loss_config, "adverseTailWeight") * tail_multiplier * (0.5 + risk))
    ce = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
    classification_loss = tf.reduce_sum(ce * ce_weight) / tf.maximum(tf.reduce_sum(ce_weight), epsilon)

    move_delta = tf.abs(move_abs - move_pred)
    quadratic = tf.minimum(move_delta, 1.0)
    linear = move_delta - quadratic
    move_loss = 0.5 * tf.square(quadratic) + linear
    move_loss = tf.reduce_sum(move_loss * sample_weight) / tf.maximum(tf.reduce_sum(sample_weight), epsilon)

    if quantile_prediction is not None:
        q_values = _resolved_quantiles(int(quantile_prediction.shape[-1]), quantiles)
        q = tf.reshape(tf.constant(q_values, dtype=tf.float32), (1, -1))
        error = tf.reshape(move_abs, (-1, 1)) - quantile_prediction[:batch_size, :len(q_values)]
        pinball = tf.reduce_mean(tf.maximum(q * error, (q - 1.0) * error), axis=-1)
        quantile_loss = tf.reduce_sum(pinball * sample_weight) / tf.maximum(tf.reduce_sum(sample_weight), epsilon)
    else:
        quantile_loss = tf.constant(0.0, dtype=tf.float32)

    probabilities = tf.nn.softmax(logits, axis=-1)
    sell_probability = probabilities[:, 0]
    buy_probability = probabilities[:, 1]
    trade_probability = tf.clip_by_value(sell_probability + buy_probability, 0.0, 1.0)

    wrong_buy = buy_probability * tf.nn.relu(-signed_move - price_cost) / move_scale
    wrong_sell = sell_probability * tf.nn.relu(signed_move - price_cost) / move_scale
    adverse_shape = 1.0 + risk + tf.clip_by_value(mae / tf.maximum(mfe, move_scale), 0.0, 3.0)
    adverse_tail_loss = tf.reduce_sum((wrong_buy + wrong_sell) * adverse_shape * sample_weight) / tf.maximum(
        tf.reduce_sum(sample_weight),
        epsilon,
    )

    risk_cost = tf.maximum(price_cost + 0.35 * mae + risk * tf.maximum(min_move, price_cost), 0.0)
    cost_risk_loss = tf.reduce_sum(trade_probability * risk_cost / move_scale * sample_weight) / tf.maximum(
        tf.reduce_sum(sample_weight),
        epsilon,
    )

    opportunity = tf.sigmoid((move_abs - price_cost - min_move) / move_scale)
    safe_fraction = tf.clip_by_value(1.0 - 0.25 * risk, 0.0, 1.0)
    target_activity = _config_value(financial_loss_config, "targetTradeProbability") * opportunity * safe_fraction
    activity_loss = tf.reduce_sum(tf.square(trade_probability - target_activity) * sample_weight) / tf.maximum(
        tf.reduce_sum(sample_weight),
        epsilon,
    )

    expected_net = buy_probability * (signed_move - price_cost) + sell_probability * (-signed_move - price_cost)
    expected_net = expected_net - trade_probability * risk_cost
    downside_loss = tf.reduce_sum(tf.nn.relu(-expected_net / move_scale) * sample_weight) / tf.maximum(
        tf.reduce_sum(sample_weight),
        epsilon,
    )

    return (
        _config_value(financial_loss_config, "classificationWeight") * classification_loss
        + _config_value(financial_loss_config, "moveWeight") * move_loss
        + _config_value(financial_loss_config, "quantileWeight") * quantile_loss
        + _config_value(financial_loss_config, "adverseTailWeight") * adverse_tail_loss
        + _config_value(financial_loss_config, "costRiskWeight") * cost_risk_loss
        + _config_value(financial_loss_config, "activityWeight") * activity_loss
        + _config_value(financial_loss_config, "downsideUtilityWeight") * downside_loss
    )
