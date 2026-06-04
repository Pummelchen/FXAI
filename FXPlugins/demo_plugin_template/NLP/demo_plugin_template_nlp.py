from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class DemoPluginTemplateTextEvent:
    event_time_utc: int
    source: str
    headline: str
    body: str = ""
    importance: float = 0.0


def _tokenize_text(text: str, max_tokens: int = 64) -> list[str]:
    cleaned = "".join(character if character.isalnum() else " " for character in text.lower())
    return cleaned.split()[:max(1, max_tokens)]


def tokenize_event(event: DemoPluginTemplateTextEvent, max_tokens: int = 64) -> list[str]:
    text = f"{event.headline} {event.body}".strip().lower()
    return _tokenize_text(text, max_tokens=max_tokens)


def event_feature_vector(
    events: Iterable[DemoPluginTemplateTextEvent | str],
    max_tokens: int = 64,
) -> list[float]:
    token_count = 0
    weighted_importance = 0.0
    event_count = 0
    for event in events:
        if isinstance(event, DemoPluginTemplateTextEvent):
            tokens = tokenize_event(event, max_tokens=max_tokens)
            importance = event.importance
        else:
            tokens = _tokenize_text(str(event), max_tokens=max_tokens)
            importance = 0.0
        token_count += len(tokens)
        weighted_importance += max(0.0, min(1.0, importance))
        event_count += 1
    if event_count == 0:
        return [0.0, 0.0, 1.0]
    return [
        min(1.0, token_count / float(max(1, event_count * max_tokens))),
        weighted_importance / float(event_count),
        0.0,
    ]


def merge_into_numeric_features(features: list[float], texts: Iterable[str]) -> list[float]:
    """Foundation NLP bridge entry point used by fxai_plugin_module_backend.py."""
    merged = list(features)
    if len(merged) < 84:
        merged.extend([0.0] * (84 - len(merged)))
    token_density, weighted_importance, no_text_flag = event_feature_vector(texts)
    merged[55] = 0.65 * merged[55] + 0.35 * token_density
    merged[56] = 0.65 * merged[56] + 0.35 * weighted_importance
    merged[57] = 0.65 * merged[57] + 0.35 * (1.0 - no_text_flag)
    return merged


def _batch_size(features: Iterable[Iterable[float]]) -> int:
    try:
        rows = list(features)
    except TypeError:
        return 1
    if not rows:
        return 0
    first = rows[0]
    if isinstance(first, (int, float)):
        return 1
    try:
        iter(first)
    except TypeError:
        return 1
    return len(rows)


def predict_batch(
    features: Iterable[Iterable[float]],
    state: Optional[object] = None,
    data_has_volume: bool = True,
    texts: Optional[Iterable[str]] = None,
) -> dict[str, list[list[float]] | list[float]]:
    del state, data_has_volume
    batch_size = _batch_size(features)
    return {
        "class_probabilities": [[0.0, 0.0, 1.0] for _ in range(batch_size)],
        "move_mean_points": [0.0] * batch_size,
        "move_quantiles": [[0.0, 0.0, 0.0] for _ in range(batch_size)],
        "text_features": event_feature_vector(texts or []),
    }


def train_step(
    features: Iterable[Iterable[float]],
    labels: Iterable[int],
    moves: Iterable[float],
    state: Optional[object] = None,
    data_has_volume: bool = True,
) -> object:
    del features, labels, moves, data_has_volume
    return state or {}
