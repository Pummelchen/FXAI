from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DemoPluginTemplateTextEvent:
    event_time_utc: int
    source: str
    headline: str
    body: str = ""
    importance: float = 0.0


def tokenize_event(event: DemoPluginTemplateTextEvent, max_tokens: int = 64) -> list[str]:
    text = f"{event.headline} {event.body}".strip().lower()
    cleaned = "".join(character if character.isalnum() else " " for character in text)
    return cleaned.split()[:max(1, max_tokens)]


def event_feature_vector(events: Iterable[DemoPluginTemplateTextEvent], max_tokens: int = 64) -> list[float]:
    token_count = 0
    weighted_importance = 0.0
    event_count = 0
    for event in events:
        tokens = tokenize_event(event, max_tokens=max_tokens)
        token_count += len(tokens)
        weighted_importance += max(0.0, min(1.0, event.importance))
        event_count += 1
    if event_count == 0:
        return [0.0, 0.0, 1.0]
    return [
        min(1.0, token_count / float(max(1, event_count * max_tokens))),
        weighted_importance / float(event_count),
        0.0,
    ]
