from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import tanh
from typing import Iterable

PLUGIN_NAME = "ai_chronos"
ARCHITECTURE_MODE = "causalTokenForecaster"

POSITIVE_TERMS = {"growth", "hawkish", "strong", "beat", "risk-on", "expansion", "surplus"}
NEGATIVE_TERMS = {"recession", "dovish", "weak", "miss", "risk-off", "contraction", "deficit"}
VOLATILITY_TERMS = {"surprise", "shock", "inflation", "rates", "payrolls", "cpi", "central", "bank"}


@dataclass(frozen=True)
class NLPFeatures:
    sentiment: float
    volatility_pressure: float
    policy_pressure: float
    token_count: int


def _tokens(text: str) -> list[str]:
    return [token.strip(".,:;!?()[]{}\"'").lower() for token in text.split() if token.strip()]


def extract_event_features(texts: Iterable[str]) -> NLPFeatures:
    tokens = []
    for text in texts:
        tokens.extend(_tokens(text))
    counts = Counter(tokens)
    total = max(sum(counts.values()), 1)
    pos = sum(counts[t] for t in POSITIVE_TERMS)
    neg = sum(counts[t] for t in NEGATIVE_TERMS)
    vol = sum(counts[t] for t in VOLATILITY_TERMS)
    policy = counts["rates"] + counts["central"] + counts["bank"] + counts["fed"] + counts["ecb"]
    return NLPFeatures(
        sentiment=tanh((pos - neg) / total * 6.0),
        volatility_pressure=tanh(vol / total * 8.0),
        policy_pressure=tanh(policy / total * 8.0),
        token_count=total,
    )


def merge_into_numeric_features(features: list[float], texts: Iterable[str]) -> list[float]:
    merged = list(features)
    if len(merged) < 84:
        merged.extend([0.0] * (84 - len(merged)))
    nlp = extract_event_features(texts)
    merged[55] = 0.65 * merged[55] + 0.35 * nlp.sentiment
    merged[56] = 0.65 * merged[56] + 0.35 * nlp.volatility_pressure
    merged[57] = 0.65 * merged[57] + 0.35 * nlp.policy_pressure
    merged[58] = max(merged[58], min(nlp.token_count / 128.0, 1.0))
    return merged
