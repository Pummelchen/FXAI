from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log1p, tanh
from typing import Iterable

PLUGIN_NAME = "ai_nhits"
ARCHITECTURE_MODE = "nhits"

POSITIVE_TERMS = {"growth", "hawkish", "strong", "beat", "risk-on", "expansion", "surplus", "upgrade", "resilient"}
NEGATIVE_TERMS = {"recession", "dovish", "weak", "miss", "risk-off", "contraction", "deficit", "downgrade", "stress"}
POLICY_TERMS = {"rates", "central", "bank", "fed", "ecb", "boj", "boe", "rba", "inflation", "cpi"}
VOLATILITY_TERMS = {"surprise", "shock", "volatility", "payrolls", "geopolitical", "sanctions", "liquidity"}
CURRENCY_TERMS = {"usd", "eur", "jpy", "gbp", "chf", "cad", "aud", "nzd", "cny"}


@dataclass(frozen=True)
class NLPFeatures:
    sentiment: float
    volatility_pressure: float
    policy_pressure: float
    currency_focus: float
    novelty: float
    token_count: int


def _tokens(text: str) -> list[str]:
    return [token.strip(".,:;!?()[]{}\"'").lower() for token in text.split() if token.strip()]


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return ["_".join(tokens[index:index + n]) for index in range(max(0, len(tokens) - n + 1))]


def extract_event_features(texts: Iterable[str]) -> NLPFeatures:
    tokens: list[str] = []
    for text in texts:
        tokens.extend(_tokens(text))
    grams = tokens + _ngrams(tokens, 2)
    counts = Counter(grams)
    total = max(sum(counts.values()), 1)
    pos = sum(counts[token] for token in POSITIVE_TERMS)
    neg = sum(counts[token] for token in NEGATIVE_TERMS)
    policy = sum(counts[token] for token in POLICY_TERMS)
    volatility = sum(counts[token] for token in VOLATILITY_TERMS)
    currency = sum(counts[token] for token in CURRENCY_TERMS)
    unique = len(counts)
    return NLPFeatures(
        sentiment=tanh((pos - neg) / total * 8.0),
        volatility_pressure=tanh(volatility / total * 10.0),
        policy_pressure=tanh(policy / total * 8.0),
        currency_focus=tanh(currency / total * 10.0),
        novelty=tanh(log1p(unique) / log1p(total + 1.0)),
        token_count=len(tokens),
    )


def merge_into_numeric_features(features: list[float], texts: Iterable[str]) -> list[float]:
    merged = list(features)
    if len(merged) < 84:
        merged.extend([0.0] * (84 - len(merged)))
    nlp = extract_event_features(texts)
    merged[55] = 0.60 * merged[55] + 0.40 * nlp.sentiment
    merged[56] = 0.60 * merged[56] + 0.40 * nlp.volatility_pressure
    merged[57] = 0.60 * merged[57] + 0.40 * nlp.policy_pressure
    merged[58] = max(merged[58], min(nlp.token_count / 128.0, 1.0))
    merged[59] = 0.65 * merged[59] + 0.35 * nlp.currency_focus
    merged[60] = 0.65 * merged[60] + 0.35 * nlp.novelty
    return merged
