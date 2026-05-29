"""
Sentiment Analysis Module
=========================
Analyzes the sentiment of Arabic text using a hybrid approach:
  1. Lexicon-based scoring across MSA and dialect-specific word lists
  2. Negation handling (ما، مش، لا، ليس، غير)
  3. Intensifier detection (جداً، أوي، كتير، بزاف)
  4. Dialect-aware scoring

Sentiment labels: POSITIVE, NEGATIVE, NEUTRAL, MIXED
"""

from __future__ import annotations

import logging
import re
import time
from typing import NamedTuple

from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.models import (
    Dialect,
    SentimentLabel,
    SentimentResult,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SENTIMENT LEXICONS
# Score range: -1.0 (very negative) to +1.0 (very positive)
# ─────────────────────────────────────────────

_POSITIVE_WORDS: dict[str, float] = {
    # Strong positive
    "ممتاز": 0.9,
    "رائع": 0.9,
    "تحفة": 1.0,
    "مذهل": 0.95,
    "عظيم": 0.85,
    "بديع": 0.85,
    "روعة": 0.9,
    "جميل": 0.8,
    "حلو": 0.75,
    "تمام": 0.8,
    "أحسن": 0.85,
    "احسن": 0.85,
    "أفضل": 0.8,
    "افضل": 0.8,
    "فخم": 0.8,
    "خرافي": 0.95,
    "اسطوري": 0.9,
    # Moderate positive
    "كويس": 0.6,
    "حسن": 0.6,
    "طيب": 0.55,
    "مريح": 0.6,
    "مفيد": 0.65,
    "صح": 0.5,
    "معقول": 0.45,
    "مقبول": 0.4,
    # Emotions
    "سعيد": 0.75,
    "فرحان": 0.75,
    "مبسوط": 0.7,
    "مسرور": 0.75,
    "محبوب": 0.7,
    "محتاج": 0.3,
    "راضي": 0.65,
    # Love / approval
    "بحب": 0.8,
    "أحب": 0.8,
    "احب": 0.8,
    "حبيت": 0.75,
    "نحب": 0.75,
    "أوصي": 0.7,
    "اوصي": 0.7,
    "ينصح": 0.65,
    "ممتنن": 0.8,
    "شكراً": 0.6,
    "شكرا": 0.6,
    "برافو": 0.8,
    "ماشاء الله": 0.85,
    "الحمد لله": 0.5,
    # Egyptian slang
    "اوي": 0.3,
    "عسل": 0.85,
    "ناصع": 0.75,
    "مظبوط": 0.6,
}

_NEGATIVE_WORDS: dict[str, float] = {
    # Strong negative
    "سيء": -0.9,
    "وحش": -0.85,
    "بايخ": -0.85,
    "مريب": -0.7,
    "فاشل": -0.85,
    "كريه": -0.9,
    "مزعج": -0.75,
    "بيع": -0.4,
    "مخيب": -0.8,
    "محبط": -0.8,
    "محزن": -0.7,
    "صعب": -0.4,
    "مؤلم": -0.75,
    "تعبان": -0.6,
    "مريض": -0.5,
    "غلط": -0.7,
    # Moderate negative
    "مش كويس": -0.6,
    "مش تمام": -0.6,
    "مش حلو": -0.6,
    "ضعيف": -0.6,
    "رديء": -0.8,
    "وضيع": -0.75,
    "بغيض": -0.85,
    # Frustration / dissatisfaction
    "مضايق": -0.65,
    "زهقت": -0.55,
    "تعبت": -0.5,
    "مللت": -0.45,
    "خيبة": -0.8,
    "خايب": -0.8,
    "مخيف": -0.55,
    "مقرف": -0.9,
    # Profanity placeholder (sanitized)
    "للأسف": -0.55,
    "مؤسف": -0.75,
    "مخزي": -0.8,
    "عيب": -0.6,
    "وحشة": -0.7,
    "بايظ": -0.85,
}

# Intensifiers multiply the adjacent word score
_INTENSIFIERS: dict[str, float] = {
    "جداً": 1.5,
    "جدا": 1.5,
    "أوي": 1.4,
    "اوي": 1.4,
    "كتير": 1.3,
    "كثيراً": 1.3,
    "كثيرا": 1.3,
    "بزاف": 1.35,
    "للغاية": 1.5,
    "تماماً": 1.3,
    "تماما": 1.3,
    "قوي": 1.2,
    "مرة": 1.25,
    "جزيل": 1.2,
    "شديد": 1.3,
    "شديداً": 1.3,
}

# Negation words flip sentiment of following token
_NEGATION_WORDS: frozenset[str] = frozenset(
    [
        "ما",
        "مش",
        "مو",
        "لا",
        "لم",
        "لن",
        "ليس",
        "ليست",
        "غير",
        "بدون",
        "عدم",
        "مستحيل",
        "ممنوع",
        "مش ممكن",
    ]
)

_NEGATION_WINDOW = 3  # Words after negation that get flipped


class _ScoredToken(NamedTuple):
    word: str
    score: float
    is_negated: bool
    is_intensified: bool


class SentimentAnalyzer:
    """
    Lexicon-based Arabic sentiment analyzer.

    Handles negation, intensifiers, and works across dialects.

    Example::

        analyzer = SentimentAnalyzer()

        result = analyzer.analyze("المنتج ده رائع جداً وبحبه أوي")
        print(result.label)           # SentimentLabel.POSITIVE
        print(result.score)           # 0.89
        print(result.positive_score)  # 0.89
        print(result.negative_score)  # 0.04

        # With negation
        result = analyzer.analyze("المنتج ده مش كويس خالص")
        print(result.label)   # SentimentLabel.NEGATIVE
    """

    def analyze(
        self,
        text: str,
        *,
        dialect: Dialect | None = None,
    ) -> SentimentResult:
        """
        Analyze sentiment of Arabic text.

        Args:
            text: Input Arabic text.
            dialect: Optional dialect hint (does not affect rule-based scoring
                     but is preserved in the result for reference).

        Returns:
            SentimentResult with label, score, and per-class probabilities.
        """
        t0 = time.perf_counter()

        # Normalize input for matching
        normalized = self._prep(text)
        words = normalized.split()

        scored_tokens = self._score_tokens(words)
        pos_score, neg_score, neu_score = self._aggregate(scored_tokens)

        total = pos_score + neg_score + neu_score
        if total == 0:
            pos_score = neg_score = 0.0
            neu_score = 1.0
            total = 1.0

        pos_prob = pos_score / total
        neg_prob = neg_score / total
        neu_prob = neu_score / total

        label, confidence = self._decide_label(pos_prob, neg_prob, neu_prob)

        elapsed = (time.perf_counter() - t0) * 1000

        return SentimentResult(
            text=text,
            label=label,
            score=round(confidence, 4),
            positive_score=round(pos_prob, 4),
            negative_score=round(neg_prob, 4),
            neutral_score=round(neu_prob, 4),
            dialect=dialect or Dialect.UNKNOWN,
            processing_time_ms=round(elapsed, 3),
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment for multiple texts."""
        return [self.analyze(t) for t in texts]

    # ─────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────

    def _score_tokens(self, words: list[str]) -> list[_ScoredToken]:
        """Score each token considering negation and intensifiers."""
        scored: list[_ScoredToken] = []
        negation_countdown = 0
        last_intensifier = 1.0

        for word in words:
            is_negated = negation_countdown > 0
            if negation_countdown > 0:
                negation_countdown -= 1

            if word in _NEGATION_WORDS:
                negation_countdown = _NEGATION_WINDOW
                continue

            if word in _INTENSIFIERS:
                last_intensifier = _INTENSIFIERS[word]
                continue

            # Look up in lexicons
            score = _POSITIVE_WORDS.get(word, 0.0) + _NEGATIVE_WORDS.get(word, 0.0)

            if score != 0.0:
                score *= last_intensifier
                if is_negated:
                    score = -score * 0.8  # Flip and slightly dampen
                scored.append(_ScoredToken(word, score, is_negated, last_intensifier != 1.0))
                last_intensifier = 1.0  # Reset after use

        return scored

    def _aggregate(self, scored: list[_ScoredToken]) -> tuple[float, float, float]:
        """Aggregate token scores into positive/negative/neutral totals."""
        pos = sum(s.score for s in scored if s.score > 0)
        neg = sum(-s.score for s in scored if s.score < 0)

        # Neutral base — gives texts with no opinion words a neutral score
        neutral_base = max(0.2, 1.0 - len(scored) * 0.05)
        neu = neutral_base if not scored else max(0.1, neutral_base - (pos + neg) * 0.1)

        return pos, neg, neu

    def _decide_label(self, pos: float, neg: float, neu: float) -> tuple[SentimentLabel, float]:
        """Decide the final label and confidence."""
        if pos > neg and pos > neu:
            if neg > 0.2 * pos:
                return SentimentLabel.MIXED, pos
            return SentimentLabel.POSITIVE, pos
        elif neg > pos and neg > neu:
            if pos > 0.2 * neg:
                return SentimentLabel.MIXED, neg
            return SentimentLabel.NEGATIVE, neg
        else:
            return SentimentLabel.NEUTRAL, neu

    @staticmethod
    def _prep(text: str) -> str:
        """Lightweight preparation for lexicon matching."""
        # Remove diacritics
        text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
        # Normalize alef
        text = re.sub(r"[أإآ]", "ا", text)
        # Remove tatweel
        text = text.replace("\u0640", "")
        # Lowercase latin
        text = text.lower()
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
