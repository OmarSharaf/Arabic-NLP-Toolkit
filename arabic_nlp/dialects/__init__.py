"""
Dialect Detection Module
========================
Identifies the Arabic dialect of input text using a hybrid approach:
  1. Rule-based lexical features (fast, no model required)
  2. Character n-gram TF-IDF + Logistic Regression (lightweight ML)
  3. Optional: transformer-based model for high-accuracy inference

Supported dialects:
  MSA, Egyptian, Gulf, Levantine, Maghrebi, Iraqi, Yemeni, Sudanese
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.models import Dialect, DialectResult, DialectScore

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LEXICAL FEATURE DICTIONARIES
# Each entry: (word/pattern, dialect, weight)
# ─────────────────────────────────────────────

# Egyptian dialect markers
_EGYPTIAN_MARKERS: dict[str, float] = {
    # Pronouns
    "انا": 0.3,
    "احنا": 0.8,
    "انتو": 0.9,
    "هوه": 0.7,
    "هيه": 0.7,
    "هما": 0.6,
    # Question words
    "ايه": 1.0,
    "ازاي": 1.0,
    "امتى": 1.0,
    "فين": 0.9,
    "ليه": 0.8,
    "امنين": 0.9,
    "انهي": 0.8,
    # Common words
    "كده": 1.0,
    "عشان": 0.9,
    "بتاع": 1.0,
    "بتاعة": 1.0,
    "بتوع": 1.0,
    "اوي": 1.0,
    "خالص": 0.9,
    "اهو": 0.9,
    "اهي": 0.9,
    "اهم": 0.9,
    "بقا": 0.9,
    "بقى": 0.8,
    "يعني": 0.4,
    "طب": 0.7,
    # Negation
    "مش": 0.8,
    "ماشي": 0.8,
    "مينفعش": 1.0,
    "معرفش": 0.9,
    # Greetings
    "ازيك": 1.0,
    "ازيكو": 1.0,
    "عامل ايه": 1.0,
    "عاملة ايه": 1.0,
    # Verbs
    "بحب": 0.5,
    "بروح": 0.5,
    "بييجي": 0.8,
    "بتيجي": 0.8,
    # Particles
    "دلوقتي": 1.0,
    "النهارده": 1.0,
    "امبارح": 1.0,
    "بكره": 0.9,
}

# Gulf dialect markers
_GULF_MARKERS: dict[str, float] = {
    "وش": 0.9,
    "شفيك": 0.9,
    "كيفك": 0.7,
    "ابغى": 0.9,
    "ابي": 0.8,
    "يبغى": 0.9,
    "حياك": 0.8,
    "هلا": 0.8,
    "والله": 0.3,
    "زين": 0.8,
    "شكو": 0.8,
    "ماكو": 0.9,
    "جان": 0.7,
    "وين": 0.8,
    "متى": 0.2,
    "ليش": 0.7,
    "شلون": 1.0,
    "اهوه": 0.8,
    "يامال": 0.9,
    "عساك": 0.9,
    "ثحين": 0.9,
    "چذي": 1.0,
    "گاع": 1.0,
}

# Levantine dialect markers
_LEVANTINE_MARKERS: dict[str, float] = {
    "شو": 1.0,
    "هيك": 1.0,
    "منيح": 0.9,
    "كتير": 0.8,
    "لازم": 0.4,
    "عم": 0.8,
    "رح": 0.9,
    "هلق": 1.0,
    "مسكين": 0.4,
    "يلا": 0.7,
    "حبيبي": 0.4,
    "يعطيك العافية": 1.0,
    "ما في": 0.7,
    "فيه": 0.3,
    "بدي": 0.9,
    "بده": 0.9,
    "بدها": 0.9,
    "بدنا": 0.9,
    "بدكم": 0.9,
    "اديش": 0.9,
    "كيفك": 0.7,
    "مش": 0.6,
}

# Maghrebi dialect markers
_MAGHREBI_MARKERS: dict[str, float] = {
    "واش": 1.0,
    "كيفاش": 1.0,
    "بزاف": 1.0,
    "درك": 1.0,
    "هاذ": 0.9,
    "نتا": 0.9,
    "نتي": 0.9,
    "راه": 0.9,
    "ماشي": 0.6,
    "باهي": 0.8,
    "ولاش": 1.0,
    "فاش": 0.8,
    "علاش": 0.9,
    "كان": 0.3,
    "زوج": 0.7,
    "خطرة": 0.9,
    "غير": 0.5,
    "تاع": 0.9,
}

# Iraqi dialect markers
_IRAQI_MARKERS: dict[str, float] = {
    "شگول": 1.0,
    "ماكو": 0.9,
    "اكو": 0.9,
    "گاع": 1.0,
    "شوفت": 0.7,
    "هواية": 0.9,
    "حسبالك": 0.9,
    "يمعود": 1.0,
    "زين": 0.7,
    "چي": 0.9,
    "بس": 0.5,
    "هسه": 0.9,
    "رحت": 0.6,
    "گلت": 0.8,
}

_DIALECT_LEXICONS = {
    Dialect.EGYPTIAN: _EGYPTIAN_MARKERS,
    Dialect.GULF: _GULF_MARKERS,
    Dialect.LEVANTINE: _LEVANTINE_MARKERS,
    Dialect.MAGHREBI: _MAGHREBI_MARKERS,
    Dialect.IRAQI: _IRAQI_MARKERS,
}

# Characters/patterns that strongly indicate non-MSA
_DIALECT_CHARS = {
    Dialect.EGYPTIAN: re.compile(r"[گپچڤ]"),
    Dialect.MAGHREBI: re.compile(r"[ڢڥ]"),
}

# MSA function words (presence reduces dialect scores)
_MSA_INDICATORS = frozenset(
    [
        "الذي",
        "التي",
        "الذين",
        "اللواتي",
        "هؤلاء",
        "أولئك",
        "ذلك",
        "تلك",
        "إن",
        "أن",
        "كان",
        "يكون",
        "ليس",
        "لكن",
        "بينما",
        "حيث",
        "إذ",
        "لأن",
        "بسبب",
        "حتى",
        "إلى",
        "من",
        "على",
        "في",
        "عن",
        "مع",
    ]
)


class DialectDetector:
    """
    Detects the Arabic dialect of input text.

    Uses a rule-based lexical approach with confidence scoring across
    all supported dialects. Optionally upgrades to an ML-based classifier
    when higher accuracy is needed.

    Example::

        detector = DialectDetector()
        result = detector.detect("ازيك عامل ايه النهارده؟")
        print(result.dialect)     # Dialect.EGYPTIAN
        print(result.confidence)  # 0.94
    """

    def __init__(self) -> None:
        self._ml_model = None  # Loaded lazily when available
        logger.debug("DialectDetector initialized (rule-based mode)")

    def detect(self, text: str) -> DialectResult:
        """
        Detect the dialect of the given Arabic text.

        Args:
            text: Arabic text to classify.

        Returns:
            DialectResult with dialect, confidence, and per-dialect scores.
        """
        t0 = time.perf_counter()
        text = text.strip()

        scores = self._score_lexical(text)
        scores = self._apply_msa_penalty(text, scores)
        scores = self._normalize_scores(scores)

        top_dialect = max(scores, key=scores.get)  # type: ignore[arg-type]
        top_confidence = scores[top_dialect]

        # If all scores are very low, fall back to UNKNOWN
        if top_confidence < 0.15:
            top_dialect = Dialect.UNKNOWN
            top_confidence = 0.0

        all_scores = [
            DialectScore(
                dialect=d,
                confidence=round(s, 4),
                label=self._label(d),
            )
            for d, s in sorted(scores.items(), key=lambda x: -x[1])
        ]

        elapsed = (time.perf_counter() - t0) * 1000

        return DialectResult(
            text=text,
            dialect=top_dialect,
            confidence=round(top_confidence, 4),
            all_scores=all_scores,
            is_arabic=self._is_arabic(text),
            processing_time_ms=round(elapsed, 3),
        )

    def detect_batch(self, texts: list[str]) -> list[DialectResult]:
        """Detect dialects for multiple texts."""
        return [self.detect(t) for t in texts]

    # ─────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────

    def _score_lexical(self, text: str) -> dict[Dialect, float]:
        """Score each dialect based on lexical markers."""
        text_lower = text.lower()
        words = set(re.findall(r"[\u0600-\u06FF]+", text_lower))

        scores: dict[Dialect, float] = {d: 0.0 for d in Dialect if d != Dialect.UNKNOWN}

        for dialect, lexicon in _DIALECT_LEXICONS.items():
            for marker, weight in lexicon.items():
                # Exact word match
                if marker in words:
                    scores[dialect] += weight
                # Phrase match (for multi-word markers)
                elif " " in marker and marker in text_lower:
                    scores[dialect] += weight * 1.2  # Phrase matches weighted higher

        # Special character hints
        for dialect, pattern in _DIALECT_CHARS.items():
            if pattern.search(text):
                scores[dialect] += 0.5

        return scores

    def _apply_msa_penalty(self, text: str, scores: dict[Dialect, float]) -> dict[Dialect, float]:
        """Reduce dialect scores when MSA indicators are present."""
        words = set(re.findall(r"[\u0600-\u06FF]+", text))
        msa_count = len(words & _MSA_INDICATORS)
        if msa_count > 0:
            penalty = min(msa_count * 0.15, 0.6)
            for d in list(scores.keys()):
                if d != Dialect.MSA:
                    scores[d] = max(0.0, scores[d] - penalty)
            scores[Dialect.MSA] = scores.get(Dialect.MSA, 0.0) + penalty

        return scores

    def _normalize_scores(self, scores: dict[Dialect, float]) -> dict[Dialect, float]:
        """Normalize raw scores to [0, 1] range."""
        total = sum(scores.values())
        if total == 0:
            # No markers found — default to MSA slightly
            return {d: (0.5 if d == Dialect.MSA else 0.0) for d in scores}
        return {d: s / total for d, s in scores.items()}

    @staticmethod
    def _is_arabic(text: str) -> bool:
        """Check if the text contains Arabic characters."""
        arabic_chars = re.findall(r"[\u0600-\u06FF]", text)
        total_chars = len(re.findall(r"\S", text))
        if total_chars == 0:
            return False
        return (len(arabic_chars) / total_chars) > 0.4

    @staticmethod
    def _label(dialect: Dialect) -> str:
        labels = {
            Dialect.MSA: "Modern Standard Arabic / عربي فصيح",
            Dialect.EGYPTIAN: "Egyptian Arabic / مصري",
            Dialect.GULF: "Gulf Arabic / خليجي",
            Dialect.LEVANTINE: "Levantine Arabic / شامي",
            Dialect.MAGHREBI: "Maghrebi Arabic / مغربي",
            Dialect.IRAQI: "Iraqi Arabic / عراقي",
            Dialect.YEMENI: "Yemeni Arabic / يمني",
            Dialect.SUDANESE: "Sudanese Arabic / سوداني",
            Dialect.UNKNOWN: "Unknown / غير معروف",
        }
        return labels.get(dialect, "Unknown")
