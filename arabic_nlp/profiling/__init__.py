"""
Arabic Text Profiling
=====================
Classify register, detect social-media signals, and score text quality
for downstream filtering and analytics pipelines.
"""

from __future__ import annotations

import logging
import re
import time
from enum import Enum

from pydantic import BaseModel, Field

from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.models import Dialect

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#[\u0600-\u06FF\w]+")
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "]",
    flags=re.UNICODE,
)
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_LATIN_RE = re.compile(r"[a-zA-Z]{2,}")


class TextRegister(str, Enum):
    """Estimated linguistic register of the text."""

    FORMAL_MSA = "formal_msa"
    SEMI_FORMAL = "semi_formal"
    DIALECTAL = "dialectal"
    SOCIAL_MEDIA = "social_media"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class TextProfile(BaseModel):
    """Comprehensive profile of an Arabic text sample."""

    text: str
    text_register: TextRegister
    register_confidence: float = Field(ge=0.0, le=1.0)
    dialect: Dialect = Dialect.UNKNOWN
    dialect_confidence: float = Field(ge=0.0, le=1.0)
    is_arabic: bool = True
    arabic_ratio: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Heuristic quality score (1.0 = clean formal prose)",
    )
    has_urls: bool = False
    has_mentions: bool = False
    has_hashtags: bool = False
    has_emojis: bool = False
    has_repeated_chars: bool = False
    has_latin_mix: bool = False
    word_count: int = Field(ge=0)
    sentence_count: int = Field(ge=0)
    avg_word_length: float = Field(ge=0.0)
    recommendations: list[str] = Field(default_factory=list)
    processing_time_ms: float = Field(ge=0.0)

    model_config = {"frozen": True}

    @property
    def register(self) -> TextRegister:
        """Alias for :attr:`text_register` (avoids Pydantic reserved name)."""
        return self.text_register

    @property
    def is_social_media(self) -> bool:
        return self.text_register == TextRegister.SOCIAL_MEDIA

    @property
    def needs_normalization(self) -> bool:
        return self.has_repeated_chars or self.has_emojis or self.quality_score < 0.6


class TextProfiler:
    """
    Profile Arabic text for register, dialect, and quality signals.

    Example::

        profiler = TextProfiler()
        profile = profiler.profile("ازيك يا صاحبي 😂 #مصر https://t.co/xyz")
        print(profile.register)        # TextRegister.SOCIAL_MEDIA
        print(profile.quality_score)   # lower for noisy social text
    """

    def profile(self, text: str) -> TextProfile:
        """
        Build a full text profile.

        Args:
            text: Arabic or mixed Arabic text.

        Raises:
            InvalidInputError: If text is empty.
        """
        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError("Text must be a non-empty string")

        t0 = time.perf_counter()
        text = text.strip()

        from arabic_nlp.dialects import DialectDetector
        from arabic_nlp.utils import detect_language, get_statistics

        stats = get_statistics(text)
        lang = detect_language(text)
        arabic_ratio = lang.get("arabic", 0.0)
        is_arabic = arabic_ratio >= 0.4

        dialect_result = DialectDetector().detect(text)

        has_urls = bool(_URL_RE.search(text))
        has_mentions = bool(_MENTION_RE.search(text))
        has_hashtags = bool(_HASHTAG_RE.search(text))
        has_emojis = bool(_EMOJI_RE.search(text))
        has_repeated = bool(_REPEAT_RE.search(text))
        has_latin = bool(_LATIN_RE.search(text))

        social_signals = sum([has_urls, has_mentions, has_hashtags, has_emojis])
        register, reg_conf = self._classify_register(
            dialect=dialect_result.dialect,
            dialect_conf=dialect_result.confidence,
            social_signals=social_signals,
            has_diacritics=stats.has_diacritics,
            avg_sent_len=stats.avg_sentence_length,
            arabic_ratio=arabic_ratio,
        )

        quality = self._quality_score(
            stats=stats,
            social_signals=social_signals,
            has_repeated=has_repeated,
            has_latin=has_latin,
            register=register,
        )

        recommendations = self._recommendations(
            register=register,
            quality=quality,
            has_urls=has_urls,
            has_repeated=has_repeated,
            stats=stats,
        )

        elapsed = (time.perf_counter() - t0) * 1000

        return TextProfile(
            text=text,
            text_register=register,
            register_confidence=round(reg_conf, 4),
            dialect=dialect_result.dialect,
            dialect_confidence=round(dialect_result.confidence, 4),
            is_arabic=is_arabic,
            arabic_ratio=arabic_ratio,
            quality_score=round(quality, 4),
            has_urls=has_urls,
            has_mentions=has_mentions,
            has_hashtags=has_hashtags,
            has_emojis=has_emojis,
            has_repeated_chars=has_repeated,
            has_latin_mix=has_latin,
            word_count=stats.total_words,
            sentence_count=stats.sentences,
            avg_word_length=stats.avg_word_length,
            recommendations=recommendations,
            processing_time_ms=round(elapsed, 3),
        )

    def profile_batch(self, texts: list[str]) -> list[TextProfile]:
        """Profile multiple texts."""
        return [self.profile(t) for t in texts]

    @staticmethod
    def _classify_register(
        *,
        dialect: Dialect,
        dialect_conf: float,
        social_signals: int,
        has_diacritics: bool,
        avg_sent_len: float,
        arabic_ratio: float,
    ) -> tuple[TextRegister, float]:
        if social_signals >= 2:
            return TextRegister.SOCIAL_MEDIA, min(0.7 + social_signals * 0.1, 0.98)

        if social_signals == 1:
            return TextRegister.MIXED, 0.75

        if dialect != Dialect.MSA and dialect != Dialect.UNKNOWN and dialect_conf >= 0.55:
            return TextRegister.DIALECTAL, dialect_conf

        if has_diacritics and avg_sent_len >= 12:
            return TextRegister.FORMAL_MSA, 0.85

        if avg_sent_len >= 8 and arabic_ratio >= 0.85:
            return TextRegister.SEMI_FORMAL, 0.78

        if dialect == Dialect.MSA and dialect_conf >= 0.5:
            return TextRegister.SEMI_FORMAL, dialect_conf

        return TextRegister.UNKNOWN, 0.5

    @staticmethod
    def _quality_score(
        *,
        stats,
        social_signals: int,
        has_repeated: bool,
        has_latin: bool,
        register: TextRegister,
    ) -> float:
        score = 1.0
        score -= social_signals * 0.12
        if has_repeated:
            score -= 0.15
        if has_latin and stats.arabic_ratio < 0.9:
            score -= 0.1
        if stats.total_words < 3:
            score -= 0.2
        if register == TextRegister.SOCIAL_MEDIA:
            score -= 0.1
        if stats.type_token_ratio < 0.3 and stats.total_words > 20:
            score -= 0.05
        return max(0.0, min(1.0, score))

    @staticmethod
    def _recommendations(
        *,
        register: TextRegister,
        quality: float,
        has_urls: bool,
        has_repeated: bool,
        stats,
    ) -> list[str]:
        recs: list[str] = []
        if register == TextRegister.SOCIAL_MEDIA:
            recs.append("Consider normalizer.clean_social() before NLP pipeline")
        if has_urls:
            recs.append("Strip URLs with normalize(remove_urls=True)")
        if has_repeated:
            recs.append("Repeated characters detected — apply social media cleanup")
        if quality < 0.5:
            recs.append("Low quality score — review input or preprocess further")
        if stats.has_diacritics:
            recs.append("Diacritized text — remove tashkeel if matching informal corpora")
        if not recs:
            recs.append("Text is suitable for standard NLP pipeline")
        return recs
