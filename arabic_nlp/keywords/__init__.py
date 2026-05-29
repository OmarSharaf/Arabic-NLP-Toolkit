"""
Arabic Keyword Extraction
=========================
TF-based keyword extraction with stopword filtering, optional stemming,
and dialect-aware stopword lists.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter

from pydantic import BaseModel, Field

from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.models import Dialect

logger = logging.getLogger(__name__)

_ARABIC_WORD = re.compile(r"[\u0600-\u06FF\u0750-\u077F]+")
_MIN_WORD_LEN = 2


class Keyword(BaseModel):
    """A single extracted keyword with relevance score."""

    text: str
    score: float = Field(ge=0.0, le=1.0)
    frequency: int = Field(ge=1)
    rank: int = Field(ge=1)

    model_config = {"frozen": True}


class KeywordResult(BaseModel):
    """Keyword extraction result for a document."""

    text: str
    keywords: list[Keyword] = Field(default_factory=list)
    total_tokens: int = Field(ge=0)
    content_tokens: int = Field(ge=0)
    dialect: Dialect = Dialect.UNKNOWN
    processing_time_ms: float = Field(ge=0.0)

    model_config = {"frozen": True}

    def top(self, n: int = 5) -> list[Keyword]:
        """Return the top *n* keywords."""
        return self.keywords[:n]

    def as_dict(self) -> dict[str, float]:
        """Map keyword text → score (for quick downstream use)."""
        return {kw.text: kw.score for kw in self.keywords}


class KeywordExtractor:
    """
    Extract salient keywords from Arabic text.

    Uses term frequency with length and position weighting, filtered by
    dialect stopwords. Optional light stemming improves recall on inflected forms.

    Example::

        extractor = KeywordExtractor()
        result = extractor.extract(
            "الذكاء الاصطناعي يغير مستقبل التعليم في العالم العربي",
            top_n=5,
        )
        for kw in result.keywords:
            print(kw.text, kw.score)
    """

    def __init__(
        self,
        *,
        min_word_length: int = _MIN_WORD_LEN,
        use_stemming: bool = False,
    ) -> None:
        self._min_word_length = min_word_length
        self._use_stemming = use_stemming
        self._stopwords: object | None = None
        self._stemmer: object | None = None

    def _get_stopwords(self):
        if self._stopwords is None:
            from arabic_nlp.stopwords import StopWords

            self._stopwords = StopWords()
        return self._stopwords

    def _get_stemmer(self):
        if self._stemmer is None:
            from arabic_nlp.stemmer import ArabicStemmer

            self._stemmer = ArabicStemmer()
        return self._stemmer

    def extract(
        self,
        text: str,
        *,
        top_n: int = 10,
        dialect: Dialect | None = None,
        min_score: float = 0.0,
    ) -> KeywordResult:
        """
        Extract ranked keywords from Arabic text.

        Args:
            text: Input document or passage.
            top_n: Maximum keywords to return.
            dialect: Stopword dialect (auto-detected if omitted).
            min_score: Minimum normalized score (0–1) to include.
        """
        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError("Text must be a non-empty string")

        t0 = time.perf_counter()
        clean = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text.strip())

        if dialect is None:
            from arabic_nlp.dialects import DialectDetector

            dialect = DialectDetector().detect(clean).dialect

        sw = self._get_stopwords().get(dialect) | self._get_stopwords().get(Dialect.MSA)

        tokens = _ARABIC_WORD.findall(clean)
        total_tokens = len(tokens)

        if self._use_stemming:
            stemmer = self._get_stemmer()
            from arabic_nlp.stemmer import StemmerMode

            tokens = [
                stemmer.stem(w, mode=StemmerMode.LIGHT).stem
                for w in tokens
                if len(w) >= self._min_word_length
            ]

        content = [w for w in tokens if len(w) >= self._min_word_length and w not in sw]
        content_tokens = len(content)

        if not content:
            elapsed = (time.perf_counter() - t0) * 1000
            return KeywordResult(
                text=text,
                keywords=[],
                total_tokens=total_tokens,
                content_tokens=0,
                dialect=dialect,
                processing_time_ms=round(elapsed, 3),
            )

        counts: Counter[str] = Counter(content)
        max_freq = max(counts.values())
        max_len = max(len(w) for w in counts)

        scored: list[tuple[str, float, int]] = []
        for word, freq in counts.items():
            tf = freq / max_freq
            length_boost = 0.5 + 0.5 * (len(word) / max_len)
            raw = tf * length_boost
            scored.append((word, raw, freq))

        scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
        top = scored[:top_n]

        if top:
            max_raw = top[0][1]
            keywords = [
                Keyword(
                    text=word,
                    score=round(raw / max_raw if max_raw else 0.0, 4),
                    frequency=freq,
                    rank=i + 1,
                )
                for i, (word, raw, freq) in enumerate(top)
                if (raw / max_raw if max_raw else 0.0) >= min_score
            ]
        else:
            keywords = []

        elapsed = (time.perf_counter() - t0) * 1000
        return KeywordResult(
            text=text,
            keywords=keywords,
            total_tokens=total_tokens,
            content_tokens=content_tokens,
            dialect=dialect,
            processing_time_ms=round(elapsed, 3),
        )

    def extract_batch(
        self,
        texts: list[str],
        *,
        top_n: int = 10,
        dialect: Dialect | None = None,
    ) -> list[KeywordResult]:
        """Extract keywords from multiple texts."""
        return [self.extract(t, top_n=top_n, dialect=dialect) for t in texts]
