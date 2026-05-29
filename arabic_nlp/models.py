"""
Shared data models (DTOs) for arabic-nlp-toolkit.

All public-facing results are strongly-typed Pydantic models so that
consumers get IDE autocompletion, serialization, and validation for free.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────


class Dialect(str, Enum):
    """Supported Arabic dialects."""

    MSA = "msa"  # Modern Standard Arabic (الفصحى)
    EGYPTIAN = "egyptian"  # Egyptian Arabic (مصري)
    GULF = "gulf"  # Gulf Arabic (خليجي)
    LEVANTINE = "levantine"  # Levantine Arabic (شامي)
    MAGHREBI = "maghrebi"  # Maghrebi Arabic (مغربي)
    IRAQI = "iraqi"  # Iraqi Arabic (عراقي)
    YEMENI = "yemeni"  # Yemeni Arabic (يمني)
    SUDANESE = "sudanese"  # Sudanese Arabic (سوداني)
    UNKNOWN = "unknown"  # Could not determine


class SentimentLabel(str, Enum):
    """Sentiment polarity labels."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class EntityLabel(str, Enum):
    """Named entity types."""

    PERSON = "PERSON"  # أشخاص
    LOCATION = "LOCATION"  # أماكن
    ORGANIZATION = "ORGANIZATION"  # منظمات وشركات
    DATE = "DATE"  # تواريخ
    TIME = "TIME"  # أوقات
    MONEY = "MONEY"  # مبالغ مالية
    PRODUCT = "PRODUCT"  # منتجات
    EVENT = "EVENT"  # أحداث
    LANGUAGE = "LANGUAGE"  # لغات
    NATIONALITY = "NATIONALITY"  # جنسيات
    MISC = "MISC"  # متنوع


class TokenType(str, Enum):
    """Token type classification."""

    WORD = "word"
    PUNCTUATION = "punctuation"
    NUMBER = "number"
    ARABIC_NUMBER = "arabic_number"
    EMOJI = "emoji"
    LATIN = "latin"
    URL = "url"
    MENTION = "mention"
    HASHTAG = "hashtag"
    WHITESPACE = "whitespace"


class Script(str, Enum):
    """Writing script for transliteration."""

    ARABIC = "arabic"
    BUCKWALTER = "buckwalter"
    ALA_LC = "ala_lc"  # ALA-LC romanization
    FRANCO = "franco"  # Franco-Arabic (chat alphabet)
    PHONETIC = "phonetic"  # IPA-style phonetic


# ─────────────────────────────────────────────
# RESULT MODELS
# ─────────────────────────────────────────────


class DialectScore(BaseModel):
    """Confidence score for a single dialect."""

    dialect: Dialect
    confidence: float = Field(ge=0.0, le=1.0)
    label: str  # Human-readable label in English and Arabic

    model_config = {"frozen": True}


class DialectResult(BaseModel):
    """Result of dialect detection."""

    text: str
    dialect: Dialect
    confidence: float = Field(ge=0.0, le=1.0)
    all_scores: list[DialectScore] = Field(default_factory=list)
    is_arabic: bool = True
    processing_time_ms: float = Field(ge=0.0)

    model_config = {"frozen": True}

    @property
    def dialect_name_ar(self) -> str:
        """Dialect name in Arabic."""
        _names = {
            Dialect.MSA: "عربي فصيح",
            Dialect.EGYPTIAN: "مصري",
            Dialect.GULF: "خليجي",
            Dialect.LEVANTINE: "شامي",
            Dialect.MAGHREBI: "مغربي",
            Dialect.IRAQI: "عراقي",
            Dialect.YEMENI: "يمني",
            Dialect.SUDANESE: "سوداني",
            Dialect.UNKNOWN: "غير معروف",
        }
        return _names.get(self.dialect, "غير معروف")


class SentimentResult(BaseModel):
    """Result of sentiment analysis."""

    text: str
    label: SentimentLabel
    score: float = Field(ge=0.0, le=1.0, description="Confidence score for predicted label")
    positive_score: float = Field(ge=0.0, le=1.0)
    negative_score: float = Field(ge=0.0, le=1.0)
    neutral_score: float = Field(ge=0.0, le=1.0)
    dialect: Dialect = Dialect.UNKNOWN
    processing_time_ms: float = Field(ge=0.0)

    model_config = {"frozen": True}

    @field_validator("score", "positive_score", "negative_score", "neutral_score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 4)


class Token(BaseModel):
    """A single token from the tokenizer."""

    text: str
    start: int = Field(ge=0, description="Start character offset in source text")
    end: int = Field(ge=0, description="End character offset in source text")
    token_type: TokenType = TokenType.WORD
    is_arabic: bool = True
    normalized: str | None = None
    lemma: str | None = None
    pos_tag: str | None = None  # Part-of-speech tag
    morphology: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}

    def __str__(self) -> str:
        return self.text

    def __len__(self) -> int:
        return len(self.text)


class Entity(BaseModel):
    """A named entity extracted from text."""

    text: str
    label: EntityLabel
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    normalized: str | None = None  # Canonical/normalized form
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}

    def __str__(self) -> str:
        return f"{self.text} ({self.label.value})"


class NERResult(BaseModel):
    """Result of named entity recognition."""

    text: str
    entities: list[Entity] = Field(default_factory=list)
    processing_time_ms: float = Field(ge=0.0)

    model_config = {"frozen": True}

    def by_label(self, label: EntityLabel) -> list[Entity]:
        """Filter entities by label type."""
        return [e for e in self.entities if e.label == label]

    @property
    def persons(self) -> list[Entity]:
        return self.by_label(EntityLabel.PERSON)

    @property
    def locations(self) -> list[Entity]:
        return self.by_label(EntityLabel.LOCATION)

    @property
    def organizations(self) -> list[Entity]:
        return self.by_label(EntityLabel.ORGANIZATION)


class NormalizationResult(BaseModel):
    """Result of text normalization."""

    original: str
    normalized: str
    changes: list[str] = Field(default_factory=list, description="List of normalizations applied")
    processing_time_ms: float = Field(ge=0.0)

    model_config = {"frozen": True}

    @property
    def was_changed(self) -> bool:
        return self.original != self.normalized


class TransliterationResult(BaseModel):
    """Result of transliteration."""

    original: str
    transliterated: str
    source_script: Script
    target_script: Script
    processing_time_ms: float = Field(ge=0.0)

    model_config = {"frozen": True}


class BatchResult(BaseModel):
    """Container for batch processing results."""

    results: list[Any]
    total: int
    successful: int
    failed: int
    total_processing_time_ms: float

    model_config = {"frozen": True}

    @property
    def success_rate(self) -> float:
        return self.successful / self.total if self.total > 0 else 0.0


class ArabicNLPConfig(BaseModel):
    """
    Runtime configuration for :class:`ArabicNLP`.

    Pass to the constructor to customize defaults across the pipeline.
    """

    default_dialect: Dialect = Dialect.MSA
    keyword_top_n: int = Field(default=10, ge=1, le=100)
    keyword_use_stemming: bool = False
    normalize_before_keywords: bool = True
    profile_on_analyze: bool = False
    extract_keywords_on_analyze: bool = False

    model_config = {"frozen": True}


class DocumentAnalysis(BaseModel):
    """
    Structured, serializable output from a full document analysis pipeline.

    Use :meth:`ArabicNLP.analyze_document` to build instances, then export via
    :meth:`to_json` or :meth:`to_dict` for APIs and data pipelines.
    """

    text: str
    dialect: DialectResult
    sentiment: SentimentResult
    entities: NERResult
    normalized: NormalizationResult
    token_count: int = Field(ge=0)
    entity_count: int = Field(ge=0)
    keywords: list[dict[str, Any]] = Field(default_factory=list)
    profile: dict[str, Any] | None = None
    pipeline_time_ms: float = Field(ge=0.0)
    version: str = ""

    model_config = {"frozen": True}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to a JSON string (UTF-8 safe for Arabic)."""
        return self.model_dump_json(indent=indent)

    @property
    def summary(self) -> dict[str, Any]:
        """Compact summary for logging and dashboards."""
        return {
            "dialect": self.dialect.dialect.value,
            "dialect_confidence": self.dialect.confidence,
            "sentiment": self.sentiment.label.value,
            "sentiment_score": self.sentiment.score,
            "entities": self.entity_count,
            "tokens": self.token_count,
            "keywords": [k.get("text") for k in self.keywords[:5]],
            "pipeline_time_ms": self.pipeline_time_ms,
        }


# ─────────────────────────────────────────────
# MORPHOLOGY / POS ENUMS (imported from sub-modules for convenience)
# ─────────────────────────────────────────────

# These are re-exported here so users can do:
#   from arabic_nlp.models import POSTag, WordClass, Gender, etc.
# without needing to know which submodule they live in.


def _import_extended_enums():
    """Lazy import to avoid circular imports at module load time."""
    try:
        from arabic_nlp.morphology import (
            Definiteness as _Definiteness,
        )
        from arabic_nlp.morphology import (
            Gender as _Gender,
        )
        from arabic_nlp.morphology import (
            Number as _Number,
        )
        from arabic_nlp.morphology import (
            VerbTense as _VerbTense,
        )
        from arabic_nlp.morphology import (
            VerbVoice as _VerbVoice,
        )
        from arabic_nlp.morphology import (
            WordClass as _WordClass,
        )
        from arabic_nlp.pos import POSTag as _POSTag

        return _Gender, _Number, _Definiteness, _VerbTense, _VerbVoice, _WordClass, _POSTag
    except ImportError:
        return None, None, None, None, None, None, None


# Expose at module level for type annotation convenience
try:
    from arabic_nlp.morphology import (
        Definiteness,
        Gender,
        Number,
        VerbTense,
        VerbVoice,
        WordClass,
    )
    from arabic_nlp.pos import POSTag
except ImportError:
    # Fallback stubs if morphology not yet initialized
    pass
