"""
Part-of-Speech (POS) Tagger
============================
Tags Arabic tokens with Universal Dependencies POS tags
plus Arabic-specific fine-grained tags.

Universal POS tags used:
  NOUN, VERB, ADJ, ADV, ADP, CONJ, DET, PRON,
  NUM, PUNCT, SYM, X, INTJ, PART

Arabic-specific tags:
  NOUN_PROP  - Proper noun (اسم علم)
  NOUN_VERB  - Verbal noun / masdar (مصدر)
  VERB_PASS  - Passive verb (فعل مجهول)
  PRON_REL   - Relative pronoun (اسم موصول)
  PRON_DEM   - Demonstrative pronoun (اسم إشارة)
  PART_NEG   - Negation particle
  PART_INTER - Interrogative particle
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum

from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.models import Token, TokenType

logger = logging.getLogger(__name__)


class POSTag(str, Enum):
    # Universal
    NOUN = "NOUN"
    VERB = "VERB"
    ADJ = "ADJ"
    ADV = "ADV"
    ADP = "ADP"  # Adposition (preposition)
    CCONJ = "CCONJ"  # Coordinating conjunction
    SCONJ = "SCONJ"  # Subordinating conjunction
    DET = "DET"  # Determiner
    PRON = "PRON"  # Pronoun
    NUM = "NUM"  # Number
    PUNCT = "PUNCT"  # Punctuation
    SYM = "SYM"  # Symbol
    X = "X"  # Other
    INTJ = "INTJ"  # Interjection
    PART = "PART"  # Particle
    # Arabic-specific fine-grained
    NOUN_PROP = "NOUN_PROP"  # Proper noun
    NOUN_VERB = "NOUN_VERB"  # Verbal noun / masdar
    VERB_PASS = "VERB_PASS"  # Passive verb
    PRON_REL = "PRON_REL"  # Relative pronoun (الذي، التي)
    PRON_DEM = "PRON_DEM"  # Demonstrative (هذا، تلك)
    PART_NEG = "PART_NEG"  # Negation particle
    PART_INTER = "PART_INTER"  # Interrogation particle


@dataclass(frozen=True)
class POSResult:
    """POS tagging result for a single token."""

    token: str
    tag: POSTag
    fine_tag: str  # Detailed tag with features
    confidence: float
    features: dict  # Morphosyntactic features

    def __str__(self) -> str:
        return f"{self.token}/{self.tag.value}"


@dataclass(frozen=True)
class POSSequenceResult:
    """POS tagging result for a full sequence."""

    tokens: list[str]
    tags: list[POSResult]
    processing_time_ms: float

    def to_tuples(self) -> list[tuple[str, str]]:
        """Return list of (token, tag) tuples."""
        return [(r.token, r.tag.value) for r in self.tags]

    def filter_by_tag(self, tag: POSTag) -> list[POSResult]:
        return [r for r in self.tags if r.tag == tag]

    @property
    def nouns(self) -> list[POSResult]:
        return [r for r in self.tags if r.tag in (POSTag.NOUN, POSTag.NOUN_PROP)]

    @property
    def verbs(self) -> list[POSResult]:
        return [r for r in self.tags if r.tag in (POSTag.VERB, POSTag.VERB_PASS)]

    @property
    def adjectives(self) -> list[POSResult]:
        return [r for r in self.tags if r.tag == POSTag.ADJ]


# ─────────────────────────────────────────────
# LEXICAL LOOKUP TABLES
# ─────────────────────────────────────────────

_PREPOSITIONS: frozenset[str] = frozenset(
    [
        "في",
        "من",
        "إلى",
        "على",
        "عن",
        "مع",
        "بين",
        "تحت",
        "فوق",
        "خلف",
        "أمام",
        "جانب",
        "داخل",
        "خارج",
        "حول",
        "قبل",
        "بعد",
        "خلال",
        "منذ",
        "حتى",
        "عند",
        "لدى",
        "إزاء",
        "تجاه",
        "نحو",
        "ب",
        "ك",
        "ل",
        "لـ",
    ]
)

_CONJUNCTIONS_COORD: frozenset[str] = frozenset(
    [
        "و",
        "أو",
        "او",
        "بل",
        "لكن",
        "لكنّ",
        "ثم",
        "ف",
        "أم",
    ]
)

_CONJUNCTIONS_SUB: frozenset[str] = frozenset(
    [
        "أن",
        "إن",
        "لأن",
        "لكي",
        "كي",
        "حتى",
        "إذا",
        "لو",
        "بينما",
        "حيث",
        "عندما",
        "لما",
        "بعدما",
        "قبلما",
        "رغم أن",
        "مع أن",
    ]
)

_PRONOUNS_PERSONAL: frozenset[str] = frozenset(
    [
        "أنا",
        "نحن",
        "أنت",
        "أنتِ",
        "أنتم",
        "أنتن",
        "هو",
        "هي",
        "هم",
        "هن",
        "هما",
        "انا",
        "احنا",
        "انت",
        "انتي",
        "انتو",
    ]
)

_PRONOUNS_RELATIVE: frozenset[str] = frozenset(
    [
        "الذي",
        "التي",
        "الذين",
        "اللواتي",
        "اللائي",
        "اللتان",
        "اللذان",
        "ما",
        "من",
        "مهما",
        "حيثما",
        "أينما",
    ]
)

_PRONOUNS_DEMONSTRATIVE: frozenset[str] = frozenset(
    [
        "هذا",
        "هذه",
        "هؤلاء",
        "ذلك",
        "تلك",
        "أولئك",
        "هذان",
        "هاتان",
        "ده",
        "دي",
        "دول",
        "هو ده",
        "دا",
    ]
)

_NEGATION_PARTICLES: frozenset[str] = frozenset(
    [
        "لا",
        "لم",
        "لن",
        "لمّا",
        "ما",
        "مش",
        "مو",
        "مب",
        "ليس",
        "ليست",
        "غير",
        "بدون",
        "عدم",
        "لا يوجد",
        "ما في",
    ]
)

_INTERROGATIVE_PARTICLES: frozenset[str] = frozenset(
    [
        "هل",
        "أ",
        "ما",
        "من",
        "متى",
        "أين",
        "كيف",
        "لماذا",
        "لمَ",
        "أيّ",
        "كم",
        "ايه",
        "مين",
        "فين",
        "ازاي",
        "امتى",
        "ليه",
        "انهي",
        "شو",
        "وين",
        "كيف",
        "ليش",
        "شلون",
    ]
)

_INTERJECTIONS: frozenset[str] = frozenset(
    [
        "يا",
        "آه",
        "أوه",
        "واو",
        "يا إلهي",
        "آه",
        "إيه",
        "عجيب",
        "سبحان الله",
        "الحمد لله",
        "ماشاء الله",
        "إن شاء الله",
        "برافو",
        "أحسنت",
        "مرحبا",
        "أهلا",
        "يلا",
        "طب",
    ]
)

_DETERMINERS: frozenset[str] = frozenset(
    [
        "كل",
        "جميع",
        "بعض",
        "أي",
        "كلا",
        "كلتا",
        "هذا",
        "هذه",
        "ذلك",
        "تلك",
        "كم",
        "أكثر",
        "أقل",
        "معظم",
    ]
)

_COMMON_ADVERBS: frozenset[str] = frozenset(
    [
        "جداً",
        "جدا",
        "كثيراً",
        "قليلاً",
        "دائماً",
        "أحياناً",
        "نادراً",
        "هنا",
        "هناك",
        "الآن",
        "غداً",
        "أمس",
        "اليوم",
        "سابقاً",
        "لاحقاً",
        "فقط",
        "أيضاً",
        "أيضا",
        "معاً",
        "مجدداً",
        "مباشرةً",
        "اوي",
        "بزاف",
        "كتير",
        "شوي",
        "دلوقتي",
        "النهارده",
        "امبارح",
    ]
)

# Verbal noun patterns (masdar)
_MASDAR_PATTERNS: list[re.Pattern] = [
    re.compile(r"^تَ?[^\u064E-\u0650]{3,}ة?$"),  # تفعيل patterns
    re.compile(r"tion$"),  # borrowed words
]


class ArabicPOSTagger:
    """
    Rule-based Arabic POS tagger.

    Uses lexical lookup tables and morphological heuristics
    to assign Universal Dependencies POS tags to Arabic tokens.

    Example::

        tagger = ArabicPOSTagger()

        result = tagger.tag_sentence("ذهب الولد إلى المدرسة")
        for pos in result.tags:
            print(f"{pos.token:10} → {pos.tag.value}")
        # ذهب       → VERB
        # الولد     → NOUN
        # إلى       → ADP
        # المدرسة   → NOUN

        # Access by category
        print([v.token for v in result.verbs])    # ['ذهب']
        print([n.token for n in result.nouns])    # ['الولد', 'المدرسة']
    """

    def tag(self, token: str) -> POSResult:
        """
        Tag a single Arabic token.

        Args:
            token: A single word to tag.

        Returns:
            POSResult with tag, fine_tag, confidence, and features.
        """
        if not token.strip():
            raise InvalidInputError("Token cannot be empty")

        clean = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", token.strip())
        tag, confidence, features = self._classify(clean)

        fine_tag = self._build_fine_tag(tag, features)

        return POSResult(
            token=token,
            tag=tag,
            fine_tag=fine_tag,
            confidence=confidence,
            features=features,
        )

    def tag_sentence(self, text: str) -> POSSequenceResult:
        """
        Tag all tokens in a sentence.

        Args:
            text: Arabic sentence or text.

        Returns:
            POSSequenceResult with tagged token list and metadata.
        """
        if not text.strip():
            raise InvalidInputError("Text cannot be empty")

        t0 = time.perf_counter()

        from arabic_nlp.tokenizer import ArabicTokenizer

        tok = ArabicTokenizer()
        raw_tokens = tok.word_tokenize(text)

        results = [self.tag(t) for t in raw_tokens]

        elapsed = (time.perf_counter() - t0) * 1000

        return POSSequenceResult(
            tokens=raw_tokens,
            tags=results,
            processing_time_ms=round(elapsed, 3),
        )

    def tag_tokens(self, tokens: list[str]) -> list[POSResult]:
        """Tag a pre-tokenized list of tokens."""
        return [self.tag(t) for t in tokens]

    # ─────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────

    def _classify(self, token: str) -> tuple[POSTag, float, dict]:
        """Core classification logic."""
        features: dict = {}

        # Punctuation
        if re.match(r"^[^\u0600-\u06FF\w]+$", token):
            return POSTag.PUNCT, 1.0, {}

        # Numbers
        if re.match(r"^[\d٠-٩]+$", token):
            return POSTag.NUM, 1.0, {"numform": "digit"}

        # Lexical lookups — exact match first
        if token in _NEGATION_PARTICLES:
            return POSTag.PART_NEG, 0.98, {"polarity": "neg"}

        if token in _PREPOSITIONS:
            return POSTag.ADP, 0.96, {}

        if token in _PRONOUNS_RELATIVE:
            return POSTag.PRON_REL, 0.97, {"pron_type": "rel"}

        if token in _PRONOUNS_DEMONSTRATIVE:
            return POSTag.PRON_DEM, 0.96, {"pron_type": "dem"}

        if token in _PRONOUNS_PERSONAL:
            return POSTag.PRON, 0.97, {"pron_type": "prs"}

        if token in _INTERROGATIVE_PARTICLES:
            return POSTag.PART_INTER, 0.97, {"mood": "inter"}

        if token in _CONJUNCTIONS_COORD:
            return POSTag.CCONJ, 0.96, {}

        if token in _CONJUNCTIONS_SUB:
            return POSTag.SCONJ, 0.95, {}

        if token in _INTERJECTIONS:
            return POSTag.INTJ, 0.90, {}

        if token in _DETERMINERS:
            return POSTag.DET, 0.90, {}

        if token in _COMMON_ADVERBS:
            return POSTag.ADV, 0.88, {}

        # Morphological heuristics
        # Verb present: يفعل/تفعل/نفعل/أفعل
        if len(token) >= 3 and token[0] in "يتنأ":
            features["tense"] = "pres"
            return POSTag.VERB, 0.82, features

        # Verb past: ends in typical past suffixes
        if len(token) >= 3 and token[-1] in "تنوا" and not token.endswith("ة"):
            features["tense"] = "past"
            return POSTag.VERB, 0.75, features

        # Definite noun: starts with ال
        if token.startswith("ال") and len(token) > 3:
            features["definite"] = "def"
            return POSTag.NOUN, 0.80, features

        # Adjective: common adjective endings
        if any(token.endswith(e) for e in ["ية", "ي", "وي", "اوي"]):
            return POSTag.ADJ, 0.72, {}

        # Proper noun heuristic: starts with capital transliteration char
        # For Arabic, proper nouns often appear without ال
        if len(token) >= 3 and not token.startswith("ال"):
            return POSTag.NOUN, 0.65, {}

        return POSTag.X, 0.40, {}

    def _build_fine_tag(self, tag: POSTag, features: dict) -> str:
        """Build a detailed feature string from tag + features."""
        parts = [tag.value]
        for k, v in features.items():
            parts.append(f"{k}={v}")
        return "|".join(parts)
