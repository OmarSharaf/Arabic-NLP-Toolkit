"""
Arabic Morphology Analysis Module
===================================
Provides deep morphological analysis of Arabic words including:

  - Root extraction (3-letter and 4-letter roots / الجذر الثلاثي والرباعي)
  - Pattern matching (أوزان الصرفية)
  - Prefix/suffix decomposition
  - Gender detection (مذكر / مؤنث)
  - Number detection (مفرد / مثنى / جمع)
  - Definiteness detection (معرفة / نكرة)
  - Verb conjugation analysis (ماضي / مضارع / أمر)
  - Clitic detection (attached prepositions, pronouns, conjunctions)

Arabic morphology is complex — words can carry:
  - Conjunction prefixes:  و ف
  - Preposition prefixes:  ب ك ل
  - Article prefix:        ال
  - Verb prefixes:         ي ت ن أ
  - Pronominal suffixes:   ه ها هم هن ك كم نا
  - Plural suffixes:       ون ين ات
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.models import Dialect

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────


class Gender(str, Enum):
    MASCULINE = "masculine"  # مذكر
    FEMININE = "feminine"  # مؤنث
    UNKNOWN = "unknown"


class Number(str, Enum):
    SINGULAR = "singular"  # مفرد
    DUAL = "dual"  # مثنى
    PLURAL = "plural"  # جمع
    UNKNOWN = "unknown"


class Definiteness(str, Enum):
    DEFINITE = "definite"  # معرفة (+ ال)
    INDEFINITE = "indefinite"  # نكرة
    UNKNOWN = "unknown"


class VerbTense(str, Enum):
    PAST = "past"  # ماضي
    PRESENT = "present"  # مضارع
    COMMAND = "command"  # أمر
    UNKNOWN = "unknown"


class VerbVoice(str, Enum):
    ACTIVE = "active"  # معلوم
    PASSIVE = "passive"  # مجهول
    UNKNOWN = "unknown"


class WordClass(str, Enum):
    VERB = "verb"  # فعل
    NOUN = "noun"  # اسم
    ADJ = "adj"  # صفة
    PREP = "prep"  # حرف جر
    CONJ = "conj"  # حرف عطف
    PRON = "pron"  # ضمير
    ADV = "adv"  # ظرف
    PART = "part"  # أداة
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────
# MORPHOLOGICAL RESULT
# ─────────────────────────────────────────────


@dataclass(frozen=True)
class MorphResult:
    """Full morphological analysis of a single Arabic word."""

    word: str
    stem: str  # Word after prefix/suffix stripping
    root: str | None  # Trilateral/quadrilateral root
    pattern: str | None  # صيغة صرفية (e.g. فَعَّلَ)
    word_class: WordClass
    gender: Gender
    number: Number
    definiteness: Definiteness
    verb_tense: VerbTense
    verb_voice: VerbVoice
    prefixes: tuple[str, ...]  # Attached prefixes (left → right)
    suffixes: tuple[str, ...]  # Attached suffixes (left → right)
    is_verb: bool
    is_noun: bool
    confidence: float
    processing_time_ms: float

    def __str__(self) -> str:
        parts = [f"word={self.word!r}", f"root={self.root!r}", f"class={self.word_class.value}"]
        return f"MorphResult({', '.join(parts)})"


# ─────────────────────────────────────────────
# KNOWN ROOTS (3-letter trilateral roots)
# This is a curated subset — production would use
# a full lexicon of ~10,000 roots
# ─────────────────────────────────────────────

_COMMON_ROOTS_3: frozenset[str] = frozenset(
    [
        # ك-ت-ب (writing)
        "كتب",
        "قرأ",
        "علم",
        "فهم",
        "درس",
        "رسم",
        "خطط",
        # ذ-هـ-ب (going)
        "ذهب",
        "جاء",
        "دخل",
        "خرج",
        "وصل",
        "رجع",
        "مشى",
        # ق-و-ل (saying)
        "قول",
        "كلم",
        "سمع",
        "نظر",
        "رأى",
        "شاف",
        # أ-ك-ل (eating)
        "أكل",
        "شرب",
        "نام",
        "قعد",
        "وقف",
        "ركض",
        # ع-م-ل (working)
        "عمل",
        "بنى",
        "صنع",
        "فعل",
        "حمل",
        "وضع",
        # ح-ب-ب (loving)
        "حبب",
        "كره",
        "خوف",
        "فرح",
        "حزن",
        "غضب",
        # ف-ت-ح (opening)
        "فتح",
        "أغلق",
        "كسر",
        "ربط",
        "قطع",
        "جمع",
        # س-أ-ل (asking)
        "سأل",
        "جاوب",
        "طلب",
        "أعطى",
        "أخذ",
        "باع",
        # ن-ص-ر (helping)
        "نصر",
        "قتل",
        "ضرب",
        "دفع",
        "سرق",
        "قلب",
        # words / knowledge
        "قلم",
        "كتاب",
        "مدرسة",
        "بيت",
        "أرض",
        "سماء",
        # body parts
        "يدي",
        "رجل",
        "عين",
        "أذن",
        "قلب",
        "رأس",
    ]
)

_COMMON_ROOTS_4: frozenset[str] = frozenset(
    [
        "دحرج",
        "زلزل",
        "وسوس",
        "تمتم",
        "بعثر",
        "شعشع",
        "طمطم",
        "دندن",
        "بلبل",
        "زمزم",
    ]
)


# ─────────────────────────────────────────────
# MORPHOLOGICAL PATTERNS (أوزان)
# ─────────────────────────────────────────────

_VERB_PATTERNS: list[tuple[str, str]] = [
    # Pattern regex → weight name
    (r"^فَعَلَ$", "فَعَلَ"),  # Base verb — كَتَبَ
    (r"^فَعَّلَ$", "فَعَّلَ"),  # Intensive — كَسَّرَ
    (r"^فَاعَلَ$", "فَاعَلَ"),  # Reciprocal — قَاتَلَ
    (r"^أَفْعَلَ$", "أَفْعَلَ"),  # Causative — أَخْرَجَ
    (r"^تَفَعَّلَ$", "تَفَعَّلَ"),  # Reflexive intensive — تَكَسَّرَ
    (r"^تَفَاعَلَ$", "تَفَاعَلَ"),  # Reciprocal reflexive — تَقَاتَلَ
    (r"^اِنْفَعَلَ$", "اِنْفَعَلَ"),  # Passive/reflexive — اِنْكَسَرَ
    (r"^اِفْتَعَلَ$", "اِفْتَعَلَ"),  # Reflexive — اِجْتَمَعَ
    (r"^اِسْتَفْعَلَ$", "اِسْتَفْعَلَ"),  # Requestive — اِسْتَخْرَجَ
]

_NOUN_PATTERNS: list[tuple[str, str]] = [
    (r"مَفْعَل", "مَفْعَل"),  # Place/instrument — مَكْتَب
    (r"فَاعِل", "فَاعِل"),  # Active participle — كَاتِب
    (r"مَفْعُول", "مَفْعُول"),  # Passive participle — مَكْتُوب
    (r"فَعِيل", "فَعِيل"),  # Adjective — كَرِيم
    (r"فَعَّال", "فَعَّال"),  # Occupation — نَجَّار
    (r"فَعْلَة", "فَعْلَة"),  # Single action — ضَرْبَة
    (r"فَعَلَان", "فَعَلَان"),  # State — عَطَشَان
    (r"أَفْعَل", "أَفْعَل"),  # Elative — أَكْبَر
    (r"تَفْعِيل", "تَفْعِيل"),  # Masdar — تَعْلِيم
    (r"فِعَال", "فِعَال"),  # Sound plural — كِتَاب
]


# ─────────────────────────────────────────────
# PREFIX / SUFFIX TABLES
# ─────────────────────────────────────────────

# Conjunctions and prepositions that attach as prefixes
_PROCLITIC_PREFIXES: list[str] = [
    "وال",
    "فال",
    "بال",
    "كال",
    "لل",  # Compound: conj/prep + article
    "ال",  # Definite article
    "و",
    "ف",  # Conjunctions
    "ب",
    "ل",
    "ك",  # Prepositions
    "س",
    "لـ",  # Future/lam of purpose
]

# Pronominal suffixes
_ENCLITIC_SUFFIXES: list[str] = [
    "هم",
    "هن",
    "كم",
    "كن",
    "نا",  # Plural pronouns
    "ها",
    "ه",  # 3rd sg pronouns
    "ك",  # 2nd sg pronoun
    "ني",
    "ي",  # 1st sg pronouns
    "تم",
    "تن",
    "تما",  # Verb conjugation
    "وا",
    "ون",
    "ين",  # Plural verb/noun
    "ات",
    "ة",  # Feminine markers
    "ان",
    "ين",  # Dual markers
]

# Feminine markers
_FEMININE_SUFFIXES: frozenset[str] = frozenset(["ة", "ات", "ى", "اء"])

# Plural markers
_PLURAL_SUFFIXES: frozenset[str] = frozenset(["ون", "ين", "ات", "اء", "ان"])
_DUAL_SUFFIXES: frozenset[str] = frozenset(["ان", "ين", "تان", "تين"])

# Verb present-tense prefixes (يفعل/تفعل/نفعل/أفعل)
_VERB_PRESENT_PREFIXES: frozenset[str] = frozenset(["ي", "ت", "ن", "أ"])

# Verb past-tense suffixes
_VERB_PAST_SUFFIXES: frozenset[str] = frozenset(
    [
        "ت",
        "تا",
        "تم",
        "تن",
        "تما",
        "نا",
        "وا",
        "ا",
    ]
)


class ArabicMorphAnalyzer:
    """
    Arabic morphological analyzer.

    Performs deep morphological decomposition of Arabic words
    without requiring external model files.

    Example::

        morph = ArabicMorphAnalyzer()

        result = morph.analyze("الكتاب")
        print(result.stem)         # "كتاب"
        print(result.root)         # "كتب"
        print(result.definiteness) # Definiteness.DEFINITE
        print(result.prefixes)     # ("ال",)

        result = morph.analyze("يكتبون")
        print(result.verb_tense)   # VerbTense.PRESENT
        print(result.number)       # Number.PLURAL
        print(result.gender)       # Gender.MASCULINE

        result = morph.analyze("كاتب")
        print(result.pattern)      # "فَاعِل"
        print(result.word_class)   # WordClass.NOUN
    """

    def analyze(self, word: str) -> MorphResult:
        """
        Perform full morphological analysis on an Arabic word.

        Args:
            word: A single Arabic word (with or without diacritics).

        Returns:
            MorphResult with complete morphological information.

        Raises:
            InvalidInputError: If word is empty or not a string.
        """
        if not isinstance(word, str) or not word.strip():
            raise InvalidInputError("Word must be a non-empty string")

        t0 = time.perf_counter()
        word = word.strip()

        # Strip diacritics for analysis (keep original for display)
        clean = self._strip_diacritics(word)

        # Known roots (e.g. كتب) must not be split by single-letter proclitics (ك)
        if clean in _COMMON_ROOTS_3 or clean in _COMMON_ROOTS_4:
            prefixes, stem_after_prefix = [], clean
        else:
            prefixes, stem_after_prefix = self._strip_prefixes(clean)
        suffixes, stem = self._strip_suffixes(stem_after_prefix)

        root = self._extract_root(stem)
        pattern = self._match_pattern(stem)
        word_class = self._classify_word(stem, prefixes, suffixes, clean)
        gender = self._detect_gender(stem, suffixes)
        number = self._detect_number(stem, suffixes)
        definiteness = self._detect_definiteness(prefixes)
        verb_tense, verb_voice = self._analyze_verb(stem, prefixes, suffixes, word_class)

        elapsed = (time.perf_counter() - t0) * 1000

        return MorphResult(
            word=word,
            stem=stem,
            root=root,
            pattern=pattern,
            word_class=word_class,
            gender=gender,
            number=number,
            definiteness=definiteness,
            verb_tense=verb_tense,
            verb_voice=verb_voice,
            prefixes=tuple(prefixes),
            suffixes=tuple(suffixes),
            is_verb=(word_class == WordClass.VERB),
            is_noun=(word_class in (WordClass.NOUN, WordClass.ADJ)),
            confidence=0.75,  # Rule-based confidence
            processing_time_ms=round(elapsed, 4),
        )

    def analyze_batch(self, words: list[str]) -> list[MorphResult]:
        """Analyze morphology for a list of words."""
        return [self.analyze(w) for w in words]

    def get_root(self, word: str) -> str | None:
        """Quick accessor — return only the root of a word."""
        return self.analyze(word).root

    def get_stem(self, word: str) -> str:
        """Quick accessor — return only the stem."""
        return self.analyze(word).stem

    # ─────────────────────────────────────────────
    # PRIVATE IMPLEMENTATION
    # ─────────────────────────────────────────────

    def _strip_diacritics(self, text: str) -> str:
        return re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)

    def _strip_prefixes(self, word: str) -> tuple[list[str], str]:
        """Strip known proclitic prefixes, longest match first."""
        prefixes: list[str] = []
        remaining = word
        for prefix in sorted(_PROCLITIC_PREFIXES, key=len, reverse=True):
            # Single-char proclitics need a stem of at least 3 letters (avoid ك+تب → تب)
            min_stem = 3 if len(prefix) == 1 else 2
            if remaining.startswith(prefix) and len(remaining) > len(prefix) + min_stem - 1:
                prefixes.append(prefix)
                remaining = remaining[len(prefix) :]
                break  # One pass — don't over-strip
        return prefixes, remaining

    def _strip_suffixes(self, word: str) -> tuple[list[str], str]:
        """Strip known enclitic suffixes, longest match first."""
        suffixes: list[str] = []
        remaining = word
        for suffix in sorted(_ENCLITIC_SUFFIXES, key=len, reverse=True):
            if remaining.endswith(suffix) and len(remaining) > len(suffix) + 1:
                suffixes.insert(0, suffix)
                remaining = remaining[: -len(suffix)]
                break
        return suffixes, remaining

    def _extract_root(self, stem: str) -> str | None:
        """
        Extract the trilateral root using consonant skeleton analysis.
        Arabic roots are typically 3 consonants (C-C-C).

        Strategy:
          1. Check direct lookup in known roots
          2. Remove long vowels (ا و ي) to get consonant skeleton
          3. Try to match a 3-consonant skeleton
        """
        if not stem:
            return None

        # Direct match
        if stem in _COMMON_ROOTS_3:
            return stem
        if stem in _COMMON_ROOTS_4:
            return stem

        # Remove long vowels and get consonant skeleton
        consonant_skeleton = re.sub(r"[اويى]", "", stem)

        # Try 3-letter root
        if len(consonant_skeleton) == 3:
            return consonant_skeleton

        # Try 4-letter root
        if len(consonant_skeleton) == 4:
            return consonant_skeleton

        # If too long, take first 3 consonants as approximation
        if len(consonant_skeleton) >= 3:
            return consonant_skeleton[:3]

        return None

    def _match_pattern(self, stem: str) -> str | None:
        """Match the stem against known Arabic morphological patterns."""
        for pattern_re, pattern_name in _NOUN_PATTERNS + _VERB_PATTERNS:
            if re.search(pattern_re, stem):
                return pattern_name
        return None

    def _classify_word(
        self,
        stem: str,
        prefixes: list[str],
        suffixes: list[str],
        original: str,
    ) -> WordClass:
        """Determine the word class from stem + affixes."""
        # Verb present tense: starts with ي/ت/ن/أ
        if stem and stem[0] in _VERB_PRESENT_PREFIXES and len(stem) >= 3:
            # Heuristic: present-tense verbs have at least 3 chars after prefix removal
            return WordClass.VERB

        # Verb past tense: ends with past suffixes
        for sfx in _VERB_PAST_SUFFIXES:
            if original.endswith(sfx) and len(original) > 3:
                return WordClass.VERB

        # Command verbs often start with اف or specific patterns
        if original.startswith("اف") or original.startswith("اك") or original.startswith("اذ"):
            return WordClass.VERB

        # Noun/adjective by default
        return WordClass.NOUN

    def _detect_gender(self, stem: str, suffixes: list[str]) -> Gender:
        """Detect grammatical gender."""
        suffix_str = "".join(suffixes)
        # Feminine: ends with ة or contains ات
        if stem.endswith("ة") or "ة" in suffix_str or "ات" in suffix_str:
            return Gender.FEMININE
        # Feminine: ends with ى
        if stem.endswith("ى") or stem.endswith("اء"):
            return Gender.FEMININE
        if stem:
            return Gender.MASCULINE
        return Gender.UNKNOWN

    def _detect_number(self, stem: str, suffixes: list[str]) -> Number:
        """Detect grammatical number."""
        suffix_str = "".join(suffixes)
        # Dual
        if suffix_str.endswith("ان") or suffix_str.endswith("ين"):
            if any(d in stem + suffix_str for d in ["تان", "تين"]):
                return Number.DUAL
            # Could be dual or masculine plural — dual if stem + ان
            if stem.endswith("ان") or suffix_str == "ان":
                return Number.DUAL
        # Plural
        if any(p in suffix_str for p in ["ون", "ين", "ات", "اء"]):
            return Number.PLURAL
        return Number.SINGULAR

    def _detect_definiteness(self, prefixes: list[str]) -> Definiteness:
        """Detect definiteness from prefixes."""
        for p in prefixes:
            if p == "ال" or p.endswith("ال"):
                return Definiteness.DEFINITE
        return Definiteness.INDEFINITE

    def _analyze_verb(
        self,
        stem: str,
        prefixes: list[str],
        suffixes: list[str],
        word_class: WordClass,
    ) -> tuple[VerbTense, VerbVoice]:
        """Determine verb tense and voice."""
        if word_class != WordClass.VERB:
            return VerbTense.UNKNOWN, VerbVoice.UNKNOWN

        "".join(suffixes)

        # Present tense: starts with mudarraa prefix
        if stem and stem[0] in _VERB_PRESENT_PREFIXES:
            # Passive present: يُفْعَل pattern (u vowel after prefix)
            voice = VerbVoice.ACTIVE  # Simplified without diacritic analysis
            return VerbTense.PRESENT, voice

        # Command: typically starts with vowel, no subject prefix
        if stem and stem[0] in "اإأ" and not prefixes:
            return VerbTense.COMMAND, VerbVoice.ACTIVE

        # Past tense: default for verbs
        return VerbTense.PAST, VerbVoice.ACTIVE
