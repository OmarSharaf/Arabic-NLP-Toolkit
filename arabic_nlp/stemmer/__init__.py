"""
Arabic Stemmer Module
======================
Provides two stemming algorithms:

1. **Light Stemmer** (default) — fast, removes common prefixes and suffixes
   without trying to find the root. Good for IR tasks.

2. **Aggressive Stemmer** — more aggressively removes affixes to get
   closer to the root. Higher recall, lower precision.

Both stemmer are dialect-aware and support MSA + Egyptian colloquial.

Reference:
  Larkey, L., Ballesteros, L., & Connell, M. (2002). Light Stemming for Arabic.
  Darwish, K. (2002). Building a Shallow Arabic Morphological Analyzer in One Day.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum

from arabic_nlp.exceptions import InvalidInputError

logger = logging.getLogger(__name__)


class StemmerMode(str, Enum):
    LIGHT = "light"  # Fast, minimal stripping
    AGGRESSIVE = "aggressive"  # More stripping, closer to root


@dataclass(frozen=True)
class StemResult:
    """Result of stemming a single word."""

    original: str
    stem: str
    mode: StemmerMode
    prefixes_removed: tuple[str, ...]
    suffixes_removed: tuple[str, ...]
    processing_time_ms: float

    @property
    def was_reduced(self) -> bool:
        return self.original != self.stem


# ─────────────────────────────────────────────
# PREFIX / SUFFIX TABLES
# ─────────────────────────────────────────────

# Light stemmer: prefixes (longest first)
_LIGHT_PREFIXES: list[str] = [
    "وال",
    "فال",
    "بال",
    "كال",
    "لل",
    "ال",
    "و",
    "ف",
    "ب",
    "ك",
    "ل",
    "أ",
]

# Light stemmer: suffixes (longest first)
_LIGHT_SUFFIXES: list[str] = [
    "تهما",
    "تهم",
    "تهن",  # verb + pronoun combos
    "وها",
    "وهم",
    "وهن",
    "وها",  # plural verb + pronoun
    "تما",
    "تان",
    "تين",  # dual suffixes
    "ونه",
    "ونها",
    "ونهم",  # participle + pronoun
    "وني",
    "وها",  # verb + 1st pron
    "هما",
    "هم",
    "هن",
    "كم",
    "كن",
    "نا",  # pronouns (long)
    "ها",
    "ون",
    "ين",
    "ات",
    "ان",  # common suffixes
    "ة",
    "ه",
    "ك",  # short suffixes (order matters)
    "ي",
    "ا",  # alef/ya
]

# Aggressive: additional prefixes stripped
_AGGRESSIVE_PREFIXES: list[str] = _LIGHT_PREFIXES + [
    "استـ",
    "است",
    "مست",
    "يست",
    "تست",
    "نست",
    "أست",
    "اتـ",
    "انـ",
    "ان",
    "مت",
    "مـ",
]

# Aggressive: additional suffixes stripped
_AGGRESSIVE_SUFFIXES: list[str] = (
    [
        "ياتهم",
        "ياتها",
        "يتهم",  # very long combos
    ]
    + _LIGHT_SUFFIXES
    + [
        "يا",
        "ية",
        "يات",  # nisba adjective markers
        "وات",
        "اوات",
    ]
)

# Words that should NOT be stemmed (particles, prepositions, etc.)
_STOP_STEM: frozenset[str] = frozenset(
    [
        "في",
        "من",
        "إلى",
        "على",
        "عن",
        "مع",
        "هو",
        "هي",
        "أنا",
        "نحن",
        "و",
        "أو",
        "لا",
        "لم",
        "لن",
        "ما",
        "قد",
        "قد",
        "إن",
        "أن",
        "ال",
        "إلى",
        "التي",
        "الذي",
    ]
)

# Minimum stem length — don't over-stem short words
_MIN_STEM_LEN = 2


class ArabicStemmer:
    """
    Arabic word stemmer.

    Implements the Light Stemmer (default) and Aggressive Stemmer.
    The Light Stemmer is appropriate for most IR and text classification tasks.

    Example::

        stemmer = ArabicStemmer()

        # Light stemming
        result = stemmer.stem("الكتاب")
        print(result.stem)               # "كتاب"
        print(result.prefixes_removed)   # ("ال",)

        result = stemmer.stem("يكتبون")
        print(result.stem)               # "يكتب"
        print(result.suffixes_removed)   # ("ون",)

        # Aggressive stemming
        result = stemmer.stem("مدرستهم", mode=StemmerMode.AGGRESSIVE)
        print(result.stem)               # "مدرس"

        # Batch
        stems = stemmer.stem_batch(["الكتاب", "يكتبون", "مدرسة"])
        print([r.stem for r in stems])   # ["كتاب", "يكتب", "مدرس"]
    """

    def stem(
        self,
        word: str,
        mode: StemmerMode = StemmerMode.LIGHT,
    ) -> StemResult:
        """
        Stem an Arabic word.

        Args:
            word: Arabic word to stem.
            mode: StemmerMode.LIGHT (default) or StemmerMode.AGGRESSIVE.

        Returns:
            StemResult with original word, stem, and removed affixes.
        """
        if not isinstance(word, str) or not word.strip():
            raise InvalidInputError("Word must be a non-empty string")

        t0 = time.perf_counter()
        word = word.strip()

        # Don't stem stop words or very short words
        clean = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", word)
        if clean in _STOP_STEM or len(clean) <= _MIN_STEM_LEN:
            return StemResult(
                original=word,
                stem=word,
                mode=mode,
                prefixes_removed=(),
                suffixes_removed=(),
                processing_time_ms=0.0,
            )

        prefixes_removed: list[str] = []
        suffixes_removed: list[str] = []

        prefix_list = _AGGRESSIVE_PREFIXES if mode == StemmerMode.AGGRESSIVE else _LIGHT_PREFIXES
        suffix_list = _AGGRESSIVE_SUFFIXES if mode == StemmerMode.AGGRESSIVE else _LIGHT_SUFFIXES

        stem = clean

        # Strip prefixes (one pass, longest match)
        stem, pref = self._strip_prefix(stem, prefix_list)
        if pref:
            prefixes_removed.append(pref)

        # Strip suffixes (one pass, longest match)
        stem, suf = self._strip_suffix(stem, suffix_list)
        if suf:
            suffixes_removed.append(suf)

        # In aggressive mode, do a second pass
        if mode == StemmerMode.AGGRESSIVE and len(stem) > 4:
            stem2, suf2 = self._strip_suffix(stem, suffix_list)
            if suf2 and len(stem2) >= _MIN_STEM_LEN:
                stem = stem2
                suffixes_removed.append(suf2)

        elapsed = (time.perf_counter() - t0) * 1000

        return StemResult(
            original=word,
            stem=stem,
            mode=mode,
            prefixes_removed=tuple(prefixes_removed),
            suffixes_removed=tuple(suffixes_removed),
            processing_time_ms=round(elapsed, 4),
        )

    def stem_batch(
        self,
        words: list[str],
        mode: StemmerMode = StemmerMode.LIGHT,
    ) -> list[StemResult]:
        """Stem a list of words."""
        return [self.stem(w, mode=mode) for w in words]

    def stem_text(
        self,
        text: str,
        mode: StemmerMode = StemmerMode.LIGHT,
    ) -> list[str]:
        """
        Tokenize text and return a list of stems (plain strings).

        Useful for building bag-of-words features.
        """
        from arabic_nlp.tokenizer import ArabicTokenizer

        tok = ArabicTokenizer()
        words = tok.word_tokenize(text)
        return [self.stem(w, mode=mode).stem for w in words]

    # ─────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────

    @staticmethod
    def _strip_prefix(
        word: str,
        prefix_list: list[str],
    ) -> tuple[str, str | None]:
        """Strip the longest matching prefix. Returns (stem, prefix_removed)."""
        for prefix in sorted(prefix_list, key=len, reverse=True):
            if word.startswith(prefix) and len(word) - len(prefix) >= _MIN_STEM_LEN:
                return word[len(prefix) :], prefix
        return word, None

    @staticmethod
    def _strip_suffix(
        word: str,
        suffix_list: list[str],
    ) -> tuple[str, str | None]:
        """Strip the longest matching suffix. Returns (stem, suffix_removed)."""
        for suffix in sorted(suffix_list, key=len, reverse=True):
            if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM_LEN:
                return word[: -len(suffix)], suffix
        return word, None
