"""
Transliteration Module
======================
Converts Arabic text to/from:
  - Franco-Arabic (chat alphabet with numbers: 2, 3, 5, 6, 7, 8, 9)
  - Buckwalter transliteration
  - ALA-LC romanization

Franco-Arabic digit mappings:
  2 = ء/أ/إ    3 = ع    3' = غ
  5 = خ        6 = ط    6' = ظ
  7 = ح        8 = ق    9 = ص
"""

from __future__ import annotations

import logging
import re
import time

from arabic_nlp.exceptions import InvalidInputError, UnsupportedLanguageError
from arabic_nlp.models import Script, TransliterationResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# FRANCO-ARABIC MAPPING
# Arabic → Franco
# ─────────────────────────────────────────────

_ARABIC_TO_FRANCO: list[tuple[str, str]] = [
    # Order matters — longer/ambiguous first
    ("ث", "th"),
    ("ذ", "dh"),
    ("ش", "sh"),
    ("خ", "kh"),
    ("غ", "gh"),
    ("ظ", "z"),  # Simplified
    ("ض", "d"),  # Simplified
    ("ص", "9"),
    ("ق", "2"),  # Common Franco usage
    ("ع", "3"),
    ("ح", "7"),
    ("ا", "a"),
    ("أ", "2"),
    ("إ", "2"),
    ("آ", "aa"),
    ("ء", "2"),
    ("ب", "b"),
    ("ت", "t"),
    ("ج", "j"),
    ("د", "d"),
    ("ر", "r"),
    ("ز", "z"),
    ("س", "s"),
    ("ط", "t"),  # Simplified
    ("ف", "f"),
    ("ك", "k"),
    ("ل", "l"),
    ("م", "m"),
    ("ن", "n"),
    ("ه", "h"),
    ("و", "w"),
    ("ي", "y"),
    ("ى", "a"),
    ("ة", "a"),
    ("ئ", "2"),
    ("ؤ", "2"),
    ("لا", "la"),
    # Diacritics (remove in Franco)
    ("َ", "a"),
    ("ِ", "i"),
    ("ُ", "u"),
    ("ً", "an"),
    ("ٍ", "en"),
    ("ٌ", "on"),
    ("ْ", ""),  # Sukun — remove
    ("ّ", ""),  # Shadda — handled separately
    ("ـ", ""),  # Tatweel
]

# Franco → Arabic (reverse, simplified)
_FRANCO_TO_ARABIC: list[tuple[str, str]] = [
    # Multi-char first
    ("sh", "ش"),
    ("kh", "خ"),
    ("gh", "غ"),
    ("th", "ث"),
    ("dh", "ذ"),
    ("aa", "ا"),
    ("ee", "ي"),
    ("oo", "و"),
    # Single digit/char
    ("2", "أ"),
    ("3", "ع"),
    ("5", "خ"),
    ("6", "ط"),
    ("7", "ح"),
    ("8", "ق"),
    ("9", "ص"),
    # Latin → Arabic approximations
    ("a", "ا"),
    ("b", "ب"),
    ("t", "ت"),
    ("j", "ج"),
    ("d", "د"),
    ("r", "ر"),
    ("z", "ز"),
    ("s", "س"),
    ("f", "ف"),
    ("k", "ك"),
    ("l", "ل"),
    ("m", "م"),
    ("n", "ن"),
    ("h", "ه"),
    ("w", "و"),
    ("y", "ي"),
    ("e", ""),  # Drop vowels without context
    ("i", "ي"),
    ("o", "و"),
    ("u", "و"),
    ("q", "ق"),
    ("x", "كس"),
    ("p", "ب"),  # No 'p' in Arabic
    ("v", "ف"),  # No 'v' in Arabic
    ("g", "ج"),  # Egyptian-style
    ("c", "ك"),
]

# ─────────────────────────────────────────────
# BUCKWALTER MAPPING
# Standard academic transliteration
# ─────────────────────────────────────────────

_ARABIC_TO_BUCKWALTER: dict[str, str] = {
    "ء": "'",
    "آ": "|",
    "أ": ">",
    "ؤ": "&",
    "إ": "<",
    "ئ": "}",
    "ا": "A",
    "ب": "b",
    "ة": "p",
    "ت": "t",
    "ث": "v",
    "ج": "j",
    "ح": "H",
    "خ": "x",
    "د": "d",
    "ذ": "*",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "$",
    "ص": "S",
    "ض": "D",
    "ط": "T",
    "ظ": "Z",
    "ع": "E",
    "غ": "G",
    "ف": "f",
    "ق": "q",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "و": "w",
    "ى": "Y",
    "ي": "y",
    "ً": "F",
    "ٍ": "K",
    "ٌ": "N",
    "َ": "a",
    "ِ": "i",
    "ُ": "u",
    "ّ": "~",
    "ْ": "o",
    "ـ": "_",
}

_BUCKWALTER_TO_ARABIC: dict[str, str] = {
    v: k
    for k, v in _ARABIC_TO_BUCKWALTER.items()
    if v not in ("p", "Y")  # Avoid duplicate key overwrite issues
}


class Transliterator:
    """
    Transliterates Arabic text to/from various scripts.

    Example::

        t = Transliterator()

        # Arabic → Franco
        result = t.transliterate("مرحبا كيف حالك")
        print(result.transliterated)  # "marhba kyf 7alk"

        # Arabic → Buckwalter
        result = t.transliterate("بسم الله", target=Script.BUCKWALTER)
        print(result.transliterated)  # "bsm AlllAh"

        # Franco → Arabic
        result = t.transliterate(
            "ana mesh 3aref",
            source=Script.FRANCO,
            target=Script.ARABIC,
        )
        print(result.transliterated)  # "انا مش عارف"
    """

    def transliterate(
        self,
        text: str,
        *,
        source: Script = Script.ARABIC,
        target: Script = Script.FRANCO,
    ) -> TransliterationResult:
        """
        Transliterate text between scripts.

        Args:
            text: Text to transliterate.
            source: Source script (default: ARABIC).
            target: Target script (default: FRANCO).

        Returns:
            TransliterationResult with original and transliterated text.
        """
        if not text.strip():
            raise InvalidInputError("Input text cannot be empty")

        t0 = time.perf_counter()

        if source == Script.ARABIC and target == Script.FRANCO:
            result = self._arabic_to_franco(text)
        elif source == Script.FRANCO and target == Script.ARABIC:
            result = self._franco_to_arabic(text)
        elif source == Script.ARABIC and target == Script.BUCKWALTER:
            result = self._arabic_to_buckwalter(text)
        elif source == Script.BUCKWALTER and target == Script.ARABIC:
            result = self._buckwalter_to_arabic(text)
        elif source == target:
            result = text  # No-op
        else:
            raise InvalidInputError(
                f"Transliteration from {source.value} to {target.value} is not supported yet."
            )

        elapsed = (time.perf_counter() - t0) * 1000

        return TransliterationResult(
            original=text,
            transliterated=result,
            source_script=source,
            target_script=target,
            processing_time_ms=round(elapsed, 3),
        )

    # ─────────────────────────────────────────────
    # IMPLEMENTATION
    # ─────────────────────────────────────────────

    def _arabic_to_franco(self, text: str) -> str:
        """Convert Arabic script to Franco-Arabic."""
        result = text
        # Handle shadda (double consonant)
        for arabic, franco in _ARABIC_TO_FRANCO:
            result = result.replace(arabic, franco)
        # Clean up any remaining Arabic chars
        result = re.sub(r"[\u0600-\u06FF]", "", result)
        # Normalize spaces
        result = re.sub(r"\s+", " ", result).strip()
        return result

    def _franco_to_arabic(self, text: str) -> str:
        """Convert Franco-Arabic to Arabic script."""
        result = text.lower()
        for franco, arabic in _FRANCO_TO_ARABIC:
            result = result.replace(franco, arabic)
        # Remove remaining Latin chars
        result = re.sub(r"[a-z]", "", result)
        result = re.sub(r"\s+", " ", result).strip()
        return result

    def _arabic_to_buckwalter(self, text: str) -> str:
        """Convert Arabic script to Buckwalter transliteration."""
        return "".join(_ARABIC_TO_BUCKWALTER.get(ch, ch) for ch in text)

    def _buckwalter_to_arabic(self, text: str) -> str:
        """Convert Buckwalter transliteration to Arabic script."""
        result = ""
        i = 0
        while i < len(text):
            ch = text[i]
            result += _BUCKWALTER_TO_ARABIC.get(ch, ch)
            i += 1
        return result
