"""
Arabic Text Normalization Module
=================================
Provides comprehensive normalization of Arabic text including:
  - Diacritics (tashkeel/harakat) removal
  - Alef normalization (أ إ آ ٱ → ا)
  - Hamza normalization
  - Teh marbuta normalization (ة → ه)
  - Tatweel/kashida removal
  - Punctuation normalization
  - Whitespace normalization
  - Social media noise removal (URLs, mentions, hashtags, emojis)
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata

from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.models import NormalizationResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# UNICODE RANGES & PATTERNS
# ─────────────────────────────────────────────

# Arabic diacritics (tashkeel + tatweel)
ARABIC_DIACRITICS = re.compile(
    r"[\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)

# Tatweel (kashida) character
TATWEEL = re.compile(r"\u0640")

# Alef variants → ا
ALEF_VARIANTS = re.compile(r"[أإآٱ]")

# Hamza variants
HAMZA_VARIANTS = re.compile(r"[ؤئ]")

# Alef maqsura ى → ي
ALEF_MAQSURA = re.compile(r"ى(?!\u064A)")  # Not followed by hamza

# Teh marbuta ة → ه
TEH_MARBUTA = re.compile(r"ة")

# Waw with hamza above/below
WAW_HAMZA = re.compile(r"ؤ")

# Ya with hamza below
YA_HAMZA = re.compile(r"ئ")

# Extra whitespace
EXTRA_WHITESPACE = re.compile(r"\s+")

# URLs
URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|"
    r"www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:/\S*)?"
)

# Mentions and hashtags
MENTION_PATTERN = re.compile(r"@[\w\u0600-\u06FF]+")
HASHTAG_PATTERN = re.compile(r"#[\w\u0600-\u06FF]+")

# Emoji pattern (covers most emoji ranges)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # Emoticons
    "\U0001f300-\U0001f5ff"  # Misc Symbols and Pictographs
    "\U0001f680-\U0001f6ff"  # Transport and Map
    "\U0001f1e0-\U0001f1ff"  # Flags
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2b55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+",
    flags=re.UNICODE,
)

# Repeated characters (more than 2 of same Arabic char in a row → 1)
REPEATED_CHARS = re.compile(r"([\u0600-\u06FF])\1{2,}")

# Arabic punctuation variants → standard
ARABIC_COMMA = re.compile(r"،")
ARABIC_SEMICOLON = re.compile(r"؛")
ARABIC_QUESTION = re.compile(r"؟")


class ArabicNormalizer:
    """
    Comprehensive Arabic text normalizer.

    Applies configurable normalization steps to Arabic text, tracking
    which transformations were applied.

    Example::

        normalizer = ArabicNormalizer()

        result = normalizer.normalize("هَذَا نَصٌّ بِالتَّشْكِيل مَعَ تطويـــل")
        print(result.normalized)  # "هذا نص بالتشكيل مع تطويل"
        print(result.changes)     # ["removed_diacritics", "removed_tatweel"]

        # Social media text
        result = normalizer.normalize(
            "شوف الفيديو ده 🔥 @ahmed #مصر http://example.com",
            remove_emojis=True,
            remove_mentions=True,
            remove_hashtags=True,
            remove_urls=True,
        )
        print(result.normalized)  # "شوف الفيديو ده"
    """

    def normalize(
        self,
        text: str,
        *,
        remove_diacritics: bool = True,
        normalize_alef: bool = True,
        normalize_teh_marbuta: bool = True,
        normalize_hamza: bool = True,
        normalize_alef_maqsura: bool = False,
        remove_tatweel: bool = True,
        remove_extra_spaces: bool = True,
        reduce_repeated_chars: bool = True,
        remove_urls: bool = False,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        remove_emojis: bool = False,
        strip: bool = True,
    ) -> NormalizationResult:
        """
        Normalize Arabic text.

        All flags default to safe values that work well for most NLP tasks.
        Social media cleanup (urls, mentions, hashtags, emojis) is off by
        default to preserve information.

        Returns:
            NormalizationResult with original, normalized text, and
            list of normalization operations that were applied.
        """
        if not isinstance(text, str):
            raise InvalidInputError(f"Expected str, got {type(text).__name__}")

        t0 = time.perf_counter()
        result = text
        changes: list[str] = []

        # --- Social media cleanup (applied first to avoid normalizing noise) ---
        if remove_urls:
            new = URL_PATTERN.sub(" ", result)
            if new != result:
                changes.append("removed_urls")
            result = new

        if remove_mentions:
            new = MENTION_PATTERN.sub(" ", result)
            if new != result:
                changes.append("removed_mentions")
            result = new

        if remove_hashtags:
            new = HASHTAG_PATTERN.sub(" ", result)
            if new != result:
                changes.append("removed_hashtags")
            result = new

        if remove_emojis:
            new = EMOJI_PATTERN.sub(" ", result)
            if new != result:
                changes.append("removed_emojis")
            result = new

        # --- Arabic-specific normalization ---
        if remove_diacritics:
            new = ARABIC_DIACRITICS.sub("", result)
            if new != result:
                changes.append("removed_diacritics")
            result = new

        if remove_tatweel:
            new = TATWEEL.sub("", result)
            if new != result:
                changes.append("removed_tatweel")
            result = new

        if normalize_alef:
            new = ALEF_VARIANTS.sub("ا", result)
            if new != result:
                changes.append("normalized_alef")
            result = new

        if normalize_hamza:
            new = WAW_HAMZA.sub("و", result)
            new = YA_HAMZA.sub("ي", new)
            if new != result:
                changes.append("normalized_hamza")
            result = new

        if normalize_teh_marbuta:
            new = TEH_MARBUTA.sub("ه", result)
            if new != result:
                changes.append("normalized_teh_marbuta")
            result = new

        if normalize_alef_maqsura:
            new = ALEF_MAQSURA.sub("ي", result)
            if new != result:
                changes.append("normalized_alef_maqsura")
            result = new

        if reduce_repeated_chars:
            new = REPEATED_CHARS.sub(r"\1", result)
            if new != result:
                changes.append("reduced_repeated_chars")
            result = new

        # --- Whitespace cleanup (always last) ---
        if remove_extra_spaces:
            new = EXTRA_WHITESPACE.sub(" ", result)
            if new != result:
                changes.append("normalized_whitespace")
            result = new

        if strip:
            result = result.strip()

        elapsed = (time.perf_counter() - t0) * 1000

        return NormalizationResult(
            original=text,
            normalized=result,
            changes=changes,
            processing_time_ms=round(elapsed, 3),
        )

    def clean_social(self, text: str) -> NormalizationResult:
        """
        Shortcut: apply all social media cleanup + standard normalization.
        Ideal for tweets, comments, and chat messages.
        """
        return self.normalize(
            text,
            remove_urls=True,
            remove_mentions=True,
            remove_hashtags=True,
            remove_emojis=True,
        )

    def minimal(self, text: str) -> NormalizationResult:
        """
        Minimal normalization: only remove diacritics and extra spaces.
        Use when preserving original text structure is important.
        """
        return self.normalize(
            text,
            normalize_alef=False,
            normalize_teh_marbuta=False,
            normalize_hamza=False,
        )

    @staticmethod
    def contains_arabic(text: str) -> bool:
        """Check whether text contains Arabic characters."""
        return bool(re.search(r"[\u0600-\u06FF]", text))

    @staticmethod
    def arabic_char_ratio(text: str) -> float:
        """Return ratio of Arabic chars to total non-whitespace chars."""
        if not text.strip():
            return 0.0
        arabic = len(re.findall(r"[\u0600-\u06FF]", text))
        total = len(re.findall(r"\S", text))
        return arabic / total if total > 0 else 0.0
