"""
Arabic NLP Utilities Module
============================
Collection of utility functions for Arabic text analysis:

  - Text statistics (word count, unique words, TTR, etc.)
  - Language detection (Arabic vs. non-Arabic)
  - Readability scoring
  - Text comparison (similarity)
  - Encoding utilities
  - Arabic-specific string helpers
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections import Counter
from collections import Counter as CounterType
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# ARABIC UNICODE RANGES
# ─────────────────────────────────────────────

ARABIC_BLOCK = re.compile(r"[\u0600-\u06FF]")
ARABIC_SUPPLEMENT = re.compile(r"[\u0750-\u077F]")
ARABIC_EXTENDED_A = re.compile(r"[\u08A0-\u08FF]")
ARABIC_PRESENTATION_A = re.compile(r"[\uFB50-\uFDFF]")
ARABIC_PRESENTATION_B = re.compile(r"[\uFE70-\uFEFF]")
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
ARABIC_WORD_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]+")
LATIN_RE = re.compile(r"[a-zA-Z]")
DIGIT_RE = re.compile(r"[0-9٠-٩]")


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────


@dataclass
class TextStatistics:
    """Comprehensive statistics for an Arabic text."""

    total_chars: int
    total_chars_no_space: int
    total_words: int
    unique_words: int
    arabic_words: int
    latin_words: int
    digit_tokens: int
    sentences: int
    paragraphs: int
    avg_word_length: float
    avg_sentence_length: float  # words per sentence
    type_token_ratio: float  # unique / total words (lexical diversity)
    arabic_ratio: float  # Arabic chars / total non-space chars
    has_diacritics: bool
    most_common_words: list[tuple[str, int]]
    longest_word: str

    def __str__(self) -> str:
        return (
            f"TextStatistics("
            f"words={self.total_words}, "
            f"unique={self.unique_words}, "
            f"TTR={self.type_token_ratio:.2f}, "
            f"arabic_ratio={self.arabic_ratio:.1%})"
        )


@dataclass
class ReadabilityScore:
    """
    Arabic text readability scores.

    Note: Classical readability formulas (Flesch, Gunning Fog) were designed
    for English. These are Arabic-adapted approximations.
    """

    word_count: int
    sentence_count: int
    syllable_estimate: int
    avg_words_per_sent: float
    avg_syllables_per_word: float
    difficulty_level: str  # "easy", "moderate", "difficult", "very_difficult"
    difficulty_score: float  # 0 (easy) to 1 (very difficult)
    recommendation: str


@dataclass
class SimilarityResult:
    """Text similarity result."""

    text1: str
    text2: str
    jaccard: float  # Jaccard similarity on word sets
    cosine: float  # Cosine similarity on word frequency vectors
    overlap: float  # Word overlap coefficient


# ─────────────────────────────────────────────
# MAIN UTILITY FUNCTIONS
# ─────────────────────────────────────────────


def get_statistics(text: str, top_n: int = 10) -> TextStatistics:
    """
    Compute comprehensive statistics for an Arabic text.

    Args:
        text: Input Arabic text.
        top_n: Number of most common words to include.

    Returns:
        TextStatistics dataclass with all metrics.

    Example::

        from arabic_nlp.utils import get_statistics

        stats = get_statistics("مرحبا بالعالم. كيف حالكم جميعاً؟")
        print(stats.total_words)      # 5
        print(stats.type_token_ratio) # 1.0 (all unique)
        print(stats.arabic_ratio)     # 1.0
    """
    if not text:
        return TextStatistics(
            total_chars=0,
            total_chars_no_space=0,
            total_words=0,
            unique_words=0,
            arabic_words=0,
            latin_words=0,
            digit_tokens=0,
            sentences=0,
            paragraphs=0,
            avg_word_length=0.0,
            avg_sentence_length=0.0,
            type_token_ratio=0.0,
            arabic_ratio=0.0,
            has_diacritics=False,
            most_common_words=[],
            longest_word="",
        )

    # Character counts
    total_chars = len(text)
    total_chars_no_space = len(text.replace(" ", "").replace("\n", ""))

    # Words
    words_all = re.findall(r"\S+", text)
    arabic_words_list = ARABIC_WORD_RE.findall(text)
    latin_words_list = re.findall(r"[a-zA-Z]+", text)
    digit_tokens = len(re.findall(r"[0-9٠-٩]+", text))

    total_words = len(words_all)
    arabic_words = len(arabic_words_list)
    latin_words = len(latin_words_list)
    unique_words = len({w.lower() for w in arabic_words_list})

    # Sentences (split on . ! ? ؟ ؛)
    sentences_list = re.split(r"[.!?؟؛\n]+", text.strip())
    sentences_list = [s.strip() for s in sentences_list if s.strip()]
    sentences = max(len(sentences_list), 1)

    # Paragraphs
    paragraphs = max(len([p for p in text.split("\n\n") if p.strip()]), 1)

    # Averages
    avg_word_length = (
        sum(len(w) for w in arabic_words_list) / arabic_words if arabic_words > 0 else 0.0
    )
    avg_sentence_length = total_words / sentences

    # Type-Token Ratio (lexical diversity)
    ttr = unique_words / arabic_words if arabic_words > 0 else 0.0

    # Arabic ratio
    arabic_chars = len(ARABIC_BLOCK.findall(text))
    non_space = len(re.sub(r"\s", "", text))
    arabic_ratio = arabic_chars / non_space if non_space > 0 else 0.0

    # Diacritics
    has_diacritics = bool(ARABIC_DIACRITICS.search(text))

    # Most common Arabic words
    word_counts: CounterType[str] = Counter(arabic_words_list)
    most_common = word_counts.most_common(top_n)

    # Longest word
    longest_word = max(arabic_words_list, key=len) if arabic_words_list else ""

    return TextStatistics(
        total_chars=total_chars,
        total_chars_no_space=total_chars_no_space,
        total_words=total_words,
        unique_words=unique_words,
        arabic_words=arabic_words,
        latin_words=latin_words,
        digit_tokens=digit_tokens,
        sentences=sentences,
        paragraphs=paragraphs,
        avg_word_length=round(avg_word_length, 2),
        avg_sentence_length=round(avg_sentence_length, 2),
        type_token_ratio=round(ttr, 4),
        arabic_ratio=round(arabic_ratio, 4),
        has_diacritics=has_diacritics,
        most_common_words=most_common,
        longest_word=longest_word,
    )


def detect_language(text: str) -> dict[str, float]:
    """
    Detect the language composition of a text.

    Returns a dict with ratios for each detected script.

    Example::

        ratios = detect_language("Hello مرحبا 123")
        # {"arabic": 0.45, "latin": 0.36, "digit": 0.19, "other": 0.0}
    """
    if not text.strip():
        return {}

    non_space = re.sub(r"\s", "", text)
    n = len(non_space)
    if n == 0:
        return {}

    arabic = len(ARABIC_BLOCK.findall(non_space))
    latin = len(LATIN_RE.findall(non_space))
    digits = len(DIGIT_RE.findall(non_space))
    other = n - arabic - latin - digits

    return {
        "arabic": round(arabic / n, 4),
        "latin": round(latin / n, 4),
        "digit": round(digits / n, 4),
        "other": round(max(other, 0) / n, 4),
    }


def is_arabic(text: str, threshold: float = 0.5) -> bool:
    """
    Return True if the text is predominantly Arabic.

    Args:
        text: Input text.
        threshold: Minimum Arabic character ratio to qualify (default 0.5).
    """
    ratios = detect_language(text)
    return ratios.get("arabic", 0.0) >= threshold


def get_readability(text: str) -> ReadabilityScore:
    """
    Estimate the readability of an Arabic text.

    Arabic readability uses syllable-count approximation
    based on vowel (long + short) counts.

    Example::

        score = get_readability("النص هذا سهل القراءة.")
        print(score.difficulty_level)  # "easy"
        print(score.avg_words_per_sent) # 5.0
    """
    stats = get_statistics(text)

    # Estimate syllables: each long vowel (ا و ي) counts as a syllable
    # Short vowels (diacritics) also count if present
    long_vowels = len(re.findall(r"[اويى]", text))
    short_vowels = len(ARABIC_DIACRITICS.findall(text))
    syllable_estimate = long_vowels + short_vowels

    avg_syllables = syllable_estimate / stats.arabic_words if stats.arabic_words > 0 else 0

    # Difficulty scoring (0=easy, 1=very difficult)
    # Factors: avg sentence length, avg word length, syllable density
    sent_factor = min(stats.avg_sentence_length / 20.0, 1.0)
    word_factor = min(stats.avg_word_length / 10.0, 1.0)
    syl_factor = min(avg_syllables / 5.0, 1.0)
    difficulty_score = round((sent_factor * 0.4 + word_factor * 0.3 + syl_factor * 0.3), 4)

    if difficulty_score < 0.25:
        level = "easy"
        rec = "مناسب للقراء من جميع المستويات"
    elif difficulty_score < 0.50:
        level = "moderate"
        rec = "مناسب للقراء المتوسطين"
    elif difficulty_score < 0.75:
        level = "difficult"
        rec = "مناسب للقراء المتقدمين"
    else:
        level = "very_difficult"
        rec = "مناسب للمتخصصين والأكاديميين"

    return ReadabilityScore(
        word_count=stats.total_words,
        sentence_count=stats.sentences,
        syllable_estimate=syllable_estimate,
        avg_words_per_sent=stats.avg_sentence_length,
        avg_syllables_per_word=round(avg_syllables, 2),
        difficulty_level=level,
        difficulty_score=difficulty_score,
        recommendation=rec,
    )


def compute_similarity(text1: str, text2: str) -> SimilarityResult:
    """
    Compute similarity between two Arabic texts using multiple metrics.

    Args:
        text1: First text.
        text2: Second text.

    Returns:
        SimilarityResult with Jaccard, cosine, and overlap scores.

    Example::

        sim = compute_similarity(
            "المنتج رائع جداً",
            "المنتج ممتاز جداً"
        )
        print(sim.jaccard)  # ~0.50 (shared: المنتج, جداً)
        print(sim.cosine)   # ~0.67
    """
    words1 = set(ARABIC_WORD_RE.findall(text1.lower()))
    words2 = set(ARABIC_WORD_RE.findall(text2.lower()))

    # Jaccard: |intersection| / |union|
    intersection = words1 & words2
    union = words1 | words2
    jaccard = len(intersection) / len(union) if union else 0.0

    # Overlap coefficient: |intersection| / min(|A|, |B|)
    overlap = (
        len(intersection) / min(len(words1), len(words2))
        if min(len(words1), len(words2)) > 0
        else 0.0
    )

    # Cosine similarity on frequency vectors
    counter1: CounterType[str] = Counter(ARABIC_WORD_RE.findall(text1))
    counter2: CounterType[str] = Counter(ARABIC_WORD_RE.findall(text2))
    all_words = set(counter1) | set(counter2)

    dot = sum(counter1.get(w, 0) * counter2.get(w, 0) for w in all_words)
    mag1 = math.sqrt(sum(v**2 for v in counter1.values()))
    mag2 = math.sqrt(sum(v**2 for v in counter2.values()))
    cosine = dot / (mag1 * mag2) if mag1 * mag2 > 0 else 0.0

    return SimilarityResult(
        text1=text1,
        text2=text2,
        jaccard=round(jaccard, 4),
        cosine=round(cosine, 4),
        overlap=round(overlap, 4),
    )


def remove_diacritics(text: str) -> str:
    """Remove all Arabic diacritics (tashkeel) from text."""
    return ARABIC_DIACRITICS.sub("", text)


def has_diacritics(text: str) -> bool:
    """Return True if the text contains Arabic diacritics."""
    return bool(ARABIC_DIACRITICS.search(text))


def count_arabic_words(text: str) -> int:
    """Count the number of Arabic word tokens in text."""
    return len(ARABIC_WORD_RE.findall(text))


def extract_arabic_only(text: str) -> str:
    """Extract only the Arabic words from mixed text, joined by spaces."""
    words = ARABIC_WORD_RE.findall(text)
    return " ".join(words)


def extract_numbers(text: str) -> list[str]:
    """Extract all number strings (Arabic-Indic and Western) from text."""
    return re.findall(r"[0-9٠-٩]+(?:[.,][0-9٠-٩]+)*", text)


def arabic_to_western_digits(text: str) -> str:
    """Convert Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩) to Western digits (0-9)."""
    mapping = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return text.translate(mapping)


def western_to_arabic_digits(text: str) -> str:
    """Convert Western digits (0-9) to Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩)."""
    mapping = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
    return text.translate(mapping)


def normalize_unicode(text: str) -> str:
    """Apply NFC unicode normalization (composing form)."""
    return unicodedata.normalize("NFC", text)


def truncate_arabic(text: str, max_words: int, ellipsis: str = "...") -> str:
    """
    Truncate Arabic text to a maximum number of words.

    Respects word boundaries (does not cut mid-word).

    Example::

        truncate_arabic("مرحبا بكم في مصر الحبيبة", 3)
        # "مرحبا بكم في..."
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + ellipsis


def reverse_arabic_text(text: str) -> str:
    """
    Reverse the word order of an Arabic text (RTL utility).
    Words themselves are not reversed, only their order.
    """
    return " ".join(reversed(text.split()))


def word_frequency(text: str, top_n: int = 20) -> list[tuple[str, int]]:
    """
    Return the most frequent Arabic words in a text.

    Args:
        text: Arabic text.
        top_n: Number of top words to return.

    Returns:
        List of (word, count) tuples sorted by frequency.
    """
    words = ARABIC_WORD_RE.findall(text)
    return Counter(words).most_common(top_n)


def sentence_split(text: str) -> list[str]:
    """
    Split Arabic text into sentences.

    Handles Arabic punctuation (؟ ، ؛) as well as standard delimiters.
    """
    sentences = re.split(r"[.!?؟؛\n]+", text)
    return [s.strip() for s in sentences if s.strip()]
