"""
Arabic Tokenizer Module
=======================
A rule-based Arabic tokenizer that:
  - Handles Arabic script, Latin, numbers, punctuation, and emojis
  - Preserves character offsets for span extraction
  - Classifies each token by type
  - Optionally performs normalization during tokenization
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator

from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.models import Token, TokenType

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# TOKENIZATION PATTERNS
# Order matters — more specific patterns first
# ─────────────────────────────────────────────

_PATTERNS: list[tuple[TokenType, re.Pattern]] = [
    # URLs (must come before punctuation)
    (
        TokenType.URL,
        re.compile(
            r"https?://[^\s]+"
            r"|www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:/\S*)?"
        ),
    ),
    # Mentions
    (TokenType.MENTION, re.compile(r"@[\w\u0600-\u06FF]+")),
    # Hashtags
    (TokenType.HASHTAG, re.compile(r"#[\w\u0600-\u06FF]+")),
    # Arabic-Indic numbers (٠١٢٣٤٥٦٧٨٩)
    (TokenType.ARABIC_NUMBER, re.compile(r"[٠-٩]+")),
    # Western numbers (with optional decimal/comma separator)
    (TokenType.NUMBER, re.compile(r"\d+(?:[.,]\d+)*")),
    # Arabic words (including hamza variants and special chars)
    (
        TokenType.WORD,
        re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"),
    ),
    # Latin words
    (TokenType.LATIN, re.compile(r"[a-zA-Z]+")),
    # Emoji
    (
        TokenType.EMOJI,
        re.compile(
            "["
            "\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff"
            "\U0001f1e0-\U0001f1ff"
            "\u2640-\u2642"
            "\u2600-\u2b55"
            "\u231a-\u231b"
            "\u23e9-\u23f3"
            "\ufe0f"
            "]+"
        ),
    ),
    # Punctuation (Arabic and Western)
    (TokenType.PUNCTUATION, re.compile(r"[،؛؟!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~،؛؟]")),
    # Whitespace
    (TokenType.WHITESPACE, re.compile(r"\s+")),
]

_ARABIC_WORD_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"
)


class ArabicTokenizer:
    """
    Rule-based Arabic tokenizer.

    Splits Arabic (and mixed-language) text into tokens with character
    offsets, type labels, and optional normalization.

    Example::

        tokenizer = ArabicTokenizer()

        tokens = tokenizer.tokenize("مرحبا بكم في مصر!")
        for tok in tokens:
            print(tok.text, tok.token_type, tok.start, tok.end)
        # مرحبا  WORD  0  5
        # بكم    WORD  7  10
        # في     WORD  11 13
        # مصر    WORD  14 17
        # !      PUNCTUATION  17 18
    """

    def tokenize(
        self,
        text: str,
        *,
        include_punctuation: bool = True,
        include_whitespace: bool = False,
        include_stopwords: bool = True,
        normalize: bool = False,
        stopwords: frozenset[str] | None = None,
    ) -> list[Token]:
        """
        Tokenize text into a list of Token objects.

        Args:
            text: Input text to tokenize.
            include_punctuation: If False, punctuation tokens are dropped.
            include_whitespace: If False (default), whitespace tokens are dropped.
            include_stopwords: If False, stop words are dropped.
            normalize: If True, normalize each token text.
            stopwords: Custom stop-word set. If None and include_stopwords=False,
                       uses the default MSA stop-word set.

        Returns:
            Ordered list of Token objects with character offsets.
        """
        return list(
            self._iter_tokens(
                text,
                include_punctuation=include_punctuation,
                include_whitespace=include_whitespace,
                include_stopwords=include_stopwords,
                normalize=normalize,
                stopwords=stopwords,
            )
        )

    def word_tokenize(self, text: str) -> list[str]:
        """
        Simple word tokenization — returns plain strings.
        Convenience method for when offsets and types aren't needed.
        """
        tokens = self.tokenize(
            text,
            include_punctuation=False,
            include_whitespace=False,
        )
        return [t.text for t in tokens]

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        """Return (start, end) character offsets for each word token."""
        tokens = self.tokenize(text, include_punctuation=False)
        return [(t.start, t.end) for t in tokens]

    def count_tokens(
        self,
        text: str,
        *,
        words_only: bool = True,
    ) -> int:
        """Count tokens in text. If words_only=True, excludes punctuation."""
        tokens = self.tokenize(
            text,
            include_punctuation=not words_only,
        )
        if words_only:
            tokens = [t for t in tokens if t.token_type == TokenType.WORD]
        return len(tokens)

    # ─────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────

    def _iter_tokens(
        self,
        text: str,
        *,
        include_punctuation: bool,
        include_whitespace: bool,
        include_stopwords: bool,
        normalize: bool,
        stopwords: frozenset[str] | None,
    ) -> Iterator[Token]:
        """Generate tokens using a greedy pattern-matching approach."""
        if not text:
            return

        pos = 0
        n = len(text)

        if not include_stopwords and stopwords is None:
            # Lazy import to avoid circular dependency
            from arabic_nlp.stopwords import StopWords

            stopwords = StopWords().get_msa()

        while pos < n:
            matched = False

            for token_type, pattern in _PATTERNS:
                match = pattern.match(text, pos)
                if not match:
                    continue

                matched = True
                token_text = match.group(0)
                start = pos
                end = pos + len(token_text)
                pos = end

                # Apply filters
                if not include_whitespace and token_type == TokenType.WHITESPACE:
                    break
                if not include_punctuation and token_type == TokenType.PUNCTUATION:
                    break
                if (
                    not include_stopwords
                    and token_type == TokenType.WORD
                    and stopwords
                    and token_text in stopwords
                ):
                    break

                # Normalize if requested
                normalized_text: str | None = None
                if normalize and token_type == TokenType.WORD:
                    from arabic_nlp.normalization import ArabicNormalizer

                    norm_result = ArabicNormalizer().minimal(token_text)
                    normalized_text = norm_result.normalized

                is_arabic = (
                    bool(_ARABIC_WORD_RE.match(token_text))
                    if token_type == TokenType.WORD
                    else False
                )

                yield Token(
                    text=token_text,
                    start=start,
                    end=end,
                    token_type=token_type,
                    is_arabic=is_arabic,
                    normalized=normalized_text,
                )
                break

            if not matched:
                # Unknown character — emit single-char token
                yield Token(
                    text=text[pos],
                    start=pos,
                    end=pos + 1,
                    token_type=TokenType.PUNCTUATION,
                    is_arabic=False,
                )
                pos += 1
