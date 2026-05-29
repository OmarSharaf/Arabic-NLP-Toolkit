"""Tests for ArabicTokenizer."""

import pytest

from arabic_nlp.models import TokenType
from arabic_nlp.tokenizer import ArabicTokenizer


@pytest.fixture
def tokenizer() -> ArabicTokenizer:
    return ArabicTokenizer()


class TestBasicTokenization:
    def test_tokenizes_arabic_words(self, tokenizer):
        tokens = tokenizer.tokenize("مرحبا بكم في مصر")
        words = [t.text for t in tokens if t.token_type == TokenType.WORD]
        assert "مرحبا" in words
        assert "مصر" in words

    def test_token_count(self, tokenizer):
        tokens = tokenizer.tokenize("كلمة وكلمة وكلمة")
        word_tokens = [t for t in tokens if t.token_type == TokenType.WORD]
        assert len(word_tokens) == 3

    def test_empty_string_returns_empty(self, tokenizer):
        tokens = tokenizer.tokenize("")
        assert tokens == []


class TestCharacterOffsets:
    def test_offsets_correct(self, tokenizer):
        text = "مرحبا"
        tokens = tokenizer.tokenize(text)
        for tok in tokens:
            assert text[tok.start : tok.end] == tok.text

    def test_offsets_correct_with_spaces(self, tokenizer):
        text = "كلمة أخرى"
        tokens = tokenizer.tokenize(text)
        for tok in tokens:
            assert text[tok.start : tok.end] == tok.text

    def test_offsets_non_overlapping(self, tokenizer):
        tokens = tokenizer.tokenize("مرحبا بالعالم")
        for i in range(len(tokens) - 1):
            assert tokens[i].end <= tokens[i + 1].start


class TestTokenTypes:
    def test_arabic_word_type(self, tokenizer):
        tokens = tokenizer.tokenize("مرحبا")
        assert tokens[0].token_type == TokenType.WORD

    def test_number_type(self, tokenizer):
        tokens = tokenizer.tokenize("123")
        assert tokens[0].token_type == TokenType.NUMBER

    def test_punctuation_type(self, tokenizer):
        tokens = tokenizer.tokenize("مرحبا!")
        punc = [t for t in tokens if t.token_type == TokenType.PUNCTUATION]
        assert len(punc) >= 1

    def test_latin_type(self, tokenizer):
        tokens = tokenizer.tokenize("hello مرحبا")
        latin = [t for t in tokens if t.token_type == TokenType.LATIN]
        assert len(latin) == 1
        assert latin[0].text == "hello"

    def test_url_type(self, tokenizer):
        tokens = tokenizer.tokenize("زور http://example.com اليوم")
        urls = [t for t in tokens if t.token_type == TokenType.URL]
        assert len(urls) == 1

    def test_mention_type(self, tokenizer):
        tokens = tokenizer.tokenize("شكرا @ahmed على مجهوده")
        mentions = [t for t in tokens if t.token_type == TokenType.MENTION]
        assert len(mentions) == 1

    def test_hashtag_type(self, tokenizer):
        tokens = tokenizer.tokenize("نحن نحب #مصر")
        hashtags = [t for t in tokens if t.token_type == TokenType.HASHTAG]
        assert len(hashtags) == 1


class TestFilterOptions:
    def test_exclude_punctuation(self, tokenizer):
        tokens = tokenizer.tokenize("مرحبا! كيف حالك؟", include_punctuation=False)
        punc = [t for t in tokens if t.token_type == TokenType.PUNCTUATION]
        assert len(punc) == 0

    def test_include_whitespace(self, tokenizer):
        tokens = tokenizer.tokenize("كلمة أخرى", include_whitespace=True)
        ws = [t for t in tokens if t.token_type == TokenType.WHITESPACE]
        assert len(ws) >= 1

    def test_default_excludes_whitespace(self, tokenizer):
        tokens = tokenizer.tokenize("كلمة أخرى")
        ws = [t for t in tokens if t.token_type == TokenType.WHITESPACE]
        assert len(ws) == 0


class TestIsArabicFlag:
    def test_arabic_word_is_arabic_true(self, tokenizer):
        tokens = tokenizer.tokenize("مرحبا")
        assert tokens[0].is_arabic is True

    def test_latin_word_is_arabic_false(self, tokenizer):
        tokens = tokenizer.tokenize("hello")
        assert tokens[0].is_arabic is False


class TestWordTokenize:
    def test_returns_strings(self, tokenizer):
        words = tokenizer.word_tokenize("مرحبا بمصر")
        assert all(isinstance(w, str) for w in words)

    def test_excludes_punctuation(self, tokenizer):
        words = tokenizer.word_tokenize("مرحبا!")
        assert "!" not in words


class TestCountTokens:
    def test_count_correct(self, tokenizer):
        count = tokenizer.count_tokens("هذه ثلاث كلمات")
        assert count == 3

    def test_count_excludes_punctuation_by_default(self, tokenizer):
        count = tokenizer.count_tokens("مرحبا! كيف حالك؟")
        assert count == 3
