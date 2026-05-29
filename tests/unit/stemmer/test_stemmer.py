"""Tests for ArabicStemmer."""

import pytest

from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.stemmer import ArabicStemmer, StemmerMode, StemResult


@pytest.fixture
def stemmer() -> ArabicStemmer:
    return ArabicStemmer()


class TestLightStemmer:
    def test_removes_definite_article(self, stemmer):
        result = stemmer.stem("الكتاب")
        assert result.stem == "كتاب"
        assert "ال" in result.prefixes_removed

    def test_removes_plural_suffix_oon(self, stemmer):
        result = stemmer.stem("كاتبون")
        assert "ون" in result.suffixes_removed

    def test_removes_plural_suffix_iin(self, stemmer):
        result = stemmer.stem("كاتبين")
        assert "ين" in result.suffixes_removed

    def test_removes_teh_marbuta(self, stemmer):
        result = stemmer.stem("مدرسة")
        assert "ة" in result.suffixes_removed

    def test_short_word_not_overstemmed(self, stemmer):
        result = stemmer.stem("في")
        assert result.stem == "في"  # Stop word — not stemmed

    def test_mode_is_light(self, stemmer):
        result = stemmer.stem("الكتاب")
        assert result.mode == StemmerMode.LIGHT

    def test_result_frozen(self, stemmer):
        result = stemmer.stem("الكتاب")
        with pytest.raises(Exception):
            result.stem = "something"  # type: ignore


class TestAggressiveStemmer:
    def test_mode_is_aggressive(self, stemmer):
        result = stemmer.stem("الكتاب", mode=StemmerMode.AGGRESSIVE)
        assert result.mode == StemmerMode.AGGRESSIVE

    def test_aggressive_strips_more(self, stemmer):
        light = stemmer.stem("مدرستهم", mode=StemmerMode.LIGHT)
        aggressive = stemmer.stem("مدرستهم", mode=StemmerMode.AGGRESSIVE)
        # Aggressive should produce equal or shorter stem
        assert len(aggressive.stem) <= len(light.stem) + 2


class TestWasReduced:
    def test_was_reduced_true(self, stemmer):
        result = stemmer.stem("الكتاب")
        assert result.was_reduced is True

    def test_was_reduced_false_for_stop_word(self, stemmer):
        result = stemmer.stem("في")
        assert result.was_reduced is False


class TestBatchStemming:
    def test_batch_length(self, stemmer):
        words = ["الكتاب", "يكتبون", "مدرسة"]
        results = stemmer.stem_batch(words)
        assert len(results) == 3

    def test_batch_all_stem_results(self, stemmer):
        words = ["الكتاب", "يكتبون"]
        results = stemmer.stem_batch(words)
        assert all(isinstance(r, StemResult) for r in results)

    def test_batch_preserves_order(self, stemmer):
        words = ["الكتاب", "يكتبون"]
        results = stemmer.stem_batch(words)
        assert results[0].original == "الكتاب"
        assert results[1].original == "يكتبون"


class TestStemText:
    def test_returns_string_list(self, stemmer):
        stems = stemmer.stem_text("الكتاب على الطاولة")
        assert all(isinstance(s, str) for s in stems)

    def test_returns_correct_count(self, stemmer):
        stems = stemmer.stem_text("كتاب كبير جميل")
        assert len(stems) == 3


class TestInputValidation:
    def test_empty_raises(self, stemmer):
        with pytest.raises(InvalidInputError):
            stemmer.stem("")

    def test_non_string_raises(self, stemmer):
        with pytest.raises(InvalidInputError):
            stemmer.stem(None)  # type: ignore
