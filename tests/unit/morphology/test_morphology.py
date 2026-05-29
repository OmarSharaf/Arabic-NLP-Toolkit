"""Tests for ArabicMorphAnalyzer."""

import pytest

from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.morphology import (
    ArabicMorphAnalyzer,
    Definiteness,
    Gender,
    Number,
    VerbTense,
)


@pytest.fixture
def analyzer() -> ArabicMorphAnalyzer:
    return ArabicMorphAnalyzer()


class TestDefiniteArticle:
    def test_detects_definite(self, analyzer):
        result = analyzer.analyze("الكتاب")
        assert result.definiteness == Definiteness.DEFINITE

    def test_detects_indefinite(self, analyzer):
        result = analyzer.analyze("كتاب")
        assert result.definiteness == Definiteness.INDEFINITE

    def test_strips_al_from_stem(self, analyzer):
        result = analyzer.analyze("الكتاب")
        assert result.stem == "كتاب"
        assert "ال" in result.prefixes

    def test_compound_prefix_waal(self, analyzer):
        result = analyzer.analyze("والكتاب")
        assert result.definiteness == Definiteness.DEFINITE
        assert len(result.prefixes) >= 1


class TestGenderDetection:
    def test_feminine_teh_marbuta(self, analyzer):
        result = analyzer.analyze("مدرسة")
        assert result.gender == Gender.FEMININE

    def test_masculine_default(self, analyzer):
        result = analyzer.analyze("كتاب")
        assert result.gender == Gender.MASCULINE

    def test_alef_maqsura_feminine(self, analyzer):
        result = analyzer.analyze("كبرى")
        assert result.gender == Gender.FEMININE


class TestNumberDetection:
    def test_singular(self, analyzer):
        result = analyzer.analyze("كتاب")
        assert result.number == Number.SINGULAR

    def test_plural_waw_nun(self, analyzer):
        result = analyzer.analyze("كاتبون")
        assert result.number == Number.PLURAL

    def test_plural_ya_nun(self, analyzer):
        result = analyzer.analyze("كاتبين")
        assert result.number == Number.PLURAL

    def test_plural_aat(self, analyzer):
        result = analyzer.analyze("مدرسات")
        assert result.number == Number.PLURAL


class TestRootExtraction:
    def test_extracts_3_letter_root(self, analyzer):
        result = analyzer.analyze("كتب")
        assert result.root is not None
        assert len(result.root) == 3

    def test_root_not_none_for_common_word(self, analyzer):
        result = analyzer.analyze("ذهب")
        assert result.root is not None

    def test_short_word_root(self, analyzer):
        result = analyzer.analyze("يد")
        # Very short — root may be None or 2-char
        assert result.root is None or len(result.root) >= 2


class TestVerbAnalysis:
    def test_present_tense_ya(self, analyzer):
        result = analyzer.analyze("يكتب")
        assert result.is_verb is True
        assert result.verb_tense == VerbTense.PRESENT

    def test_present_tense_ta(self, analyzer):
        result = analyzer.analyze("تكتب")
        assert result.is_verb is True

    def test_past_tense(self, analyzer):
        result = analyzer.analyze("كتبت")
        assert result.is_verb is True

    def test_noun_is_not_verb(self, analyzer):
        result = analyzer.analyze("الكتاب")
        assert result.is_verb is False
        assert result.is_noun is True


class TestBatchAnalysis:
    def test_batch_returns_correct_count(self, analyzer):
        words = ["الكتاب", "يكتب", "مدرسة"]
        results = analyzer.analyze_batch(words)
        assert len(results) == 3

    def test_batch_preserves_order(self, analyzer):
        words = ["الكتاب", "يكتب"]
        results = analyzer.analyze_batch(words)
        assert results[0].word == "الكتاب"
        assert results[1].word == "يكتب"


class TestResultProperties:
    def test_result_is_frozen(self, analyzer):
        result = analyzer.analyze("الكتاب")
        with pytest.raises(Exception):
            result.stem = "something"  # type: ignore

    def test_processing_time_positive(self, analyzer):
        result = analyzer.analyze("مرحبا")
        assert result.processing_time_ms >= 0

    def test_word_preserved(self, analyzer):
        result = analyzer.analyze("الكتاب")
        assert result.word == "الكتاب"

    def test_confidence_in_range(self, analyzer):
        result = analyzer.analyze("الكتاب")
        assert 0 <= result.confidence <= 1


class TestInputValidation:
    def test_empty_string_raises(self, analyzer):
        with pytest.raises(InvalidInputError):
            analyzer.analyze("")

    def test_non_string_raises(self, analyzer):
        with pytest.raises(InvalidInputError):
            analyzer.analyze(123)  # type: ignore

    def test_whitespace_only_raises(self, analyzer):
        with pytest.raises(InvalidInputError):
            analyzer.analyze("   ")
