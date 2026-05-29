"""Tests for SentimentAnalyzer."""

import pytest

from arabic_nlp.models import Dialect, SentimentLabel
from arabic_nlp.sentiment import SentimentAnalyzer


@pytest.fixture
def analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


class TestPositiveSentiment:
    def test_obvious_positive(self, analyzer):
        result = analyzer.analyze("المنتج ده رائع جداً وبحبه أوي")
        assert result.label == SentimentLabel.POSITIVE

    def test_positive_score_dominates(self, analyzer):
        result = analyzer.analyze("ممتاز جداً وتحفة")
        assert result.positive_score > result.negative_score

    def test_intensifier_boosts_score(self, analyzer):
        weak = analyzer.analyze("كويس")
        strong = analyzer.analyze("كويس جداً")
        assert strong.positive_score >= weak.positive_score


class TestNegativeSentiment:
    def test_obvious_negative(self, analyzer):
        result = analyzer.analyze("المنتج ده سيء جداً ومش عاجبني")
        assert result.label == SentimentLabel.NEGATIVE

    def test_negative_score_dominates(self, analyzer):
        result = analyzer.analyze("وحش ومزعج وبايخ")
        assert result.negative_score > result.positive_score


class TestNegationHandling:
    def test_negated_positive_becomes_negative(self, analyzer):
        without_neg = analyzer.analyze("كويس")
        with_neg = analyzer.analyze("مش كويس")
        assert with_neg.negative_score >= without_neg.negative_score

    def test_negation_flips_sentiment(self, analyzer):
        result = analyzer.analyze("مش رائع ولا حلو")
        assert result.label in (SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL)


class TestNeutralSentiment:
    def test_factual_text_neutral(self, analyzer):
        result = analyzer.analyze("ذهب الولد إلى المدرسة في الصباح")
        assert result.label == SentimentLabel.NEUTRAL

    def test_neutral_scores(self, analyzer):
        result = analyzer.analyze("ذهب إلى البيت")
        assert result.neutral_score > 0


class TestScoreRange:
    def test_scores_between_0_and_1(self, analyzer):
        result = analyzer.analyze("رائع جداً")
        assert 0 <= result.positive_score <= 1
        assert 0 <= result.negative_score <= 1
        assert 0 <= result.neutral_score <= 1
        assert 0 <= result.score <= 1

    def test_processing_time_positive(self, analyzer):
        result = analyzer.analyze("مرحبا")
        assert result.processing_time_ms >= 0

    def test_dialect_preserved(self, analyzer):
        result = analyzer.analyze("ممتاز", dialect=Dialect.EGYPTIAN)
        assert result.dialect == Dialect.EGYPTIAN


class TestBatchAnalysis:
    def test_batch_returns_correct_count(self, analyzer):
        texts = ["رائع", "سيء", "عادي"]
        results = analyzer.analyze_batch(texts)
        assert len(results) == 3

    def test_batch_results_correct_types(self, analyzer):
        from arabic_nlp.models import SentimentResult

        results = analyzer.analyze_batch(["رائع", "سيء"])
        assert all(isinstance(r, SentimentResult) for r in results)
