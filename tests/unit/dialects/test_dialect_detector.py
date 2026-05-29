"""Tests for DialectDetector."""

import pytest

from arabic_nlp.dialects import DialectDetector
from arabic_nlp.models import Dialect


@pytest.fixture
def detector() -> DialectDetector:
    return DialectDetector()


class TestEgyptianDetection:
    def test_detects_ايه(self, detector):
        result = detector.detect("ايه ده بقى")
        assert result.dialect == Dialect.EGYPTIAN

    def test_detects_ازيك(self, detector):
        result = detector.detect("ازيك عامل ايه النهارده؟")
        assert result.dialect == Dialect.EGYPTIAN
        assert result.confidence > 0.5

    def test_detects_النهارده(self, detector):
        result = detector.detect("النهارده الجو حلو اوي")
        assert result.dialect == Dialect.EGYPTIAN

    def test_confidence_between_0_and_1(self, detector):
        result = detector.detect("ايه ده")
        assert 0.0 <= result.confidence <= 1.0

    def test_all_scores_sum_to_1(self, detector):
        result = detector.detect("ازيك عامل ايه")
        total = sum(s.confidence for s in result.all_scores)
        assert abs(total - 1.0) < 0.01


class TestGulfDetection:
    def test_detects_شلون(self, detector):
        result = detector.detect("شلون حالك اليوم؟")
        assert result.dialect == Dialect.GULF

    def test_detects_وين(self, detector):
        result = detector.detect("وين رحت الليلة؟")
        assert result.dialect == Dialect.GULF


class TestLevantineDetection:
    def test_detects_شو(self, detector):
        result = detector.detect("شو بدك هلق؟")
        assert result.dialect == Dialect.LEVANTINE

    def test_detects_هلق(self, detector):
        result = detector.detect("هلق وين رح تروح؟")
        assert result.dialect == Dialect.LEVANTINE


class TestMSADetection:
    def test_detects_msa_function_words(self, detector):
        result = detector.detect("الذي يجتهد في عمله سينجح بإذن الله")
        # Should score MSA higher or at least not Egyptian
        msa_score = next((s.confidence for s in result.all_scores if s.dialect == Dialect.MSA), 0)
        egy_score = next(
            (s.confidence for s in result.all_scores if s.dialect == Dialect.EGYPTIAN), 0
        )
        assert msa_score >= egy_score


class TestResultStructure:
    def test_has_all_dialects_in_scores(self, detector):
        result = detector.detect("مرحبا")
        score_dialects = {s.dialect for s in result.all_scores}
        expected = {
            Dialect.MSA,
            Dialect.EGYPTIAN,
            Dialect.GULF,
            Dialect.LEVANTINE,
            Dialect.MAGHREBI,
            Dialect.IRAQI,
        }
        assert expected.issubset(score_dialects)

    def test_is_arabic_true_for_arabic(self, detector):
        result = detector.detect("مرحبا بالعالم")
        assert result.is_arabic is True

    def test_processing_time_positive(self, detector):
        result = detector.detect("مرحبا")
        assert result.processing_time_ms >= 0

    def test_text_preserved_in_result(self, detector):
        text = "ازيك يا صديقي"
        result = detector.detect(text)
        assert result.text == text

    def test_dialect_name_ar(self, detector):
        result = detector.detect("ازيك عامل ايه")
        if result.dialect == Dialect.EGYPTIAN:
            assert result.dialect_name_ar == "مصري"


class TestBatchDetection:
    def test_batch_returns_list(self, detector):
        texts = ["ايه ده", "شلون حالك", "شو بدك"]
        results = detector.detect_batch(texts)
        assert len(results) == 3

    def test_batch_preserves_order(self, detector):
        texts = ["ايه ده", "شلون حالك"]
        results = detector.detect_batch(texts)
        assert results[0].text == texts[0]
        assert results[1].text == texts[1]
