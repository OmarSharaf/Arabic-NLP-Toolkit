"""Tests for keyword extraction."""

import pytest

from arabic_nlp import ArabicNLP
from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.keywords import KeywordExtractor


@pytest.fixture
def extractor() -> KeywordExtractor:
    return KeywordExtractor()


@pytest.fixture
def nlp() -> ArabicNLP:
    return ArabicNLP()


class TestKeywordExtractor:
    def test_extracts_keywords(self, extractor):
        text = "الذكاء الاصطناعي يغير التعليم في العالم العربي"
        result = extractor.extract(text, top_n=5)
        assert len(result.keywords) > 0
        assert result.keywords[0].rank == 1
        assert 0.0 < result.keywords[0].score <= 1.0

    def test_filters_stopwords(self, extractor):
        text = "في من على الذكاء الاصطناعي"
        result = extractor.extract(text, top_n=10)
        texts = {k.text for k in result.keywords}
        assert "في" not in texts
        assert "من" not in texts

    def test_empty_raises(self, extractor):
        with pytest.raises(InvalidInputError):
            extractor.extract("   ")

    def test_as_dict(self, extractor):
        result = extractor.extract("التعليم الرقمي مستقبل التعليم", top_n=3)
        d = result.as_dict()
        assert isinstance(d, dict)
        assert all(isinstance(v, float) for v in d.values())

    def test_batch(self, extractor):
        texts = ["الذكاء الاصطناعي", "التعليم الرقمي"]
        results = extractor.extract_batch(texts, top_n=3)
        assert len(results) == 2


class TestArabicNLPKeywords:
    def test_extract_keywords_via_facade(self, nlp):
        result = nlp.extract_keywords(
            "الاقتصاد المصري ينمو بشكل ملحوظ هذا العام",
            top_n=5,
        )
        assert result.content_tokens > 0
        assert len(result.keywords) <= 5

    def test_content_words(self, nlp):
        words = nlp.content_words("في المدرسة تعلم الأطفال القراءة")
        assert "في" not in words
        assert len(words) >= 2
