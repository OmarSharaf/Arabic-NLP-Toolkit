"""Tests for text profiling."""

import pytest

from arabic_nlp import ArabicNLP
from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.profiling import TextProfiler, TextRegister


@pytest.fixture
def profiler() -> TextProfiler:
    return TextProfiler()


@pytest.fixture
def nlp() -> ArabicNLP:
    return ArabicNLP()


class TestTextProfiler:
    def test_formal_text(self, profiler):
        text = "إن التعليم هو أساس تقدم الأمم وازدهارها في العصر الحديث."
        result = profiler.profile(text)
        assert result.register in (
            TextRegister.FORMAL_MSA,
            TextRegister.SEMI_FORMAL,
            TextRegister.UNKNOWN,
        )
        assert result.quality_score >= 0.5

    def test_social_media_text(self, profiler):
        text = "ازيك يا صاحبي 😂 #مصر https://example.com"
        result = profiler.profile(text)
        assert result.register == TextRegister.SOCIAL_MEDIA
        assert result.has_emojis or result.has_urls or result.has_hashtags
        assert result.quality_score < 1.0

    def test_recommendations_not_empty(self, profiler):
        result = profiler.profile("مرحبا بالعالم")
        assert len(result.recommendations) >= 1

    def test_empty_raises(self, profiler):
        with pytest.raises(InvalidInputError):
            profiler.profile("")

    def test_is_social_media_property(self, profiler):
        result = profiler.profile("شوف اللينك @user #tag 😀")
        assert result.is_social_media


class TestDocumentAnalysis:
    def test_analyze_document(self, nlp):
        doc = nlp.analyze_document(
            "زيارة محمد صلاح للقاهرة كانت رائعة جداً",
            include_keywords=True,
            include_profile=True,
        )
        assert doc.token_count > 0
        assert doc.dialect.dialect is not None
        assert doc.sentiment.label is not None
        assert len(doc.keywords) > 0
        assert doc.profile is not None

    def test_to_json(self, nlp):
        doc = nlp.analyze_document("المنتج ممتاز")
        json_str = doc.to_json()
        assert "dialect" in json_str
        assert "sentiment" in json_str

    def test_summary(self, nlp):
        doc = nlp.analyze_document("خدمة ممتازة")
        s = doc.summary
        assert "dialect" in s
        assert "sentiment" in s

    def test_batch_analyze(self, nlp):
        batch = nlp.batch_analyze(["رائع", "سيء"], include_keywords=False)
        assert batch.total == 2
        assert batch.successful == 2
