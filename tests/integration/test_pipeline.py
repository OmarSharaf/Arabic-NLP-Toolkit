"""
Integration tests for the full ArabicNLP pipeline.
Tests that all modules work together correctly.
"""

import pytest

from arabic_nlp import ArabicNLP
from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.models import (
    Dialect,
    DialectResult,
    NERResult,
    NormalizationResult,
    Script,
    SentimentLabel,
    SentimentResult,
    TokenType,
    TransliterationResult,
)


@pytest.fixture(scope="module")
def nlp() -> ArabicNLP:
    """Shared ArabicNLP instance across all integration tests."""
    return ArabicNLP()


class TestDialectDetection:
    def test_egyptian_sentence(self, nlp):
        result = nlp.detect_dialect("ازيك عامل ايه النهارده يا صديقي؟")
        assert isinstance(result, DialectResult)
        assert result.dialect == Dialect.EGYPTIAN
        assert result.confidence > 0.4

    def test_gulf_sentence(self, nlp):
        result = nlp.detect_dialect("شلون حالك اليوم يا أخي؟")
        assert result.dialect == Dialect.GULF

    def test_levantine_sentence(self, nlp):
        result = nlp.detect_dialect("شو بدك هلق؟")
        assert result.dialect == Dialect.LEVANTINE

    def test_result_is_frozen(self, nlp):
        result = nlp.detect_dialect("مرحبا")
        with pytest.raises(Exception):
            result.dialect = Dialect.GULF  # type: ignore


class TestSentimentAnalysis:
    def test_positive_review(self, nlp):
        result = nlp.sentiment("المنتج رائع جداً وأنصح به")
        assert isinstance(result, SentimentResult)
        assert result.label == SentimentLabel.POSITIVE

    def test_negative_review(self, nlp):
        result = nlp.sentiment("المنتج سيء جداً ولا أنصح به أبداً")
        assert result.label == SentimentLabel.NEGATIVE

    def test_with_dialect_hint(self, nlp):
        result = nlp.sentiment("المنتج تحفة أوي", dialect=Dialect.EGYPTIAN)
        assert result.dialect == Dialect.EGYPTIAN

    def test_scores_sum_approximately_one(self, nlp):
        result = nlp.sentiment("رائع")
        total = result.positive_score + result.negative_score + result.neutral_score
        assert abs(total - 1.0) < 0.01


class TestTokenization:
    def test_basic_tokenization(self, nlp):
        tokens = nlp.tokenize("مرحبا بمصر الحبيبة")
        assert len(tokens) > 0
        assert all(hasattr(t, "text") for t in tokens)

    def test_offsets_correct(self, nlp):
        text = "مرحبا بالعالم"
        tokens = nlp.tokenize(text)
        for tok in tokens:
            assert text[tok.start : tok.end] == tok.text

    def test_exclude_punctuation(self, nlp):
        tokens = nlp.tokenize("مرحبا!", include_punctuation=False)
        punc = [t for t in tokens if t.token_type == TokenType.PUNCTUATION]
        assert len(punc) == 0


class TestNamedEntityRecognition:
    def test_detects_person(self, nlp):
        result = nlp.extract_entities("زار محمد صلاح ملعب الأهلي")
        assert isinstance(result, NERResult)
        persons = result.persons
        assert len(persons) > 0

    def test_detects_location(self, nlp):
        result = nlp.extract_entities("سافر إلى القاهرة")
        locations = result.locations
        assert len(locations) > 0
        assert any("القاهرة" in e.text for e in locations)

    def test_detects_organization(self, nlp):
        result = nlp.extract_entities("يلعب في الأهلي منذ سنوات")
        orgs = result.organizations
        assert len(orgs) > 0

    def test_entities_sorted_by_position(self, nlp):
        result = nlp.extract_entities("زار محمد صلاح القاهرة")
        starts = [e.start for e in result.entities]
        assert starts == sorted(starts)


class TestNormalization:
    def test_removes_diacritics(self, nlp):
        result = nlp.normalize("هَذَا نَصٌّ مُشَكَّل")
        assert isinstance(result, NormalizationResult)
        assert "َ" not in result.normalized

    def test_social_media_cleaning(self, nlp):
        result = nlp.normalize(
            "@user شوف ده 🔥 http://example.com #مصر",
            remove_mentions=True,
            remove_emojis=True,
            remove_urls=True,
            remove_hashtags=True,
        )
        assert "@user" not in result.normalized
        assert "🔥" not in result.normalized


class TestTransliteration:
    def test_arabic_to_franco(self, nlp):
        result = nlp.transliterate("مرحبا")
        assert isinstance(result, TransliterationResult)
        assert result.source_script == Script.ARABIC
        assert result.target_script == Script.FRANCO
        assert len(result.transliterated) > 0

    def test_arabic_to_buckwalter(self, nlp):
        result = nlp.transliterate("بسم", target=Script.BUCKWALTER)
        assert result.transliterated == "bsm"

    def test_franco_to_arabic(self, nlp):
        result = nlp.transliterate(
            "marhaba",
            source=Script.FRANCO,
            target=Script.ARABIC,
        )
        assert len(result.transliterated) > 0


class TestStopWords:
    def test_msa_stopwords(self, nlp):
        sw = nlp.get_stopwords(Dialect.MSA)
        assert "في" in sw
        assert "من" in sw

    def test_egyptian_stopwords(self, nlp):
        sw = nlp.get_stopwords(Dialect.EGYPTIAN)
        assert "ايه" in sw

    def test_is_stopword(self, nlp):
        assert nlp.is_stopword("في", Dialect.MSA) is True
        assert nlp.is_stopword("برج", Dialect.MSA) is False


class TestFullPipeline:
    def test_analyze_returns_all_keys(self, nlp):
        analysis = nlp.analyze("زيارة محمد صلاح لملعب الأهلي كانت رائعة جداً")
        assert "dialect" in analysis
        assert "sentiment" in analysis
        assert "tokens" in analysis
        assert "entities" in analysis
        assert "normalized" in analysis
        assert "pipeline_time_ms" in analysis

    def test_analyze_types(self, nlp):
        analysis = nlp.analyze("رائع جداً")
        assert isinstance(analysis["dialect"], DialectResult)
        assert isinstance(analysis["sentiment"], SentimentResult)
        assert isinstance(analysis["tokens"], list)
        assert isinstance(analysis["entities"], NERResult)
        assert isinstance(analysis["normalized"], NormalizationResult)

    def test_pipeline_time_positive(self, nlp):
        analysis = nlp.analyze("مرحبا")
        assert analysis["pipeline_time_ms"] >= 0


class TestInputValidation:
    def test_empty_string_raises(self, nlp):
        with pytest.raises(InvalidInputError):
            nlp.detect_dialect("")

    def test_whitespace_only_raises(self, nlp):
        with pytest.raises(InvalidInputError):
            nlp.sentiment("   ")

    def test_non_string_raises(self, nlp):
        with pytest.raises(InvalidInputError):
            nlp.tokenize(123)  # type: ignore


class TestBatchOperations:
    def test_batch_sentiment(self, nlp):
        texts = ["رائع", "سيء", "عادي", "ممتاز"]
        result = nlp.batch_sentiment(texts)
        assert result.total == 4
        assert result.successful == 4
        assert result.failed == 0

    def test_batch_dialect(self, nlp):
        texts = ["ازيك عامل ايه", "شلون حالك", "شو بدك"]
        result = nlp.batch_dialect(texts)
        assert result.total == 3
        assert result.success_rate == 1.0


class TestRepr:
    def test_repr_includes_version(self, nlp):
        r = repr(nlp)
        assert "ArabicNLP" in r
        assert "version=" in r
