"""
Performance tests for arabic-nlp-toolkit.

These tests measure execution time to catch regressions.
Run with: pytest tests/performance/ -v --tb=short

Note: These are not strict benchmarks — they just ensure
operations complete within reasonable time bounds.
"""

import time

import pytest

from arabic_nlp import ArabicNLP


@pytest.fixture(scope="module")
def nlp():
    return ArabicNLP()


SAMPLE_SHORT = "ازيك عامل ايه النهارده يا صديقي؟"
SAMPLE_MEDIUM = """
    زار محمد صلاح ملعب الأهلي في القاهرة أمس الجمعة.
    وقال الدكتور أحمد إن الزيارة كانت رائعة جداً وأنه سعيد بلقاء الجماهير.
    تجمع آلاف المشجعين أمام الملعب للترحيب باللاعب المشهور.
"""
SAMPLE_LONG = SAMPLE_MEDIUM * 10


class TestDialectDetectionPerformance:
    def test_short_text_under_100ms(self, nlp):
        t0 = time.perf_counter()
        nlp.detect_dialect(SAMPLE_SHORT)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 100, f"Dialect detection took {elapsed:.1f}ms (limit: 100ms)"

    def test_medium_text_under_200ms(self, nlp):
        t0 = time.perf_counter()
        nlp.detect_dialect(SAMPLE_MEDIUM)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 200, f"Dialect detection took {elapsed:.1f}ms (limit: 200ms)"


class TestSentimentPerformance:
    def test_short_text_under_50ms(self, nlp):
        t0 = time.perf_counter()
        nlp.sentiment(SAMPLE_SHORT)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 50, f"Sentiment took {elapsed:.1f}ms (limit: 50ms)"

    def test_batch_100_texts_under_2s(self, nlp):
        texts = [SAMPLE_SHORT] * 100
        t0 = time.perf_counter()
        nlp.batch_sentiment(texts)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"Batch (100 items) took {elapsed:.2f}s (limit: 2s)"


class TestTokenizationPerformance:
    def test_short_text_under_20ms(self, nlp):
        t0 = time.perf_counter()
        nlp.tokenize(SAMPLE_SHORT)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 20, f"Tokenization took {elapsed:.1f}ms (limit: 20ms)"

    def test_long_text_under_500ms(self, nlp):
        t0 = time.perf_counter()
        nlp.tokenize(SAMPLE_LONG)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 500, f"Tokenization took {elapsed:.1f}ms (limit: 500ms)"


class TestNormalizationPerformance:
    def test_short_text_under_10ms(self, nlp):
        t0 = time.perf_counter()
        nlp.normalize(SAMPLE_SHORT)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 10, f"Normalization took {elapsed:.1f}ms (limit: 10ms)"


class TestNERPerformance:
    def test_medium_text_under_200ms(self, nlp):
        t0 = time.perf_counter()
        nlp.extract_entities(SAMPLE_MEDIUM)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 200, f"NER took {elapsed:.1f}ms (limit: 200ms)"


class TestFullPipelinePerformance:
    def test_full_pipeline_under_500ms(self, nlp):
        t0 = time.perf_counter()
        nlp.analyze(SAMPLE_SHORT)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 500, f"Full pipeline took {elapsed:.1f}ms (limit: 500ms)"


class TestStemmerPerformance:
    def test_stem_100_words_under_100ms(self, nlp):
        words = ["الكتاب", "يكتبون", "مدرسة"] * 34
        from arabic_nlp.stemmer import ArabicStemmer

        stemmer = ArabicStemmer()
        t0 = time.perf_counter()
        stemmer.stem_batch(words)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 100, f"Stemming 100 words took {elapsed:.1f}ms (limit: 100ms)"
