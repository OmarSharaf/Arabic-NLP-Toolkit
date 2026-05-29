"""Tests for arabic_nlp.utils."""

from arabic_nlp.utils import (
    arabic_to_western_digits,
    compute_similarity,
    count_arabic_words,
    detect_language,
    extract_arabic_only,
    extract_numbers,
    get_readability,
    get_statistics,
    has_diacritics,
    is_arabic,
    remove_diacritics,
    sentence_split,
    truncate_arabic,
    western_to_arabic_digits,
    word_frequency,
)


class TestGetStatistics:
    def test_word_count(self):
        stats = get_statistics("مرحبا بكم في مصر")
        assert stats.total_words == 4

    def test_arabic_words_count(self):
        stats = get_statistics("مرحبا بكم في مصر")
        assert stats.arabic_words == 4

    def test_unique_words(self):
        stats = get_statistics("كلمة كلمة أخرى")
        assert stats.unique_words == 2  # "كلمة" repeated

    def test_arabic_ratio_pure_arabic(self):
        stats = get_statistics("مرحبا")
        assert stats.arabic_ratio > 0.8

    def test_has_diacritics_true(self):
        stats = get_statistics("هَذَا")
        assert stats.has_diacritics is True

    def test_has_diacritics_false(self):
        stats = get_statistics("هذا")
        assert stats.has_diacritics is False

    def test_empty_text(self):
        stats = get_statistics("")
        assert stats.total_words == 0

    def test_most_common_words(self):
        stats = get_statistics("كلمة كلمة أخرى")
        assert len(stats.most_common_words) >= 1
        assert stats.most_common_words[0][0] == "كلمة"
        assert stats.most_common_words[0][1] == 2

    def test_sentences_count(self):
        stats = get_statistics("جملة أولى. جملة ثانية. جملة ثالثة.")
        assert stats.sentences == 3


class TestDetectLanguage:
    def test_pure_arabic(self):
        ratios = detect_language("مرحبا بالعالم")
        assert ratios["arabic"] > 0.8

    def test_mixed_arabic_latin(self):
        ratios = detect_language("hello مرحبا")
        assert ratios["arabic"] > 0
        assert ratios["latin"] > 0

    def test_empty_returns_empty(self):
        ratios = detect_language("")
        assert ratios == {}

    def test_digits(self):
        ratios = detect_language("12345")
        assert ratios["digit"] > 0.9


class TestIsArabic:
    def test_arabic_text_true(self):
        assert is_arabic("مرحبا بالعالم") is True

    def test_latin_text_false(self):
        assert is_arabic("hello world") is False

    def test_custom_threshold(self):
        assert is_arabic("مرحبا hello", threshold=0.3) is True
        assert is_arabic("مرحبا hello", threshold=0.9) is False


class TestReadability:
    def test_returns_level(self):
        score = get_readability("الكتاب على الطاولة.")
        assert score.difficulty_level in ("easy", "moderate", "difficult", "very_difficult")

    def test_score_in_range(self):
        score = get_readability("مرحبا بكم.")
        assert 0 <= score.difficulty_score <= 1

    def test_recommendation_not_empty(self):
        score = get_readability("نص بسيط.")
        assert score.recommendation

    def test_word_count_correct(self):
        score = get_readability("كلمة واحدة اثنتان ثلاثة.")
        assert score.word_count >= 3


class TestSimilarity:
    def test_identical_texts_high_similarity(self):
        sim = compute_similarity("المنتج رائع جداً", "المنتج رائع جداً")
        assert sim.jaccard == 1.0
        assert sim.cosine == 1.0

    def test_different_texts_low_similarity(self):
        sim = compute_similarity("مرحبا بالعالم", "السماء زرقاء جميلة")
        assert sim.jaccard < 0.3

    def test_partial_overlap(self):
        sim = compute_similarity("المنتج رائع جداً", "المنتج ممتاز جداً")
        assert 0.3 < sim.jaccard < 0.9

    def test_similarity_scores_in_range(self):
        sim = compute_similarity("نص أول", "نص ثانٍ")
        assert 0 <= sim.jaccard <= 1
        assert 0 <= sim.cosine <= 1
        assert 0 <= sim.overlap <= 1


class TestStringHelpers:
    def test_remove_diacritics(self):
        assert remove_diacritics("هَذَا") == "هذا"

    def test_has_diacritics_true(self):
        assert has_diacritics("هَذَا") is True

    def test_has_diacritics_false(self):
        assert has_diacritics("هذا") is False

    def test_count_arabic_words(self):
        assert count_arabic_words("مرحبا بكم في مصر") == 4

    def test_extract_arabic_only(self):
        result = extract_arabic_only("hello مرحبا world بالعالم 123")
        assert "مرحبا" in result
        assert "hello" not in result

    def test_extract_numbers(self):
        numbers = extract_numbers("عدد 123 وعدد ٤٥٦")
        assert len(numbers) == 2

    def test_arabic_to_western_digits(self):
        assert arabic_to_western_digits("٢٠٢٥") == "2025"

    def test_western_to_arabic_digits(self):
        assert western_to_arabic_digits("2025") == "٢٠٢٥"

    def test_truncate_arabic_under_limit(self):
        text = "كلمة واحدة"
        assert truncate_arabic(text, 5) == text

    def test_truncate_arabic_over_limit(self):
        text = "هذه جملة فيها كلمات كثيرة جداً"
        result = truncate_arabic(text, 3)
        assert result.endswith("...")
        assert len(result.split()) <= 4  # 3 words + ellipsis

    def test_word_frequency(self):
        freq = word_frequency("كلمة كلمة أخرى")
        assert freq[0] == ("كلمة", 2)

    def test_sentence_split(self):
        sentences = sentence_split("جملة أولى. جملة ثانية.")
        assert len(sentences) == 2
