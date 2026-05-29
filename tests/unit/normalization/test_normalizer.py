"""Tests for ArabicNormalizer."""

import pytest

from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.normalization import ArabicNormalizer


@pytest.fixture
def normalizer() -> ArabicNormalizer:
    return ArabicNormalizer()


class TestDiacriticsRemoval:
    def test_removes_fatha(self, normalizer):
        result = normalizer.normalize("كَتَبَ")
        assert "َ" not in result.normalized

    def test_removes_kasra(self, normalizer):
        result = normalizer.normalize("كِتَاب")
        assert "ِ" not in result.normalized

    def test_removes_damma(self, normalizer):
        result = normalizer.normalize("يَذْهَبُ")
        assert "ُ" not in result.normalized

    def test_removes_full_tashkeel(self, normalizer):
        result = normalizer.normalize("هَذَا نَصٌّ مُشَكَّل")
        assert result.normalized == "هذا نص مشكل"
        assert "removed_diacritics" in result.changes

    def test_keeps_structure_without_diacritics(self, normalizer):
        text = "مرحبا بالعالم"
        result = normalizer.normalize(text)
        assert result.normalized == text
        assert "removed_diacritics" not in result.changes


class TestAlefNormalization:
    def test_normalizes_alef_hamza_above(self, normalizer):
        result = normalizer.normalize("أحمد")
        assert result.normalized == "احمد"
        assert "normalized_alef" in result.changes

    def test_normalizes_alef_hamza_below(self, normalizer):
        result = normalizer.normalize("إبراهيم")
        assert result.normalized == "ابراهيم"

    def test_normalizes_alef_madda(self, normalizer):
        result = normalizer.normalize("آمن")
        assert result.normalized == "امن"

    def test_no_change_for_bare_alef(self, normalizer):
        result = normalizer.normalize("الكتاب")
        assert "normalized_alef" not in result.changes


class TestTehMarbutaNormalization:
    def test_normalizes_teh_marbuta(self, normalizer):
        result = normalizer.normalize("مدرسة")
        assert result.normalized == "مدرسه"
        assert "normalized_teh_marbuta" in result.changes

    def test_disabled_teh_marbuta(self, normalizer):
        result = normalizer.normalize("مدرسة", normalize_teh_marbuta=False)
        assert result.normalized == "مدرسة"
        assert "normalized_teh_marbuta" not in result.changes


class TestTatweelRemoval:
    def test_removes_tatweel(self, normalizer):
        result = normalizer.normalize("جميـــل")
        assert "ـ" not in result.normalized
        assert "removed_tatweel" in result.changes

    def test_no_tatweel_no_change(self, normalizer):
        result = normalizer.normalize("جميل")
        assert "removed_tatweel" not in result.changes


class TestSocialMediaCleaning:
    def test_removes_urls(self, normalizer):
        result = normalizer.normalize("شوف الخبر ده http://example.com/news", remove_urls=True)
        assert "http" not in result.normalized
        assert "removed_urls" in result.changes

    def test_removes_mentions(self, normalizer):
        result = normalizer.normalize("شكرا @ahmed على المساعدة", remove_mentions=True)
        assert "@ahmed" not in result.normalized
        assert "removed_mentions" in result.changes

    def test_removes_hashtags(self, normalizer):
        result = normalizer.normalize("أحب #مصر كتير", remove_hashtags=True)
        assert "#مصر" not in result.normalized
        assert "removed_hashtags" in result.changes

    def test_removes_emojis(self, normalizer):
        result = normalizer.normalize("حلو كتير 🔥", remove_emojis=True)
        assert "🔥" not in result.normalized
        assert "removed_emojis" in result.changes

    def test_clean_social_shortcut(self, normalizer):
        result = normalizer.clean_social("@user شوف ده 🔥 http://x.com #tag")
        assert "@user" not in result.normalized
        assert "🔥" not in result.normalized
        assert "http" not in result.normalized
        assert "#tag" not in result.normalized


class TestWhitespaceNormalization:
    def test_collapses_multiple_spaces(self, normalizer):
        result = normalizer.normalize("كلمة    وكلمة")
        assert "    " not in result.normalized
        assert "normalized_whitespace" in result.changes

    def test_strips_leading_trailing(self, normalizer):
        result = normalizer.normalize("  نص هنا  ")
        assert not result.normalized.startswith(" ")
        assert not result.normalized.endswith(" ")


class TestWasChangedProperty:
    def test_was_changed_true(self, normalizer):
        result = normalizer.normalize("هَذَا")
        assert result.was_changed is True

    def test_was_changed_false(self, normalizer):
        result = normalizer.normalize("هذا")
        assert result.was_changed is False


class TestMinimalNormalization:
    def test_minimal_only_diacritics_and_spaces(self, normalizer):
        result = normalizer.minimal("هَذَا نَصٌّ")
        assert "َ" not in result.normalized
        assert result.normalized == "هذا نص"


class TestContainsArabic:
    def test_arabic_text(self, normalizer):
        assert normalizer.contains_arabic("مرحبا")

    def test_latin_text(self, normalizer):
        assert not normalizer.contains_arabic("hello world")

    def test_mixed_text(self, normalizer):
        assert normalizer.contains_arabic("hello مرحبا world")


class TestInvalidInput:
    def test_raises_on_non_string(self, normalizer):
        with pytest.raises(InvalidInputError):
            normalizer.normalize(123)  # type: ignore
