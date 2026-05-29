"""Tests for Transliterator."""

import pytest

from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.models import Script, TransliterationResult
from arabic_nlp.transliteration import Transliterator


@pytest.fixture
def t() -> Transliterator:
    return Transliterator()


class TestArabicToFranco:
    def test_basic_transliteration(self, t):
        result = t.transliterate("مرحبا")
        assert len(result.transliterated) > 0

    def test_ح_becomes_7(self, t):
        result = t.transliterate("حسن")
        assert "7" in result.transliterated

    def test_ع_becomes_3(self, t):
        result = t.transliterate("عمر")
        assert "3" in result.transliterated

    def test_script_metadata(self, t):
        result = t.transliterate("مرحبا")
        assert result.source_script == Script.ARABIC
        assert result.target_script == Script.FRANCO

    def test_result_type(self, t):
        result = t.transliterate("مرحبا")
        assert isinstance(result, TransliterationResult)


class TestArabicToBuckwalter:
    def test_basic_buckwalter(self, t):
        result = t.transliterate("بسم", target=Script.BUCKWALTER)
        assert result.transliterated == "bsm"

    def test_buckwalter_alef(self, t):
        result = t.transliterate("الله", target=Script.BUCKWALTER)
        assert "A" in result.transliterated  # ا → A in Buckwalter

    def test_buckwalter_shin(self, t):
        result = t.transliterate("شمس", target=Script.BUCKWALTER)
        assert "$" in result.transliterated  # ش → $ in Buckwalter


class TestFrancoToArabic:
    def test_basic_franco_to_arabic(self, t):
        result = t.transliterate("marhaba", source=Script.FRANCO, target=Script.ARABIC)
        assert len(result.transliterated) > 0

    def test_digit_3_becomes_ain(self, t):
        result = t.transliterate("3arab", source=Script.FRANCO, target=Script.ARABIC)
        assert "ع" in result.transliterated

    def test_digit_7_becomes_ha(self, t):
        result = t.transliterate("7asan", source=Script.FRANCO, target=Script.ARABIC)
        assert "ح" in result.transliterated

    def test_source_target_metadata(self, t):
        result = t.transliterate("marhaba", source=Script.FRANCO, target=Script.ARABIC)
        assert result.source_script == Script.FRANCO
        assert result.target_script == Script.ARABIC


class TestNoOpTransliteration:
    def test_same_script_returns_original(self, t):
        result = t.transliterate("مرحبا", source=Script.ARABIC, target=Script.ARABIC)
        assert result.transliterated == "مرحبا"


class TestProcessingTime:
    def test_processing_time_positive(self, t):
        result = t.transliterate("مرحبا")
        assert result.processing_time_ms >= 0


class TestInputValidation:
    def test_empty_raises(self, t):
        with pytest.raises(InvalidInputError):
            t.transliterate("")

    def test_whitespace_raises(self, t):
        with pytest.raises(InvalidInputError):
            t.transliterate("   ")
