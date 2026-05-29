"""Tests for StopWords module."""

import pytest

from arabic_nlp.exceptions import UnsupportedDialectError
from arabic_nlp.models import Dialect
from arabic_nlp.stopwords import StopWords


@pytest.fixture
def sw() -> StopWords:
    return StopWords()


class TestMSAStopWords:
    def test_في_is_stopword(self, sw):
        assert sw.is_stopword("في", Dialect.MSA) is True

    def test_من_is_stopword(self, sw):
        assert sw.is_stopword("من", Dialect.MSA) is True

    def test_برج_is_not_stopword(self, sw):
        assert sw.is_stopword("برج", Dialect.MSA) is False

    def test_returns_frozenset(self, sw):
        result = sw.get_msa()
        assert isinstance(result, frozenset)

    def test_msa_not_empty(self, sw):
        assert len(sw.get_msa()) > 10


class TestEgyptianStopWords:
    def test_ايه_is_stopword(self, sw):
        assert sw.is_stopword("ايه", Dialect.EGYPTIAN) is True

    def test_دلوقتي_not_in_msa(self, sw):
        assert sw.is_stopword("دلوقتي", Dialect.MSA) is False

    def test_egyptian_set_not_empty(self, sw):
        assert len(sw.get_egyptian()) > 5


class TestGulfStopWords:
    def test_وين_is_stopword(self, sw):
        assert sw.is_stopword("وين", Dialect.GULF) is True

    def test_gulf_set_not_empty(self, sw):
        assert len(sw.get_gulf()) > 5


class TestLevantineStopWords:
    def test_شو_is_stopword(self, sw):
        assert sw.is_stopword("شو", Dialect.LEVANTINE) is True

    def test_levantine_set_not_empty(self, sw):
        assert len(sw.get_levantine()) > 5


class TestUniversalSet:
    def test_universal_contains_all_dialects(self, sw):
        msa = sw.get_msa()
        egy = sw.get_egyptian()
        universal = sw.universal
        # Universal should contain words from both
        assert len(universal) >= len(msa)
        assert len(universal) >= len(egy)

    def test_is_stopword_any(self, sw):
        assert sw.is_stopword_any("في") is True
        assert sw.is_stopword_any("ايه") is True

    def test_universal_is_frozenset(self, sw):
        assert isinstance(sw.universal, frozenset)


class TestSupportedDialects:
    def test_supported_dialects_not_empty(self, sw):
        dialects = sw.supported_dialects()
        assert len(dialects) >= 4

    def test_msa_in_supported(self, sw):
        assert Dialect.MSA in sw.supported_dialects()

    def test_egyptian_in_supported(self, sw):
        assert Dialect.EGYPTIAN in sw.supported_dialects()


class TestUnsupportedDialect:
    def test_unsupported_raises(self, sw):
        with pytest.raises(UnsupportedDialectError):
            sw.get(Dialect.YEMENI)  # Not yet in lookup


class TestFrozenness:
    def test_stopword_set_is_immutable(self, sw):
        msa = sw.get_msa()
        with pytest.raises(AttributeError):
            msa.add("something")  # type: ignore


class TestRepr:
    def test_repr_shows_sizes(self, sw):
        r = repr(sw)
        assert "StopWords" in r
        assert "msa" in r
