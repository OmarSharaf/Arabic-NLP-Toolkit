"""Tests for ArabicPOSTagger."""

import pytest

from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.pos import ArabicPOSTagger, POSTag


@pytest.fixture
def tagger() -> ArabicPOSTagger:
    return ArabicPOSTagger()


class TestSingleTokenTagging:
    def test_preposition_في(self, tagger):
        result = tagger.tag("في")
        assert result.tag == POSTag.ADP

    def test_preposition_من(self, tagger):
        result = tagger.tag("من")
        assert result.tag == POSTag.ADP

    def test_coordinating_conjunction_و(self, tagger):
        result = tagger.tag("و")
        assert result.tag == POSTag.CCONJ

    def test_personal_pronoun(self, tagger):
        result = tagger.tag("أنا")
        assert result.tag == POSTag.PRON

    def test_relative_pronoun(self, tagger):
        result = tagger.tag("الذي")
        assert result.tag == POSTag.PRON_REL

    def test_demonstrative_pronoun(self, tagger):
        result = tagger.tag("هذا")
        assert result.tag == POSTag.PRON_DEM

    def test_negation_particle(self, tagger):
        result = tagger.tag("لا")
        assert result.tag == POSTag.PART_NEG

    def test_interrogative_particle(self, tagger):
        result = tagger.tag("هل")
        assert result.tag == POSTag.PART_INTER

    def test_number(self, tagger):
        result = tagger.tag("123")
        assert result.tag == POSTag.NUM

    def test_arabic_number(self, tagger):
        result = tagger.tag("٢٠٢٥")
        assert result.tag == POSTag.NUM

    def test_interjection(self, tagger):
        result = tagger.tag("برافو")
        assert result.tag == POSTag.INTJ

    def test_adverb_جدا(self, tagger):
        result = tagger.tag("جداً")
        assert result.tag == POSTag.ADV


class TestSentenceTagging:
    def test_basic_sentence(self, tagger):
        result = tagger.tag_sentence("ذهب الولد إلى المدرسة")
        assert len(result.tags) >= 3

    def test_has_verb(self, tagger):
        result = tagger.tag_sentence("يكتب الطالب الدرس")
        verbs = result.verbs
        assert len(verbs) >= 1

    def test_has_nouns(self, tagger):
        result = tagger.tag_sentence("الكتاب على الطاولة")
        nouns = result.nouns
        assert len(nouns) >= 1

    def test_to_tuples(self, tagger):
        result = tagger.tag_sentence("ذهب الولد")
        tuples = result.to_tuples()
        assert all(isinstance(t, tuple) and len(t) == 2 for t in tuples)

    def test_processing_time_positive(self, tagger):
        result = tagger.tag_sentence("مرحبا بالعالم")
        assert result.processing_time_ms >= 0

    def test_tokens_preserved(self, tagger):
        text = "مرحبا بكم"
        result = tagger.tag_sentence(text)
        assert len(result.tokens) == len(result.tags)


class TestTaggingList:
    def test_tag_list(self, tagger):
        tokens = ["الكتاب", "على", "الطاولة"]
        results = tagger.tag_tokens(tokens)
        assert len(results) == 3

    def test_filter_by_tag(self, tagger):
        result = tagger.tag_sentence("في البيت و في المدرسة")
        adpositions = result.filter_by_tag(POSTag.ADP)
        assert len(adpositions) >= 1


class TestConfidenceScores:
    def test_known_word_high_confidence(self, tagger):
        result = tagger.tag("في")
        assert result.confidence > 0.9

    def test_confidence_in_range(self, tagger):
        result = tagger.tag("مرحبا")
        assert 0.0 <= result.confidence <= 1.0

    def test_fine_tag_not_empty(self, tagger):
        result = tagger.tag("في")
        assert result.fine_tag


class TestInputValidation:
    def test_empty_raises(self, tagger):
        with pytest.raises(InvalidInputError):
            tagger.tag("")

    def test_empty_sentence_raises(self, tagger):
        with pytest.raises(InvalidInputError):
            tagger.tag_sentence("")
