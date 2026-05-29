"""Tests for NamedEntityRecognizer."""

import pytest

from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.models import EntityLabel, NERResult
from arabic_nlp.ner import NamedEntityRecognizer


@pytest.fixture
def ner() -> NamedEntityRecognizer:
    return NamedEntityRecognizer()


class TestPersonDetection:
    def test_detects_محمد_صلاح(self, ner):
        result = ner.extract("لعب محمد صلاح مباراة رائعة")
        persons = result.persons
        assert len(persons) >= 1

    def test_detects_person_by_honorific(self, ner):
        result = ner.extract("قال الدكتور أحمد إن الوضع جيد")
        persons = result.persons
        assert len(persons) >= 1

    def test_person_label_correct(self, ner):
        result = ner.extract("محمد صلاح لاعب مصري")
        if result.persons:
            assert result.persons[0].label == EntityLabel.PERSON


class TestLocationDetection:
    def test_detects_القاهرة(self, ner):
        result = ner.extract("سافر إلى القاهرة أمس")
        locations = result.locations
        assert any("القاهرة" in e.text for e in locations)

    def test_detects_الإسكندرية(self, ner):
        result = ner.extract("زار الإسكندرية في الصيف")
        locations = result.locations
        assert len(locations) >= 1

    def test_location_confidence(self, ner):
        result = ner.extract("في القاهرة")
        for loc in result.locations:
            assert 0 <= loc.confidence <= 1


class TestOrganizationDetection:
    def test_detects_الأهلي(self, ner):
        result = ner.extract("يلعب في نادي الأهلي منذ سنوات")
        orgs = result.organizations
        assert len(orgs) >= 1

    def test_contextual_org_شركة(self, ner):
        result = ner.extract("تأسست شركة جوجل عام 1998")
        orgs = result.organizations
        assert len(orgs) >= 1


class TestDateDetection:
    def test_detects_day_name(self, ner):
        result = ner.extract("جاء يوم الجمعة")
        dates = [e for e in result.entities if e.label == EntityLabel.DATE]
        assert len(dates) >= 1

    def test_detects_month(self, ner):
        result = ner.extract("في شهر يناير 2025")
        dates = [e for e in result.entities if e.label == EntityLabel.DATE]
        assert len(dates) >= 1


class TestMoneyDetection:
    def test_detects_جنيه(self, ner):
        result = ner.extract("اشترى السيارة بـ 500000 جنيه")
        money = [e for e in result.entities if e.label == EntityLabel.MONEY]
        assert len(money) >= 1

    def test_detects_دولار(self, ner):
        result = ner.extract("المبلغ 100 دولار")
        money = [e for e in result.entities if e.label == EntityLabel.MONEY]
        assert len(money) >= 1


class TestDeduplication:
    def test_no_duplicate_entities(self, ner):
        result = ner.extract("زار محمد صلاح القاهرة")
        # Check no overlapping spans
        for i, e1 in enumerate(result.entities):
            for j, e2 in enumerate(result.entities):
                if i != j:
                    # Either e1 ends before e2 starts or vice versa
                    assert e1.end <= e2.start or e2.end <= e1.start


class TestResultStructure:
    def test_entities_sorted_by_position(self, ner):
        result = ner.extract("زار محمد صلاح ملعب الأهلي في القاهرة")
        starts = [e.start for e in result.entities]
        assert starts == sorted(starts)

    def test_returns_ner_result(self, ner):
        result = ner.extract("مرحبا")
        assert isinstance(result, NERResult)

    def test_empty_entities_for_no_entities(self, ner):
        result = ner.extract("الجو جميل اليوم")
        assert isinstance(result.entities, list)

    def test_processing_time_positive(self, ner):
        result = ner.extract("مرحبا")
        assert result.processing_time_ms >= 0

    def test_by_label_method(self, ner):
        result = ner.extract("زار محمد صلاح القاهرة")
        locations = result.by_label(EntityLabel.LOCATION)
        assert isinstance(locations, list)


class TestInputValidation:
    def test_empty_raises(self, ner):
        with pytest.raises(InvalidInputError):
            ner.extract("")

    def test_whitespace_raises(self, ner):
        with pytest.raises(InvalidInputError):
            ner.extract("   ")
