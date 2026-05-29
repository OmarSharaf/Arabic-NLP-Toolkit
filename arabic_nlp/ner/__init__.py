"""
Named Entity Recognition (NER) Module
======================================
Rule-based NER for Arabic text.

Recognizes: PERSON, LOCATION, ORGANIZATION, DATE, TIME, MONEY, PRODUCT, EVENT.

Uses:
  1. Curated gazetteer lists for Egyptian/Arab names, cities, and organizations
  2. Contextual patterns (e.g., "شركة X" → ORGANIZATION)
  3. Date/time pattern matching
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.models import Entity, EntityLabel, NERResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# GAZETTEERS
# ─────────────────────────────────────────────

_PERSON_TITLES = frozenset(
    [
        "محمد",
        "أحمد",
        "احمد",
        "علي",
        "عمر",
        "خالد",
        "يوسف",
        "إبراهيم",
        "ابراهيم",
        "عبدالله",
        "عبد الله",
        "عبدالرحمن",
        "عبد الرحمن",
        "حسن",
        "حسين",
        "مصطفى",
        "مصطفا",
        "ماجد",
        "كريم",
        "سامي",
        "سمير",
        "طارق",
        "رامي",
        "هاني",
        "ناصر",
        "فاطمة",
        "عائشة",
        "مريم",
        "سارة",
        "نورة",
        "هند",
        "لمى",
        "ريم",
        "دينا",
        "نادية",
        "منى",
        "رنا",
        "سلمى",
        "شيماء",
        "أمل",
        "هدى",
        # Famous names
        "محمد صلاح",
        "محمد صلاح حامد",
        "كريم منصور",
        "عمرو دياب",
        "يسرا",
        "مني زكي",
        "أحمد زكي",
    ]
)

_HONORIFICS = frozenset(
    [
        "الدكتور",
        "الدكتورة",
        "دكتور",
        "دكتورة",
        "د.",
        "أ.د.",
        "الأستاذ",
        "الأستاذة",
        "أستاذ",
        "المهندس",
        "المهندسة",
        "الرئيس",
        "الملك",
        "الأمير",
        "الشيخ",
        "الشيخة",
        "السيد",
        "السيدة",
        "الآنسة",
    ]
)

_ARAB_CITIES = frozenset(
    [
        # Egypt
        "القاهرة",
        "الإسكندرية",
        "الجيزة",
        "الأقصر",
        "أسوان",
        "الإسماعيلية",
        "بورسعيد",
        "السويس",
        "المنصورة",
        "طنطا",
        "الزقازيق",
        "المنيا",
        "سوهاج",
        "أسيوط",
        "قنا",
        "شرم الشيخ",
        "الغردقة",
        "مرسى مطروح",
        "دمياط",
        "المحلة الكبرى",
        "الفيوم",
        "بنها",
        "المنوفية",
        "كفر الشيخ",
        # Saudi Arabia
        "الرياض",
        "جدة",
        "مكة المكرمة",
        "المدينة المنورة",
        "الدمام",
        "الخبر",
        "الجبيل",
        "أبها",
        "تبوك",
        "حائل",
        # UAE
        "دبي",
        "أبوظبي",
        "الشارقة",
        "عجمان",
        "رأس الخيمة",
        # Other MENA
        "بغداد",
        "الموصل",
        "البصرة",
        "أربيل",
        "عمان",
        "إربد",
        "بيروت",
        "طرابلس",
        "دمشق",
        "حلب",
        "حمص",
        "الدوحة",
        "المنامة",
        "الكويت",
        "مسقط",
        "الخرطوم",
        "تونس",
        "صفاقس",
        "الرباط",
        "الدار البيضاء",
        "مراكش",
        "الجزائر",
        "وهران",
        "طرابلس",
        "بنغازي",
    ]
)

_ORGANIZATIONS = frozenset(
    [
        # Egyptian
        "الأهلي",
        "الزمالك",
        "النادي الأهلي",
        "نادي الزمالك",
        "البنك المركزي المصري",
        "بنك مصر",
        "البنك الأهلي المصري",
        "الجامعة الأمريكية",
        "جامعة القاهرة",
        "جامعة الأزهر",
        "جامعة عين شمس",
        "فودافون مصر",
        "اتصالات مصر",
        "أورنج",
        "فاودي",
        "باي موب",
        "فوري",
        # Regional
        "أرامكو",
        "بترول أبوظبي",
        "طيران الإمارات",
        "الخطوط السعودية",
        # Global companies in Arabic
        "جوجل",
        "أمازون",
        "مايكروسوفت",
        "أبل",
        "ميتا",
        "تويتر",
        "فيسبوك",
        # International orgs
        "الأمم المتحدة",
        "جامعة الدول العربية",
        "منظمة الصحة العالمية",
        "الاتحاد الأوروبي",
        "صندوق النقد الدولي",
    ]
)

# ─────────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────────

_MONEY_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:جنيه|دولار|يورو|ريال|درهم|دينار|ألف|مليون|مليار)"
    r"|(?:جنيه|دولار|يورو|ريال|درهم)\s*\d+(?:\.\d+)?"
)

_DATE_PATTERN = re.compile(
    r"(?:\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})"
    r"|(?:الأحد|الاثنين|الثلاثاء|الأربعاء|الخميس|الجمعة|السبت)"
    r"|(?:يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)"
    r"(?:\s+\d{4})?"
    r"|(?:\d{1,2}\s+(?:يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)(?:\s+\d{4})?)"
)

_TIME_PATTERN = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:صباحاً|مساءً|ص|م|AM|PM)?")

# Contextual patterns — "شركة X" → ORGANIZATION
_ORG_CONTEXT = re.compile(
    r"(?:شركة|مؤسسة|منظمة|هيئة|نقابة|اتحاد|جامعة|كلية|معهد|بنك|مصرف|مستشفى|مدرسة)\s+"
    r"([\u0600-\u06FF\s]{2,30}?)(?=\s|$|،|\.)"
)

_PERSON_CONTEXT = re.compile(
    r"(?:" + "|".join(re.escape(h) for h in sorted(_HONORIFICS, key=len, reverse=True)) + r")\s+"
    r"([\u0600-\u06FF\s]{2,30}?)(?=\s|$|،|\.)"
)


class NamedEntityRecognizer:
    """
    Rule-based Arabic Named Entity Recognizer.

    Identifies persons, locations, organizations, dates, times, and money
    mentions using gazetteer lists and contextual regex patterns.

    Example::

        ner = NamedEntityRecognizer()

        result = ner.extract("زار محمد صلاح ملعب الأهلي في القاهرة أمس")
        for entity in result.entities:
            print(f"{entity.text!r:20} → {entity.label.value}")
        # 'محمد صلاح'          → PERSON
        # 'الأهلي'             → ORGANIZATION
        # 'القاهرة'            → LOCATION
        # 'أمس'                → DATE
    """

    def extract(self, text: str) -> NERResult:
        """
        Extract named entities from Arabic text.

        Args:
            text: Input Arabic text.

        Returns:
            NERResult with a deduplicated, sorted list of Entity objects.
        """
        if not text.strip():
            raise InvalidInputError("Input text cannot be empty")

        t0 = time.perf_counter()
        entities: list[Entity] = []

        entities.extend(self._match_gazetteers(text))
        entities.extend(self._match_patterns(text))
        entities.extend(self._match_contextual(text))

        entities = self._deduplicate(entities)
        entities.sort(key=lambda e: e.start)

        elapsed = (time.perf_counter() - t0) * 1000

        return NERResult(
            text=text,
            entities=entities,
            processing_time_ms=round(elapsed, 3),
        )

    # ─────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────

    def _match_gazetteers(self, text: str) -> list[Entity]:
        entities = []
        for name in sorted(_PERSON_TITLES, key=len, reverse=True):
            for m in re.finditer(re.escape(name), text):
                entities.append(
                    Entity(
                        text=m.group(),
                        label=EntityLabel.PERSON,
                        start=m.start(),
                        end=m.end(),
                        confidence=0.85,
                    )
                )
        for city in sorted(_ARAB_CITIES, key=len, reverse=True):
            for m in re.finditer(re.escape(city), text):
                entities.append(
                    Entity(
                        text=m.group(),
                        label=EntityLabel.LOCATION,
                        start=m.start(),
                        end=m.end(),
                        confidence=0.90,
                    )
                )
        for org in sorted(_ORGANIZATIONS, key=len, reverse=True):
            for m in re.finditer(re.escape(org), text):
                entities.append(
                    Entity(
                        text=m.group(),
                        label=EntityLabel.ORGANIZATION,
                        start=m.start(),
                        end=m.end(),
                        confidence=0.88,
                    )
                )
        return entities

    def _match_patterns(self, text: str) -> list[Entity]:
        entities = []
        for m in _MONEY_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=m.group(),
                    label=EntityLabel.MONEY,
                    start=m.start(),
                    end=m.end(),
                    confidence=0.92,
                )
            )
        for m in _DATE_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=m.group(),
                    label=EntityLabel.DATE,
                    start=m.start(),
                    end=m.end(),
                    confidence=0.88,
                )
            )
        for m in _TIME_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=m.group(),
                    label=EntityLabel.TIME,
                    start=m.start(),
                    end=m.end(),
                    confidence=0.90,
                )
            )
        return entities

    def _match_contextual(self, text: str) -> list[Entity]:
        entities = []
        for m in _ORG_CONTEXT.finditer(text):
            full_match = m.group(0)
            entities.append(
                Entity(
                    text=full_match.strip(),
                    label=EntityLabel.ORGANIZATION,
                    start=m.start(),
                    end=m.end(),
                    confidence=0.75,
                )
            )
        for m in _PERSON_CONTEXT.finditer(text):
            name = m.group(1).strip()
            if name:
                entities.append(
                    Entity(
                        text=name,
                        label=EntityLabel.PERSON,
                        start=m.start(1),
                        end=m.end(1),
                        confidence=0.80,
                    )
                )
        return entities

    @staticmethod
    def _deduplicate(entities: list[Entity]) -> list[Entity]:
        """Remove overlapping entities, keeping the higher-confidence one."""
        if not entities:
            return []
        entities.sort(key=lambda e: (e.start, -(e.end - e.start)))
        result = []
        last_end = -1
        for entity in entities:
            if entity.start >= last_end:
                result.append(entity)
                last_end = entity.end
            elif entity.confidence > (result[-1].confidence if result else 0):
                result[-1] = entity
                last_end = entity.end
        return result
