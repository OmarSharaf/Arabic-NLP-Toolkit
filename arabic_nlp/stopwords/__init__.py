"""
Stop Words Module
=================
Provides dialect-aware Arabic stop word sets.
All sets are frozen (immutable) for thread-safe access and fast membership testing.
"""

from __future__ import annotations

import logging

from arabic_nlp.exceptions import UnsupportedDialectError
from arabic_nlp.models import Dialect

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# STOP WORD SETS
# ─────────────────────────────────────────────

_MSA_STOPWORDS: frozenset[str] = frozenset(
    [
        # Pronouns
        "أنا",
        "نحن",
        "أنت",
        "أنتِ",
        "أنتم",
        "أنتن",
        "هو",
        "هي",
        "هم",
        "هن",
        "هما",
        # Articles & prepositions
        "في",
        "من",
        "إلى",
        "على",
        "عن",
        "مع",
        "إلى",
        "حتى",
        "خلال",
        "بين",
        "بعد",
        "قبل",
        "تحت",
        "فوق",
        "داخل",
        "خارج",
        "حول",
        "أمام",
        "وراء",
        "بدون",
        # Conjunctions
        "و",
        "أو",
        "لكن",
        "بينما",
        "إذا",
        "لو",
        "إن",
        "أن",
        "كي",
        "لأن",
        "لذلك",
        "لذا",
        "ثم",
        "سواء",
        "حيث",
        "بما",
        "مما",
        # Common verbs
        "كان",
        "يكون",
        "تكون",
        "كانت",
        "كانوا",
        "ليس",
        "ليست",
        "يوجد",
        "توجد",
        "قال",
        "يقول",
        "أراد",
        "يريد",
        "تريد",
        "يمكن",
        "يجب",
        "ينبغي",
        # Demonstratives
        "هذا",
        "هذه",
        "هؤلاء",
        "ذلك",
        "تلك",
        "أولئك",
        # Relative pronouns
        "الذي",
        "التي",
        "الذين",
        "اللواتي",
        "اللائي",
        # Question words
        "ما",
        "من",
        "متى",
        "أين",
        "كيف",
        "لماذا",
        "هل",
        "أي",
        "كم",
        # Other common
        "قد",
        "لقد",
        "إذ",
        "عند",
        "لدى",
        "أي",
        "كل",
        "جميع",
        "بعض",
        "كلا",
        "كلتا",
        "أكثر",
        "أقل",
        "جدا",
        "جداً",
        "الآن",
        "هنا",
        "هناك",
        "دائما",
        "أحيانا",
        "ربما",
        "فقط",
        "أيضا",
        "أيضاً",
        "مثل",
        "مثلا",
    ]
)

_EGYPTIAN_STOPWORDS: frozenset[str] = frozenset(
    [
        # Pronouns (Egyptian)
        "انا",
        "احنا",
        "انت",
        "انتي",
        "انتو",
        "هو",
        "هي",
        "هما",
        "هم",
        # Prepositions (Egyptian)
        "في",
        "من",
        "على",
        "عن",
        "مع",
        "لحد",
        "عند",
        "بعد",
        "قبل",
        "فوق",
        "تحت",
        "جنب",
        "دخل",
        "برا",
        # Conjunctions (Egyptian)
        "و",
        "أو",
        "او",
        "بس",
        "لكن",
        "وإن",
        "لو",
        "اذا",
        "عشان",
        "لأن",
        "علشان",
        "فـ",
        "مع إن",
        # Common verbs (Egyptian)
        "كان",
        "بقى",
        "بقا",
        "عاوز",
        "عايز",
        "عايزة",
        "ممكن",
        "لازم",
        "قال",
        "يقول",
        "اتكلم",
        # Demonstratives (Egyptian)
        "ده",
        "دي",
        "دول",
        "دا",
        "هو ده",
        "هي دي",
        # Question words (Egyptian)
        "ايه",
        "مين",
        "امتى",
        "فين",
        "ازاي",
        "ليه",
        # Other common (Egyptian)
        "مش",
        "ما",
        "مع",
        "زي",
        "كده",
        "هنا",
        "هناك",
        "هناك",
        "بقى",
        "يعني",
        "طب",
        "يلا",
        "تمام",
        "اوك",
        "اوكي",
    ]
)

_GULF_STOPWORDS: frozenset[str] = frozenset(
    [
        "انا",
        "احنا",
        "انت",
        "انتي",
        "انتم",
        "هو",
        "هي",
        "هم",
        "في",
        "من",
        "على",
        "عن",
        "مع",
        "لين",
        "عند",
        "بعد",
        "قبل",
        "و",
        "أو",
        "بس",
        "لكن",
        "لو",
        "اذا",
        "عشان",
        "لأن",
        "كان",
        "يكون",
        "ابغى",
        "ابي",
        "لازم",
        "ممكن",
        "هذا",
        "هذي",
        "هذيل",
        "ذاك",
        "وش",
        "وين",
        "متى",
        "كيف",
        "ليش",
        "ما",
        "مو",
        "مب",
        "بدون",
        "زين",
        "الحين",
    ]
)

_LEVANTINE_STOPWORDS: frozenset[str] = frozenset(
    [
        "انا",
        "نحنا",
        "انت",
        "انتي",
        "انتو",
        "هو",
        "هي",
        "هني",
        "في",
        "من",
        "على",
        "عن",
        "مع",
        "لعند",
        "عند",
        "بعد",
        "قبل",
        "و",
        "أو",
        "بس",
        "لكن",
        "لو",
        "اذا",
        "لأنو",
        "لأنها",
        "كان",
        "رح",
        "لازم",
        "ممكن",
        "بدي",
        "بده",
        "هاد",
        "هاي",
        "هدول",
        "شو",
        "مين",
        "لما",
        "وين",
        "كيف",
        "ليش",
        "ما",
        "مش",
        "مو",
        "هلق",
        "هون",
        "هناك",
    ]
)

_ALL_STOPWORDS: dict[Dialect, frozenset[str]] = {
    Dialect.MSA: _MSA_STOPWORDS,
    Dialect.EGYPTIAN: _EGYPTIAN_STOPWORDS,
    Dialect.GULF: _GULF_STOPWORDS,
    Dialect.LEVANTINE: _LEVANTINE_STOPWORDS,
}

# Union of all stop word sets — useful for pan-Arabic preprocessing
_UNIVERSAL_STOPWORDS: frozenset[str] = frozenset(
    word for sw_set in _ALL_STOPWORDS.values() for word in sw_set
)


class StopWords:
    """
    Dialect-aware Arabic stop word manager.

    Example::

        sw = StopWords()

        # Get MSA stop words
        msa = sw.get(Dialect.MSA)
        print("في" in msa)   # True

        # Get Egyptian stop words
        egy = sw.get(Dialect.EGYPTIAN)
        print("ايه" in egy)  # True

        # Check a word
        sw.is_stopword("انا", Dialect.EGYPTIAN)  # True

        # Get universal set (all dialects combined)
        universal = sw.universal
    """

    def get(self, dialect: Dialect = Dialect.MSA) -> frozenset[str]:
        """Return the stop-word set for a specific dialect."""
        if dialect not in _ALL_STOPWORDS:
            supported = [d.value for d in _ALL_STOPWORDS]
            raise UnsupportedDialectError(dialect.value, supported)
        return _ALL_STOPWORDS[dialect]

    def get_msa(self) -> frozenset[str]:
        """Return MSA stop words."""
        return _MSA_STOPWORDS

    def get_egyptian(self) -> frozenset[str]:
        """Return Egyptian dialect stop words."""
        return _EGYPTIAN_STOPWORDS

    def get_gulf(self) -> frozenset[str]:
        """Return Gulf dialect stop words."""
        return _GULF_STOPWORDS

    def get_levantine(self) -> frozenset[str]:
        """Return Levantine dialect stop words."""
        return _LEVANTINE_STOPWORDS

    @property
    def universal(self) -> frozenset[str]:
        """Union of all dialect stop-word sets."""
        return _UNIVERSAL_STOPWORDS

    def is_stopword(self, word: str, dialect: Dialect = Dialect.MSA) -> bool:
        """Check whether a word is a stop word for the given dialect."""
        return word in self.get(dialect)

    def is_stopword_any(self, word: str) -> bool:
        """Check whether a word is a stop word in any dialect."""
        return word in _UNIVERSAL_STOPWORDS

    def supported_dialects(self) -> list[Dialect]:
        """Return list of dialects with stop-word support."""
        return list(_ALL_STOPWORDS.keys())

    def __repr__(self) -> str:
        sizes = {d.value: len(sw) for d, sw in _ALL_STOPWORDS.items()}
        return f"StopWords(dialects={sizes})"
