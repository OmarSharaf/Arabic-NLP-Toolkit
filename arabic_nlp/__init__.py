"""
arabic-nlp-toolkit
==================
A comprehensive, production-ready Python library for Arabic Natural Language Processing.

Supports Modern Standard Arabic (فصحى) and all major dialects:
Egyptian, Gulf, Levantine, Maghrebi, Iraqi, Yemeni, Sudanese.

Quick start::

    from arabic_nlp import ArabicNLP

    nlp = ArabicNLP()

    # Dialect detection
    result = nlp.detect_dialect("ازيك عامل ايه النهارده")
    print(result.dialect)    # Dialect.EGYPTIAN
    print(result.confidence) # 0.94

    # Sentiment analysis
    result = nlp.sentiment("المنتج ده رائع جداً وبحبه أوي")
    print(result.label)  # SentimentLabel.POSITIVE

    # Full pipeline
    analysis = nlp.analyze("زيارة محمد صلاح لملعب الأهلي في القاهرة كانت رائعة")
    print(analysis["dialect"].dialect)       # Dialect.MSA
    print(analysis["sentiment"].label)       # SentimentLabel.POSITIVE
    print(len(analysis["entities"].entities)) # 3

Full documentation: https://arabic-nlp-toolkit.readthedocs.io
"""

from arabic_nlp._version import __version__, __version_info__
from arabic_nlp.core import ArabicNLP
from arabic_nlp.dialects import DialectDetector
from arabic_nlp.exceptions import (
    ArabicNLPError,
    InvalidInputError,
    ModelNotFoundError,
    UnsupportedDialectError,
    UnsupportedLanguageError,
)
from arabic_nlp.keywords import Keyword, KeywordExtractor, KeywordResult
from arabic_nlp.models import (
    ArabicNLPConfig,
    BatchResult,
    Definiteness,
    Dialect,
    DialectResult,
    DocumentAnalysis,
    Entity,
    EntityLabel,
    Gender,
    NERResult,
    NormalizationResult,
    Number,
    POSTag,
    Script,
    SentimentLabel,
    SentimentResult,
    Token,
    TokenType,
    TransliterationResult,
    VerbTense,
    WordClass,
)
from arabic_nlp.morphology import ArabicMorphAnalyzer
from arabic_nlp.ner import NamedEntityRecognizer
from arabic_nlp.normalization import ArabicNormalizer
from arabic_nlp.pos import ArabicPOSTagger
from arabic_nlp.profiling import TextProfile, TextProfiler, TextRegister
from arabic_nlp.sentiment import SentimentAnalyzer
from arabic_nlp.stemmer import ArabicStemmer
from arabic_nlp.stopwords import StopWords
from arabic_nlp.tokenizer import ArabicTokenizer
from arabic_nlp.transliteration import Transliterator

__all__ = [
    # Core façade
    "ArabicNLP",
    # Individual modules
    "DialectDetector",
    "SentimentAnalyzer",
    "ArabicTokenizer",
    "NamedEntityRecognizer",
    "ArabicNormalizer",
    "Transliterator",
    "StopWords",
    "ArabicMorphAnalyzer",
    "ArabicPOSTagger",
    "ArabicStemmer",
    "KeywordExtractor",
    "TextProfiler",
    # Enums
    "Dialect",
    "SentimentLabel",
    "EntityLabel",
    "POSTag",
    "TokenType",
    "Script",
    "WordClass",
    "Gender",
    "Number",
    "Definiteness",
    "VerbTense",
    # Result models
    "DialectResult",
    "SentimentResult",
    "Token",
    "Entity",
    "NERResult",
    "NormalizationResult",
    "TransliterationResult",
    "BatchResult",
    "ArabicNLPConfig",
    "DocumentAnalysis",
    "Keyword",
    "KeywordResult",
    "TextProfile",
    "TextRegister",
    # Exceptions
    "ArabicNLPError",
    "ModelNotFoundError",
    "InvalidInputError",
    "UnsupportedDialectError",
    "UnsupportedLanguageError",
    # Version
    "__version__",
    "__version_info__",
]

__author__ = "Omar S. M. Abdelfatah"
__email__ = "omar@omarsharaf.me"
__license__ = "MIT"
__url__ = "https://github.com/OmarSharaf/arabic-nlp-toolkit"
