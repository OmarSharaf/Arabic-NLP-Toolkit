"""
Core ArabicNLP façade.

Provides a single unified entry point for all NLP operations.
All individual modules are lazily instantiated on first use to
minimize import time and memory overhead.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from arabic_nlp.exceptions import InvalidInputError
from arabic_nlp.models import (
    ArabicNLPConfig,
    BatchResult,
    Dialect,
    DialectResult,
    DocumentAnalysis,
    NERResult,
    NormalizationResult,
    Script,
    SentimentResult,
    Token,
    TransliterationResult,
)

if TYPE_CHECKING:
    from arabic_nlp.dialects import DialectDetector
    from arabic_nlp.keywords import KeywordExtractor, KeywordResult
    from arabic_nlp.morphology import ArabicMorphAnalyzer
    from arabic_nlp.ner import NamedEntityRecognizer
    from arabic_nlp.normalization import ArabicNormalizer
    from arabic_nlp.pos import ArabicPOSTagger
    from arabic_nlp.profiling import TextProfile, TextProfiler
    from arabic_nlp.sentiment import SentimentAnalyzer
    from arabic_nlp.stemmer import ArabicStemmer
    from arabic_nlp.stopwords import StopWords
    from arabic_nlp.tokenizer import ArabicTokenizer
    from arabic_nlp.transliteration import Transliterator

logger = logging.getLogger(__name__)


class ArabicNLP:
    """
    Unified façade for all Arabic NLP operations.

    All sub-modules are lazily loaded on first access.

    Example::

        from arabic_nlp import ArabicNLP

        nlp = ArabicNLP()

        # Detect dialect
        result = nlp.detect_dialect("ازيك عامل ايه النهارده")
        print(result.dialect)     # Dialect.EGYPTIAN
        print(result.confidence)  # 0.93

        # Analyze sentiment
        sentiment = nlp.sentiment("المنتج ده تحفة")
        print(sentiment.label)    # SentimentLabel.POSITIVE

        # Tokenize
        tokens = nlp.tokenize("مرحبا بالعالم")
        print([t.text for t in tokens])

        # Named entities
        ner = nlp.ner("زار محمد صلاح القاهرة أمس")
        for entity in ner.entities:
            print(entity.text, entity.label)

        # Normalize
        result = nlp.normalize("هَذَا نَصٌّ مُشَكَّل")
        print(result.normalized)

        # Transliterate
        result = nlp.transliterate("مرحبا", target=Script.FRANCO)
        print(result.transliterated)  # "marhaba"

        # Pipeline — run all operations at once
        analysis = nlp.analyze("زيارة محمد صلاح كانت رائعة جداً")
    """

    def __init__(
        self,
        *,
        lazy: bool = True,
        log_level: int = logging.WARNING,
        config: ArabicNLPConfig | None = None,
    ) -> None:
        """
        Initialize ArabicNLP.

        Args:
            lazy: If True (default), sub-modules are loaded on first use.
                  Set to False to load everything upfront.
            log_level: Python logging level for the library logger.
            config: Optional :class:`ArabicNLPConfig` for pipeline defaults.
        """
        logging.getLogger("arabic_nlp").setLevel(log_level)
        self.config = config or ArabicNLPConfig()

        self._dialect_detector: DialectDetector | None = None
        self._sentiment_analyzer: SentimentAnalyzer | None = None
        self._tokenizer: ArabicTokenizer | None = None
        self._ner: NamedEntityRecognizer | None = None
        self._normalizer: ArabicNormalizer | None = None
        self._transliterator: Transliterator | None = None
        self._stopwords: StopWords | None = None
        self._keyword_extractor: KeywordExtractor | None = None
        self._text_profiler: TextProfiler | None = None
        self._morph_analyzer: ArabicMorphAnalyzer | None = None
        self._pos_tagger: ArabicPOSTagger | None = None
        self._stemmer: ArabicStemmer | None = None

        if not lazy:
            _ = (
                self.dialect_detector,
                self.sentiment_analyzer,
                self.tokenizer,
                self.ner,
                self.normalizer,
                self.transliterator,
                self.stopwords,
            )

    # ─────────────────────────────────────────────
    # LAZY PROPERTIES
    # ─────────────────────────────────────────────

    @property
    def dialect_detector(self) -> DialectDetector:
        if self._dialect_detector is None:
            from arabic_nlp.dialects import DialectDetector

            self._dialect_detector = DialectDetector()
        return self._dialect_detector

    @property
    def sentiment_analyzer(self) -> SentimentAnalyzer:
        if self._sentiment_analyzer is None:
            from arabic_nlp.sentiment import SentimentAnalyzer

            self._sentiment_analyzer = SentimentAnalyzer()
        return self._sentiment_analyzer

    @property
    def tokenizer(self) -> ArabicTokenizer:
        if self._tokenizer is None:
            from arabic_nlp.tokenizer import ArabicTokenizer

            self._tokenizer = ArabicTokenizer()
        return self._tokenizer

    @property
    def ner(self) -> NamedEntityRecognizer:
        if self._ner is None:
            from arabic_nlp.ner import NamedEntityRecognizer

            self._ner = NamedEntityRecognizer()
        return self._ner

    @property
    def normalizer(self) -> ArabicNormalizer:
        if self._normalizer is None:
            from arabic_nlp.normalization import ArabicNormalizer

            self._normalizer = ArabicNormalizer()
        return self._normalizer

    @property
    def transliterator(self) -> Transliterator:
        if self._transliterator is None:
            from arabic_nlp.transliteration import Transliterator

            self._transliterator = Transliterator()
        return self._transliterator

    @property
    def stopwords(self) -> StopWords:
        if self._stopwords is None:
            from arabic_nlp.stopwords import StopWords

            self._stopwords = StopWords()
        return self._stopwords

    @property
    def keyword_extractor(self) -> KeywordExtractor:
        if self._keyword_extractor is None:
            from arabic_nlp.keywords import KeywordExtractor

            self._keyword_extractor = KeywordExtractor(
                use_stemming=self.config.keyword_use_stemming,
            )
        return self._keyword_extractor

    @property
    def text_profiler(self) -> TextProfiler:
        if self._text_profiler is None:
            from arabic_nlp.profiling import TextProfiler

            self._text_profiler = TextProfiler()
        return self._text_profiler

    # ─────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────

    def detect_dialect(self, text: str) -> DialectResult:
        """
        Detect the Arabic dialect of the input text.

        Args:
            text: Arabic text to analyze.

        Returns:
            DialectResult with dialect, confidence, and all scores.

        Raises:
            InvalidInputError: If text is empty or None.

        Example::

            result = nlp.detect_dialect("ازيك عامل ايه النهارده")
            print(result.dialect)     # Dialect.EGYPTIAN
            print(result.confidence)  # 0.93
        """
        self._validate_input(text)
        return self.dialect_detector.detect(text)

    def sentiment(
        self,
        text: str,
        dialect: Dialect | None = None,
    ) -> SentimentResult:
        """
        Analyze the sentiment of Arabic text.

        Args:
            text: Arabic text to analyze.
            dialect: Optional hint for the dialect to improve accuracy.
                     Auto-detected if not provided.

        Returns:
            SentimentResult with label and confidence scores.

        Example::

            result = nlp.sentiment("المنتج ده تحفة وبحبه أوي")
            print(result.label)           # SentimentLabel.POSITIVE
            print(result.positive_score)  # 0.91
        """
        self._validate_input(text)
        return self.sentiment_analyzer.analyze(text, dialect=dialect)

    def tokenize(
        self,
        text: str,
        *,
        include_punctuation: bool = True,
        include_stopwords: bool = True,
        normalize: bool = False,
    ) -> list[Token]:
        """
        Tokenize Arabic text into a list of Token objects.

        Args:
            text: Arabic text to tokenize.
            include_punctuation: Whether to include punctuation tokens.
            include_stopwords: Whether to include stop-word tokens.
            normalize: Whether to normalize tokens during tokenization.

        Returns:
            List of Token objects with offsets and metadata.

        Example::

            tokens = nlp.tokenize("مرحباً بكم في مصر")
            print([t.text for t in tokens])
            # ["مرحباً", "بكم", "في", "مصر"]
        """
        self._validate_input(text)
        return self.tokenizer.tokenize(
            text,
            include_punctuation=include_punctuation,
            include_stopwords=include_stopwords,
            normalize=normalize,
        )

    def extract_entities(self, text: str) -> NERResult:
        """
        Extract named entities from Arabic text.

        Args:
            text: Arabic text to analyze.

        Returns:
            NERResult containing a list of Entity objects.

        Example::

            result = nlp.extract_entities("زار محمد صلاح القاهرة أمس")
            for entity in result.entities:
                print(entity.text, entity.label)
            # محمد صلاح → PERSON
            # القاهرة  → LOCATION
        """
        self._validate_input(text)
        return self.ner.extract(text)

    def normalize(
        self,
        text: str,
        *,
        remove_diacritics: bool = True,
        normalize_alef: bool = True,
        normalize_teh_marbuta: bool = True,
        normalize_hamza: bool = True,
        remove_tatweel: bool = True,
        remove_extra_spaces: bool = True,
        remove_urls: bool = False,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        remove_emojis: bool = False,
    ) -> NormalizationResult:
        """
        Normalize Arabic text with fine-grained control.

        Args:
            text: Arabic text to normalize.
            remove_diacritics: Remove tashkeel/harakat (default True).
            normalize_alef: Normalize أ إ آ to ا (default True).
            normalize_teh_marbuta: Normalize ة to ه (default True).
            normalize_hamza: Normalize hamza variants (default True).
            remove_tatweel: Remove kashida/tatweel ـ (default True).
            remove_extra_spaces: Collapse multiple spaces (default True).
            remove_urls: Strip URLs from text (default False).
            remove_mentions: Strip @mentions (default False).
            remove_hashtags: Strip #hashtags (default False).
            remove_emojis: Strip emoji characters (default False).

        Returns:
            NormalizationResult with original, normalized text, and changes list.

        Example::

            result = nlp.normalize("هَذَا نَصٌّ بِالتَّشْكِيل")
            print(result.normalized)  # "هذا نص بالتشكيل"
            print(result.changes)     # ["removed_diacritics"]
        """
        self._validate_input(text)
        return self.normalizer.normalize(
            text,
            remove_diacritics=remove_diacritics,
            normalize_alef=normalize_alef,
            normalize_teh_marbuta=normalize_teh_marbuta,
            normalize_hamza=normalize_hamza,
            remove_tatweel=remove_tatweel,
            remove_extra_spaces=remove_extra_spaces,
            remove_urls=remove_urls,
            remove_mentions=remove_mentions,
            remove_hashtags=remove_hashtags,
            remove_emojis=remove_emojis,
        )

    def transliterate(
        self,
        text: str,
        *,
        source: Script = Script.ARABIC,
        target: Script = Script.FRANCO,
    ) -> TransliterationResult:
        """
        Transliterate between Arabic and other scripts.

        Args:
            text: Text to transliterate.
            source: Source script (default Arabic).
            target: Target script (default Franco-Arabic).

        Returns:
            TransliterationResult with original and transliterated text.

        Example::

            # Arabic → Franco
            result = nlp.transliterate("مرحبا كيف حالك")
            print(result.transliterated)  # "marhaba kif 7alek"

            # Franco → Arabic
            result = nlp.transliterate(
                "ana mesh 3aref",
                source=Script.FRANCO,
                target=Script.ARABIC,
            )
            print(result.transliterated)  # "انا مش عارف"
        """
        self._validate_input(text)
        return self.transliterator.transliterate(text, source=source, target=target)

    def get_stopwords(self, dialect: Dialect = Dialect.MSA) -> frozenset[str]:
        """
        Return the stop-word set for a dialect.

        Args:
            dialect: The dialect to retrieve stop words for.

        Returns:
            Frozenset of stop words.

        Example::

            sw = nlp.get_stopwords(Dialect.EGYPTIAN)
            print("في" in sw)  # True
        """
        return self.stopwords.get(dialect)

    def is_stopword(self, word: str, dialect: Dialect = Dialect.MSA) -> bool:
        """Check whether a word is a stop word for the given dialect."""
        return self.stopwords.is_stopword(word, dialect)

    def analyze(self, text: str) -> dict:
        """
        Run the full NLP pipeline on a single text.

        Runs dialect detection, sentiment, tokenization, NER, and normalization
        in one call and returns a unified dict.

        Args:
            text: Arabic text to analyze.

        Returns:
            Dictionary with keys: dialect, sentiment, tokens, entities, normalized.

        Example::

            analysis = nlp.analyze("زيارة محمد صلاح كانت رائعة جداً")
            print(analysis["dialect"].dialect)       # Dialect.MSA
            print(analysis["sentiment"].label)       # SentimentLabel.POSITIVE
            print(len(analysis["entities"].entities))  # 1
        """
        self._validate_input(text)
        t0 = time.perf_counter()

        dialect_result = self.detect_dialect(text)
        sentiment_result = self.sentiment(text, dialect=dialect_result.dialect)
        tokens = self.tokenize(text)
        entities = self.extract_entities(text)
        normalized = self.normalize(text)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("Full pipeline completed in %.1f ms", elapsed)

        return {
            "text": text,
            "dialect": dialect_result,
            "sentiment": sentiment_result,
            "tokens": tokens,
            "entities": entities,
            "normalized": normalized,
            "pipeline_time_ms": round(elapsed, 2),
        }

    def batch_sentiment(self, texts: list[str]) -> BatchResult:
        """
        Analyze sentiment for a list of texts efficiently.

        Args:
            texts: List of Arabic texts.

        Returns:
            BatchResult containing individual SentimentResult objects.
        """
        if not texts:
            raise InvalidInputError("texts list cannot be empty")

        t0 = time.perf_counter()
        results = []
        failed = 0

        for text in texts:
            try:
                self._validate_input(text)
                results.append(self.sentiment(text))
            except Exception as exc:
                logger.warning("Failed to analyze text: %s — %s", text[:50], exc)
                results.append(None)
                failed += 1

        elapsed = (time.perf_counter() - t0) * 1000

        return BatchResult(
            results=results,
            total=len(texts),
            successful=len(texts) - failed,
            failed=failed,
            total_processing_time_ms=round(elapsed, 2),
        )

    def batch_dialect(self, texts: list[str]) -> BatchResult:
        """Detect dialects for a list of texts efficiently."""
        if not texts:
            raise InvalidInputError("texts list cannot be empty")

        t0 = time.perf_counter()
        results = []
        failed = 0

        for text in texts:
            try:
                self._validate_input(text)
                results.append(self.detect_dialect(text))
            except Exception as exc:
                logger.warning("Failed to detect dialect: %s — %s", text[:50], exc)
                results.append(None)
                failed += 1

        elapsed = (time.perf_counter() - t0) * 1000

        return BatchResult(
            results=results,
            total=len(texts),
            successful=len(texts) - failed,
            failed=failed,
            total_processing_time_ms=round(elapsed, 2),
        )

    # ─────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────

    @staticmethod
    def _validate_input(text: str) -> None:
        if not isinstance(text, str):
            raise InvalidInputError(f"Input must be a string, got {type(text).__name__}")
        if not text.strip():
            raise InvalidInputError("Input text cannot be empty or whitespace")

    def __repr__(self) -> str:
        from arabic_nlp._version import __version__

        loaded = [
            name
            for name, attr in [
                ("dialect", self._dialect_detector),
                ("sentiment", self._sentiment_analyzer),
                ("tokenizer", self._tokenizer),
                ("ner", self._ner),
                ("normalizer", self._normalizer),
                ("transliterator", self._transliterator),
                ("stopwords", self._stopwords),
            ]
            if attr is not None
        ]
        return f"ArabicNLP(version={__version__!r}, loaded_modules={loaded})"

    @property
    def morph_analyzer(self) -> ArabicMorphAnalyzer:
        if self._morph_analyzer is None:
            from arabic_nlp.morphology import ArabicMorphAnalyzer

            self._morph_analyzer = ArabicMorphAnalyzer()
        return self._morph_analyzer

    @property
    def pos_tagger(self) -> ArabicPOSTagger:
        if self._pos_tagger is None:
            from arabic_nlp.pos import ArabicPOSTagger

            self._pos_tagger = ArabicPOSTagger()
        return self._pos_tagger

    @property
    def stemmer(self) -> ArabicStemmer:
        if self._stemmer is None:
            from arabic_nlp.stemmer import ArabicStemmer

            self._stemmer = ArabicStemmer()
        return self._stemmer

    def morphology(self, word: str):
        """
        Perform morphological analysis on a single Arabic word.

        Args:
            word: Single Arabic word.

        Returns:
            MorphResult with root, stem, pattern, gender, number, etc.

        Example::

            result = nlp.morphology("الكاتب")
            print(result.root)         # "كتب"
            print(result.definiteness) # Definiteness.DEFINITE
        """
        self._validate_input(word)
        return self.morph_analyzer.analyze(word)

    def tag_pos(self, text: str):
        """
        Tag all tokens in a sentence with POS labels.

        Args:
            text: Arabic sentence.

        Returns:
            POSSequenceResult with per-token POS tags.

        Example::

            result = nlp.tag_pos("ذهب الولد إلى المدرسة")
            for pos in result.tags:
                print(pos.token, pos.tag.value)
        """
        self._validate_input(text)
        return self.pos_tagger.tag_sentence(text)

    def stem(self, word: str, *, aggressive: bool = False) -> str:
        """
        Stem an Arabic word and return the stem string.

        Args:
            word: Arabic word to stem.
            aggressive: Use aggressive stemming (default False).

        Returns:
            Stem string.

        Example::

            nlp.stem("الكتاب")     # "كتاب"
            nlp.stem("يكتبون")    # "يكتب"
        """
        from arabic_nlp.stemmer import StemmerMode

        self._validate_input(word)
        mode = StemmerMode.AGGRESSIVE if aggressive else StemmerMode.LIGHT
        return self.stemmer.stem(word, mode=mode).stem

    def stem_text(self, text: str, *, aggressive: bool = False) -> list[str]:
        """
        Tokenize and stem all Arabic words in a text.

        Returns list of stem strings (useful for bag-of-words features).
        """
        from arabic_nlp.stemmer import StemmerMode

        self._validate_input(text)
        mode = StemmerMode.AGGRESSIVE if aggressive else StemmerMode.LIGHT
        return self.stemmer.stem_text(text, mode=mode)

    def get_text_stats(self, text: str):
        """
        Compute comprehensive statistics for an Arabic text.

        Returns:
            TextStatistics with word count, TTR, Arabic ratio, readability, etc.
        """
        from arabic_nlp.utils import get_statistics

        self._validate_input(text)
        return get_statistics(text)

    def get_readability(self, text: str):
        """Compute readability score for an Arabic text."""
        from arabic_nlp.utils import get_readability

        self._validate_input(text)
        return get_readability(text)

    def similarity(self, text1: str, text2: str):
        """
        Compute similarity between two Arabic texts.

        Returns SimilarityResult with Jaccard, cosine, and overlap scores.
        """
        from arabic_nlp.utils import compute_similarity

        self._validate_input(text1)
        self._validate_input(text2)
        return compute_similarity(text1, text2)

    def extract_keywords(
        self,
        text: str,
        *,
        top_n: int | None = None,
        dialect: Dialect | None = None,
    ) -> KeywordResult:
        """
        Extract ranked keywords from Arabic text.

        Args:
            text: Document or passage.
            top_n: Max keywords (defaults to ``config.keyword_top_n``).
            dialect: Stopword dialect hint (auto-detected if omitted).

        Example::

            result = nlp.extract_keywords(
                "الذكاء الاصطناعي يغير التعليم في العالم العربي",
                top_n=5,
            )
            print([k.text for k in result.keywords])
        """
        self._validate_input(text)
        if self.config.normalize_before_keywords:
            text = self.normalize(text).normalized
        return self.keyword_extractor.extract(
            text,
            top_n=top_n or self.config.keyword_top_n,
            dialect=dialect,
        )

    def profile(self, text: str) -> TextProfile:
        """
        Profile text register, quality, and social-media signals.

        Returns:
            :class:`TextProfile` with register, quality score, and recommendations.

        Example::

            p = nlp.profile("ازيك يا صاحبي 😂 #مصر")
            print(p.register)       # social_media
            print(p.quality_score)
        """
        self._validate_input(text)
        return self.text_profiler.profile(text)

    def content_words(
        self,
        text: str,
        *,
        dialect: Dialect | None = None,
    ) -> list[str]:
        """
        Return Arabic content words with stopwords removed.

        Useful for search indexing, topic modeling, and feature extraction.
        """
        self._validate_input(text)
        dialect = dialect or self.config.default_dialect
        sw = self.get_stopwords(dialect) | self.get_stopwords(Dialect.MSA)
        tokens = self.tokenizer.word_tokenize(text)
        return [t for t in tokens if t not in sw and len(t) > 1]

    def split_sentences(self, text: str) -> list[str]:
        """Split Arabic text into sentences."""
        from arabic_nlp.utils import sentence_split

        self._validate_input(text)
        return sentence_split(text)

    def analyze_document(
        self,
        text: str,
        *,
        include_keywords: bool | None = None,
        include_profile: bool | None = None,
    ) -> DocumentAnalysis:
        """
        Run the NLP pipeline and return a structured, serializable result.

        Ideal for REST APIs, ETL pipelines, and analytics dashboards.

        Example::

            doc = nlp.analyze_document("زيارة محمد صلاح للقاهرة كانت رائعة")
            print(doc.summary)
            print(doc.to_json())
        """
        from arabic_nlp._version import __version__

        self._validate_input(text)
        t0 = time.perf_counter()

        do_keywords = (
            include_keywords
            if include_keywords is not None
            else self.config.extract_keywords_on_analyze
        )
        do_profile = (
            include_profile if include_profile is not None else self.config.profile_on_analyze
        )

        dialect_result = self.detect_dialect(text)
        sentiment_result = self.sentiment(text, dialect=dialect_result.dialect)
        tokens = self.tokenize(text)
        entities = self.extract_entities(text)
        normalized = self.normalize(text)

        keywords_data: list[dict] = []
        if do_keywords:
            kw = self.extract_keywords(text, dialect=dialect_result.dialect)
            keywords_data = [k.model_dump() for k in kw.keywords]

        profile_data: dict | None = None
        if do_profile:
            profile_data = self.profile(text).model_dump()

        elapsed = (time.perf_counter() - t0) * 1000

        return DocumentAnalysis(
            text=text,
            dialect=dialect_result,
            sentiment=sentiment_result,
            entities=entities,
            normalized=normalized,
            token_count=len(tokens),
            entity_count=len(entities.entities),
            keywords=keywords_data,
            profile=profile_data,
            pipeline_time_ms=round(elapsed, 2),
            version=__version__,
        )

    def batch_analyze(
        self,
        texts: list[str],
        *,
        include_keywords: bool = False,
    ) -> BatchResult:
        """
        Analyze multiple documents through :meth:`analyze_document`.

        Returns:
            BatchResult with :class:`DocumentAnalysis` items (or None on failure).
        """
        if not texts:
            raise InvalidInputError("texts list cannot be empty")

        t0 = time.perf_counter()
        results: list = []
        failed = 0

        for text in texts:
            try:
                results.append(self.analyze_document(text, include_keywords=include_keywords))
            except Exception as exc:
                logger.warning("Failed document analysis: %s — %s", text[:50], exc)
                results.append(None)
                failed += 1

        elapsed = (time.perf_counter() - t0) * 1000
        return BatchResult(
            results=results,
            total=len(texts),
            successful=len(texts) - failed,
            failed=failed,
            total_processing_time_ms=round(elapsed, 2),
        )

    def full_analysis(self, text: str) -> dict:
        """
        Extended pipeline: runs ALL modules including morphology, POS, stemming,
        statistics, and readability in addition to the standard pipeline.

        Returns:
            Dict with keys: dialect, sentiment, tokens, entities, normalized,
            pos_tags, statistics, readability, pipeline_time_ms.
        """
        self._validate_input(text)
        t0 = time.perf_counter()

        base = self.analyze(text)

        # Additional modules
        pos_result = self.tag_pos(text)
        stats = self.get_text_stats(text)
        readability = self.get_readability(text)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("Full extended pipeline completed in %.1f ms", elapsed)

        return {
            **base,
            "pos_tags": pos_result,
            "statistics": stats,
            "readability": readability,
            "pipeline_time_ms": round(elapsed, 2),
        }
