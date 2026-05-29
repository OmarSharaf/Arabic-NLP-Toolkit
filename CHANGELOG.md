# Changelog

All notable changes to `arabic-nlp-toolkit` are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [Unreleased]

### Added
- **Keyword extraction** (`arabic_nlp.keywords`) — TF-based ranked keywords with dialect stopwords
- **Text profiling** (`arabic_nlp.profiling`) — register detection, quality score, preprocessing recommendations
- **`DocumentAnalysis`** — JSON-serializable pipeline output with `.to_json()` and `.summary`
- **`ArabicNLPConfig`** — centralized runtime configuration
- **Web demo** (`webapp/`) — FastAPI + RTL UI for interactive testing
- CLI commands: `keywords`, `profile`, `export`
- `ArabicNLP.extract_keywords()`, `.profile()`, `.analyze_document()`, `.batch_analyze()`, `.content_words()`, `.split_sentences()`
- CI/CD: GitHub Actions (lint, test matrix, build, web smoke), release workflow, Dependabot
- Documentation: `CODE_OF_CONDUCT.md`, `SECURITY.md`, `docs/API.md`, `docs/WEBAPP.md`

### Fixed
- Morphology root extraction for trilateral words like `كتب` (proclitic `ك` no longer over-strips)
- POS tagger: `من` classified as preposition before interrogative particle

---

## [2.0.0] — 2025-01-01

### 🚀 Major Release — New Modules

**New: Morphological Analyzer** (`arabic_nlp.morphology`)
- Root extraction (3-letter and 4-letter trilateral/quadrilateral roots)
- Prefix/suffix decomposition with named detection
- Gender (masculine/feminine), Number (singular/dual/plural), Definiteness detection
- Verb tense (past/present/command) and voice (active/passive) analysis
- Arabic morphological pattern matching (أوزان: فَعَلَ, فَاعِل, مَفْعُول, ...)
- `ArabicMorphAnalyzer` with `analyze()`, `analyze_batch()`, `get_root()`, `get_stem()`
- Full test suite: 30+ unit tests

**New: POS Tagger** (`arabic_nlp.pos`)
- Universal Dependencies POS tags: NOUN, VERB, ADJ, ADV, ADP, CCONJ, SCONJ, DET, PRON, NUM, PUNCT, INTJ, PART
- Arabic-specific fine-grained tags: NOUN_PROP, PRON_REL, PRON_DEM, PART_NEG, PART_INTER, VERB_PASS
- Lexical lookup tables: prepositions, conjunctions, pronouns, particles, adverbs, interjections
- Morphological heuristics for verb/noun classification
- `ArabicPOSTagger` with `tag()`, `tag_sentence()`, `tag_tokens()`
- `POSSequenceResult` with `.verbs`, `.nouns`, `.adjectives`, `.filter_by_tag()` properties
- Full test suite: 25+ unit tests

**New: Arabic Stemmer** (`arabic_nlp.stemmer`)
- Light Stemmer (Larkey-style) — removes common prefixes/suffixes for IR tasks
- Aggressive Stemmer — multi-pass stripping for closer-to-root reduction
- Stop-word guard (particles and prepositions not stemmed)
- Minimum stem length protection (prevents over-stemming)
- `ArabicStemmer` with `stem()`, `stem_batch()`, `stem_text()`
- `StemResult` with `was_reduced`, `prefixes_removed`, `suffixes_removed`
- Full test suite: 20+ unit tests

**New: Utilities** (`arabic_nlp.utils`)
- `get_statistics()` — 15+ text metrics: word count, TTR, Arabic ratio, sentence count, most common words
- `get_readability()` — difficulty scoring (easy/moderate/difficult/very_difficult) with recommendation
- `compute_similarity()` — Jaccard + cosine + overlap coefficient
- `detect_language()` — Arabic/Latin/digit ratio breakdown
- `is_arabic()` — threshold-based Arabic detection
- `remove_diacritics()`, `has_diacritics()`, `count_arabic_words()`
- `arabic_to_western_digits()`, `western_to_arabic_digits()`
- `extract_numbers()`, `extract_arabic_only()`, `truncate_arabic()`
- `word_frequency()`, `sentence_split()`, `normalize_unicode()`
- Full test suite: 35+ unit tests

**New: Full CLI** (`arabic-nlp` command)
- `arabic-nlp detect-dialect "text"` — dialect detection with visual confidence bars
- `arabic-nlp sentiment "text"` — sentiment with score breakdown
- `arabic-nlp tokenize "text"` — token table with offsets and types
- `arabic-nlp ner "text"` — entity table with emoji labels
- `arabic-nlp normalize "text"` — before/after diff with change list
- `arabic-nlp transliterate "text" --target franco` — multi-script support
- `arabic-nlp stem "text" --mode light|aggressive` — stem table
- `arabic-nlp morph "word"` — full morphological breakdown
- `arabic-nlp pos "sentence"` — POS tag table
- `arabic-nlp stats "text"` — statistics + readability dashboard
- `arabic-nlp analyze "text"` — full pipeline summary
- All commands support `--output json` for programmatic use

**New: Examples**
- `examples/usage.py` — 12-section interactive tour of all features

**New: Performance Tests**
- `tests/performance/` — timing assertions for all core operations

**Core Enhancements**
- `ArabicNLP.morphology(word)` — morphological analysis via facade
- `ArabicNLP.tag_pos(text)` — POS tagging via facade
- `ArabicNLP.stem(word)` — quick stemming via facade
- `ArabicNLP.stem_text(text)` — bulk stemming
- `ArabicNLP.get_text_stats(text)` — statistics via facade
- `ArabicNLP.get_readability(text)` — readability via facade
- `ArabicNLP.similarity(text1, text2)` — similarity via facade
- `ArabicNLP.full_analysis(text)` — extended pipeline (adds POS, stats, readability)

---

## [1.0.0] — 2024-12-01

### 🎉 Initial Release

- Dialect detection (8 dialects: MSA, Egyptian, Gulf, Levantine, Maghrebi, Iraqi, Yemeni, Sudanese)
- Sentiment analysis with negation handling and intensifiers
- Tokenizer with character offsets and 10 token types
- Named Entity Recognition (PERSON, LOCATION, ORGANIZATION, DATE, TIME, MONEY)
- Text normalization (diacritics, alef, hamza, teh marbuta, tatweel, social media)
- Transliteration (Arabic ↔ Franco-Arabic, Arabic ↔ Buckwalter)
- Dialect-aware stop words (MSA, Egyptian, Gulf, Levantine)
- `ArabicNLP` unified façade with lazy loading
- Batch processing for sentiment and dialect
- Frozen Pydantic v2 result models throughout
- CI matrix: Python 3.9–3.12 × Ubuntu/Windows/macOS

---

[Unreleased]: https://github.com/OmarSharaf/arabic-nlp-toolkit/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/OmarSharaf/arabic-nlp-toolkit/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/OmarSharaf/arabic-nlp-toolkit/releases/tag/v1.0.0
