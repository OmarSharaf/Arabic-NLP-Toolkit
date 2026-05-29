# API Reference (Quick)

Full API is available via `ArabicNLP` and individual module classes.

## `ArabicNLP` façade

```python
from arabic_nlp import ArabicNLP
from arabic_nlp.models import ArabicNLPConfig, Dialect, Script

nlp = ArabicNLP(config=ArabicNLPConfig(keyword_top_n=15))
```

| Method | Returns | Description |
|--------|---------|-------------|
| `detect_dialect(text)` | `DialectResult` | 8 Arabic dialects |
| `sentiment(text, dialect=None)` | `SentimentResult` | Polarity + scores |
| `tokenize(text, ...)` | `list[Token]` | Tokens with offsets |
| `extract_entities(text)` | `NERResult` | Persons, places, orgs, … |
| `normalize(text, ...)` | `NormalizationResult` | Text cleanup |
| `transliterate(text, source, target)` | `TransliterationResult` | Script conversion |
| `extract_keywords(text, top_n=10)` | `KeywordResult` | Ranked keywords |
| `profile(text)` | `TextProfile` | Register + quality |
| `analyze_document(text, ...)` | `DocumentAnalysis` | JSON-serializable pipeline |
| `tag_pos(text)` | `POSSequenceResult` | POS tags |
| `morphology(word)` | `MorphResult` | Morphological analysis |
| `stem(word)` | `str` | Light stemming |
| `analyze(text)` | `dict` | Core pipeline |
| `full_analysis(text)` | `dict` | Extended pipeline |
| `batch_sentiment(texts)` | `BatchResult` | Batch sentiment |
| `batch_dialect(texts)` | `BatchResult` | Batch dialect |
| `batch_analyze(texts)` | `BatchResult` | Batch documents |

## CLI

```bash
pip install arabic-nlp-toolkit
arabic-nlp detect-dialect "ازيك عامل ايه"
arabic-nlp keywords "الذكاء الاصطناعي" --top 5
arabic-nlp profile "نص سوشيال ميديا 😂"
arabic-nlp export "نص كامل" --keywords --profile -o json
```

## Exceptions

| Exception | When |
|-----------|------|
| `InvalidInputError` | Empty or non-string input |
| `UnsupportedDialectError` | Unknown dialect for stopwords |
| `ArabicNLPError` | Base class for library errors |

See module docstrings and [README](../README.md) for examples.
