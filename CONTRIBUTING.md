# Contributing to arabic-nlp-toolkit

Thank you for considering a contribution — every PR, issue, and lexicon addition helps the entire Arabic NLP community. 🙏

---

## Table of Contents

- [What We Need Most](#what-we-need-most)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Running Tests](#running-tests)
- [Adding Lexicon Entries](#adding-lexicon-entries)
- [Commit Convention](#commit-convention)
- [Pull Request Process](#pull-request-process)
- [Code of Conduct](#code-of-conduct)

---

## What We Need Most

1. **Lexicon expansions** — More dialect markers (Iraqi, Yemeni, Sudanese), more sentiment words, more NER gazetteers
2. **Test cases** — Edge cases, dialect-specific examples, adversarial inputs
3. **Benchmark datasets** — Links or contributions of labeled Arabic text datasets for evaluation
4. **Bug reports** — Especially wrong dialect predictions or incorrect entity spans
5. **ML model integration** — Wrappers for AraBERT, CAMeLBERT, or other Arabic transformers

---

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/OmarSharaf/arabic-nlp-toolkit.git
cd arabic-nlp-toolkit

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install in editable mode with all dev dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Run the examples to verify everything works
python examples/usage.py

# 6. (Optional) Launch the web demo
pip install -e ".[web]"
python webapp/app.py
```

---

## Project Structure

```
arabic_nlp/
├── dialects/       # Dialect detection — add markers to _DIALECT_LEXICONS
├── sentiment/      # Sentiment — add words to _POSITIVE_WORDS / _NEGATIVE_WORDS
├── tokenizer/      # Tokenizer — extend _PATTERNS for new token types
├── ner/            # NER — add to _PERSON_TITLES, _ARAB_CITIES, _ORGANIZATIONS
├── normalization/  # Normalization — add new regex patterns
├── transliteration/# Transliteration — extend mapping tables
├── stopwords/      # Stop words — add to dialect-specific frozensets
├── morphology/     # Morphology — extend root/pattern tables
├── pos/            # POS tagger — extend lexical lookup tables
├── stemmer/        # Stemmer — extend prefix/suffix lists
├── keywords/       # Keywords — tune scoring / stopword handling
├── profiling/      # Profiling — register heuristics, quality signals
├── utils/          # Utilities — add new helper functions
└── webapp/         # Web demo — FastAPI routes + static UI
```

---

## Coding Standards

- **Python 3.9+** syntax only
- **Type hints everywhere** — `mypy --strict` must pass
- **Docstrings** on all public functions and classes (Google-style)
- **No external dependencies** in core modules — only `pydantic>=2` is required
- **Frozen Pydantic models** for all result types
- **Test every change** — maintain ≥80% coverage

---

## Running Tests

```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/unit/dialects/
pytest tests/unit/sentiment/

# Run with coverage report
pytest --cov=arabic_nlp --cov-report=html
open htmlcov/index.html

# Run performance tests
pytest tests/performance/ -v

# Type checking
mypy arabic_nlp/

# Linting
ruff check arabic_nlp/ tests/

# Formatting
black arabic_nlp/ tests/
```

---

## Adding Lexicon Entries

### Dialect markers (`arabic_nlp/dialects/__init__.py`)

```python
# Add to the appropriate dict, e.g. _IRAQI_MARKERS
_IRAQI_MARKERS: dict[str, float] = {
    ...
    "new_marker": 0.9,   # weight between 0.0 and 1.0
    "multi word marker": 1.0,   # phrases work too
}
```

### Sentiment words (`arabic_nlp/sentiment/__init__.py`)

```python
# Positive: score between +0.1 and +1.0
_POSITIVE_WORDS["new_word"] = 0.8

# Negative: score between -0.1 and -1.0
_NEGATIVE_WORDS["new_word"] = -0.7
```

### NER gazetteers (`arabic_nlp/ner/__init__.py`)

```python
# Add to the appropriate frozenset
_ARAB_CITIES = frozenset([
    ...
    "مدينة جديدة",
])
```

---

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>
```

**Types:** `feat`, `fix`, `docs`, `test`, `perf`, `refactor`, `chore`

**Scopes:** `dialects`, `sentiment`, `ner`, `tokenizer`, `normalization`, `transliteration`, `morphology`, `pos`, `stemmer`, `keywords`, `profiling`, `utils`, `cli`, `webapp`, `core`, `ci`

**Examples:**
```
feat(dialects): add 50 new Iraqi dialect markers
fix(sentiment): correct negation window for مش
perf(ner): optimize gazetteer lookup with frozenset
test(morphology): add dual number detection tests
docs(readme): update quickstart examples
```

---

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Make your changes with tests (maintain ≥80% coverage)
3. Ensure `pytest` and `ruff check` pass locally
4. Update `CHANGELOG.md` under `[Unreleased]`
5. Open a PR targeting `main` with a clear description and test plan
6. CI must pass (lint, test matrix, build) before merge

---

## Code of Conduct

Please read [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md). We are committed to a welcoming, inclusive community.

---

Thank you for contributing to Arabic NLP! 🌍🔤
