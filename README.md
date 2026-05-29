<div align="center">

<img src="assets/banner.svg" alt="Arabic NLP Toolkit — معالجة عربية احترافية" width="100%" />

<br/><br/>

<img src="assets/logo.svg" alt="Arabic NLP Toolkit logo" width="88" height="88" />

<h1>arabic-nlp-toolkit</h1>

<p><strong>Production-ready Arabic NLP for real-world text</strong><br/>
<strong>معالجة اللغة العربية الاحترافية للنصوص الحقيقية</strong></p>

<p>Dialects · Sentiment · NER · Morphology · POS · Keywords · Profiling · Web demo</p>

[![PyPI version](https://img.shields.io/pypi/v/arabic-nlp-toolkit?style=flat-square&color=blue)](https://pypi.org/project/arabic-nlp-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/arabic-nlp-toolkit?style=flat-square)](https://pypi.org/project/arabic-nlp-toolkit/)
[![CI](https://img.shields.io/github/actions/workflow/status/OmarSharaf/Arabic-NLP-Toolkit/ci.yml?style=flat-square&label=CI)](https://github.com/OmarSharaf/Arabic-NLP-Toolkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-301%2B%20passed-success?style=flat-square)](./tests/)
[![Coverage](https://img.shields.io/badge/coverage-~89%25-brightgreen?style=flat-square)](./tests/)

<br/>

**[Quick Start](#quick-start)** · **[Web Demo](#web-demo)** · **[Features](#features)** · **[Pipeline](#nlp-pipeline)** · **[API](./docs/API.md)** · **[Contributing](./CONTRIBUTING.md)**

</div>

---

## Table of contents

- [Why this library?](#why-this-library)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Web Demo](#web-demo)
- [Features](#features)
- [NLP Pipeline](#nlp-pipeline)
- [Dialects](#dialects)
- [CLI](#cli)
- [Architecture](#architecture)
- [Development](#development)
- [CI/CD](#cicd)
- [Documentation](#documentation)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Author](#author)
- [License](#license)

---

## Why this library?

Most Arabic NLP tools target **Modern Standard Arabic (MSA)** only. Over 400 million people speak **Egyptian, Gulf, Levantine, and Maghrebi** Arabic daily — on social media, in reviews, and in chat.

`arabic-nlp-toolkit` is built for that reality:

| | |
|---|---|
| 🌍 | **8 dialects** with confidence scores |
| 📱 | **Social-media-ready** normalization and profiling |
| 🔤 | **Franco-Arabic** transliteration (chat alphabet) |
| 📦 | **Typed Pydantic v2** results — IDE-friendly, JSON-serializable |
| ⚡ | **Zero required dependencies** beyond `pydantic` for the core library |
| 🖥️ | **Interactive web demo** — test every feature in the browser |

### Comparison

| Capability | arabic-nlp-toolkit | camel-tools | farasa | pyarabic |
|---|:---:|:---:|:---:|:---:|
| Dialect detection (8 dialects) | ✅ | ❌ | ❌ | ❌ |
| Dialect sentiment | ✅ | ❌ | ❌ | ❌ |
| Franco-Arabic ↔ Arabic | ✅ | ❌ | ❌ | ❌ |
| Keyword extraction | ✅ | ❌ | ❌ | ❌ |
| Text profiling (register/quality) | ✅ | ❌ | ❌ | ❌ |
| Morphology + POS + Stemming | ✅ | ✅ | ⚠️ | ⚠️ |
| Social media cleanup | ✅ | ❌ | ❌ | ❌ |
| Pydantic v2 results | ✅ | ❌ | ❌ | ❌ |
| Interactive web demo | ✅ | ❌ | ❌ | ❌ |
| Core deps | `pydantic` only | many | API | none |

---

## Installation

```bash
pip install arabic-nlp-toolkit
```

**Optional extras:**

```bash
pip install arabic-nlp-toolkit[dev]           # pytest, ruff, mypy, …
pip install arabic-nlp-toolkit[web]           # FastAPI web demo
pip install arabic-nlp-toolkit[ml]            # scikit-learn (future ML models)
pip install arabic-nlp-toolkit[transformers]  # HuggingFace (future)
```

**Requirements:** Python 3.9 – 3.12

---

## Quick Start

```python
from arabic_nlp import ArabicNLP

nlp = ArabicNLP()

# Dialect
r = nlp.detect_dialect("ازيك عامل ايه النهارده؟")
print(r.dialect, r.confidence)  # egyptian 0.93

# Sentiment
r = nlp.sentiment("المنتج ده رائع جداً!")
print(r.label, r.score)  # positive 0.91

# Full pipeline → JSON-ready document
doc = nlp.analyze_document("زيارة محمد صلاح للقاهرة كانت رائعة")
print(doc.summary)
print(doc.to_json())
```

---

## Web Demo

Test every feature in the browser — no code required.

```bash
pip install -e ".[web]"
python webapp/app.py
```

Opens **http://127.0.0.1:8765** with a polished dark RTL UI:

| Tab | What you get |
|-----|----------------|
| **Playground** | Live analysis — dialect, sentiment, NER, keywords, profiling, POS, stats, transliteration |
| **Profile** | Project showcase — stats, feature cards, pipeline & dialect diagrams |

**Screens & assets**

- Favicon & logo: [`assets/logo.svg`](./assets/logo.svg)
- Hero banner: [`assets/banner.svg`](./assets/banner.svg)
- Pipeline diagram: [`assets/pipeline.svg`](./assets/pipeline.svg)
- Dialect map: [`assets/dialects.svg`](./assets/dialects.svg)

**URLs**

- Playground: http://127.0.0.1:8765/#playground  
- Project profile: http://127.0.0.1:8765/#profile  

Set `ARABIC_NLP_NO_BROWSER=1` or `CI=1` to skip auto-opening Chrome.

See [docs/WEBAPP.md](./docs/WEBAPP.md) for API endpoints and development notes.

**GitHub social preview:** upload [`assets/social-preview.svg`](./assets/social-preview.svg) in repo **Settings → General → Social preview**.

---

## Features

### Dialect detection

```python
result = nlp.detect_dialect("شلون حالك اليوم؟ وين رحت؟")
print(result.dialect)           # Dialect.GULF
print(result.dialect_name_ar)   # خليجي
for s in result.all_scores[:3]:
    print(s.dialect, f"{s.confidence:.0%}")
```

### Sentiment analysis

Handles negation (`مش`), intensifiers, and dialect hints.

```python
nlp.sentiment("المنتج مش كويس")                    # negative
nlp.sentiment("تحفة أوي", dialect=Dialect.EGYPTIAN)
```

### Named Entity Recognition

```python
result = nlp.extract_entities("زار محمد صلاح ملعب الأهلي في القاهرة")
print(result.persons, result.locations, result.organizations)
```

### Tokenization & normalization

```python
for tok in nlp.tokenize("مرحباً بكم! 🇪🇬"):
    print(tok.text, tok.token_type, tok.start, tok.end)

nlp.normalize("@user شوف #مصر 🔥", remove_mentions=True, remove_emojis=True)
```

### Transliteration

```python
from arabic_nlp.models import Script

nlp.transliterate("مرحبا", target=Script.FRANCO)           # mrhba …
nlp.transliterate("ana mesh 3aref", source=Script.FRANCO, target=Script.ARABIC)
```

### Keywords & text profiling *(v2.0+)*

```python
kw = nlp.extract_keywords("الذكاء الاصطناعي يغير التعليم", top_n=5)
profile = nlp.profile("ازيك 😂 #مصر https://example.com")
print(profile.text_register, profile.quality_score)
print(profile.recommendations)
```

### Morphology, POS, stemming

```python
nlp.morphology("الكاتب")   # root, gender, definiteness, …
nlp.tag_pos("ذهب الولد إلى المدرسة")
nlp.stem("الكتاب")         # كتاب
```

### Structured document export

```python
from arabic_nlp.models import ArabicNLPConfig

nlp = ArabicNLP(config=ArabicNLPConfig(
    extract_keywords_on_analyze=True,
    profile_on_analyze=True,
))
doc = nlp.analyze_document("نص للتحليل الكامل")
doc.to_dict()   # REST API / ETL pipelines
doc.summary     # logging dashboards
```

### Batch processing

```python
nlp.batch_sentiment(["رائع", "سيء", "عادي"])
nlp.batch_dialect(texts)
nlp.batch_analyze(texts, include_keywords=True)
```

---

## NLP Pipeline

End-to-end flow from raw Arabic text to structured output:

<p align="center">
  <img src="assets/pipeline.svg" alt="Arabic NLP pipeline: normalize → tokenize → dialect → sentiment → NER → keywords → profile" width="100%" />
</p>

```
Input text
    → Normalize (diacritics, alef, social cleanup)
    → Tokenize (offset-aware)
    → Dialect detection
    → Sentiment · NER · POS · Morphology
    → Keywords · Text profile
    → DocumentAnalysis (JSON / summary)
```

---

## Dialects

Eight Arabic varieties with confidence-ranked scores:

<p align="center">
  <img src="assets/dialects.svg" alt="Supported dialects: MSA, Egyptian, Gulf, Levantine, Maghrebi, Iraqi, Yemeni, Sudanese" width="100%" />
</p>

| Code | Dialect | العربية |
|------|---------|---------|
| `msa` | Modern Standard Arabic | فصحى |
| `egyptian` | Egyptian | مصري |
| `gulf` | Gulf | خليجي |
| `levantine` | Levantine | شامي |
| `maghrebi` | Maghrebi | مغاربي |
| `iraqi` | Iraqi | عراقي |
| `yemeni` | Yemeni | يمني |
| `sudanese` | Sudanese | سوداني |

---

## CLI

```bash
arabic-nlp detect-dialect "ازيك عامل ايه"
arabic-nlp sentiment "المنتج رائع"
arabic-nlp keywords "الذكاء الاصطناعي" --top 5
arabic-nlp profile "نص سوشيال ميديا 😂"
arabic-nlp export "نص كامل" --keywords --profile -o json
arabic-nlp analyze "نص للتحليل الشامل"
arabic-nlp --help
```

All commands support `--output json`.

---

## Architecture

```
arabic_nlp/
├── core.py              # ArabicNLP façade
├── models.py            # Pydantic v2 DTOs
├── cli/                 # arabic-nlp CLI
├── dialects/            # 8-dialect detection
├── sentiment/           # Lexicon + negation
├── tokenizer/           # Offset-aware tokenizer
├── ner/                 # Gazetteer + patterns
├── normalization/       # Diacritics, alef, social
├── transliteration/     # Franco, Buckwalter
├── stopwords/           # Per-dialect sets
├── morphology/          # Roots, patterns, features
├── pos/                 # Universal + Arabic POS
├── stemmer/             # Light / aggressive
├── keywords/            # TF keyword extraction
├── profiling/           # Register + quality
└── utils/               # Stats, readability, similarity

webapp/                  # FastAPI interactive demo
assets/                  # SVG logos, banners, diagrams
tests/                   # 301+ unit & integration tests
```

**Design principles**

- Lazy module loading — fast `import arabic_nlp`
- Frozen Pydantic models — immutable, thread-safe results
- Rule-based core — no GPU, no download, works offline
- Optional ML extras — upgrade path via `[ml]` / `[transformers]`

---

## Development

```bash
git clone https://github.com/OmarSharaf/Arabic-NLP-Toolkit.git
cd Arabic-NLP-Toolkit
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -e ".[dev,web]"
pytest                    # run tests (301+)
ruff check arabic_nlp/ tests/ webapp/
python examples/usage.py  # feature tour
python webapp/app.py      # web demo
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for lexicon guidelines, commit format, and PR process.

---

## CI/CD

Every push and PR runs [GitHub Actions](.github/workflows/ci.yml):

| Job | What it does |
|-----|----------------|
| **Lint** | `ruff check` + format check |
| **Test** | `pytest` on Python 3.9–3.12 × Ubuntu & Windows |
| **Build** | `python -m build` + `twine check` |
| **Web smoke** | FastAPI health + dialect API |

Releases tagged `v*.*.*` trigger [release.yml](.github/workflows/release.yml) (PyPI publish + GitHub Release).

---

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](./README.md) | You are here |
| [CHANGELOG.md](./CHANGELOG.md) | Version history |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | How to contribute |
| [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) | Community standards |
| [SECURITY.md](./SECURITY.md) | Vulnerability reporting |
| [docs/API.md](./docs/API.md) | Quick API reference |
| [docs/WEBAPP.md](./docs/WEBAPP.md) | Web demo guide |
| [docs/PROFILE.md](./docs/PROFILE.md) | Text profiling guide |

---

## Roadmap

| Version | Status | Highlights |
|---------|--------|------------|
| **v2.0** | ✅ | Morphology, POS, stemming, keywords, profiling, web demo, document export |
| **v2.1** | 🔜 | ML dialect classifier, AraBERT sentiment, expanded gazetteers |
| **v2.2** | 📋 | Transformer NER, summarization, async API |

Track progress in [CHANGELOG.md](./CHANGELOG.md) and [GitHub Issues](https://github.com/OmarSharaf/Arabic-NLP-Toolkit/issues).

---

## Citation

```bibtex
@software{abdelfatah2025arabicnlp,
  author  = {Abdelfatah, Omar S. M.},
  title   = {arabic-nlp-toolkit: Comprehensive Arabic NLP Library},
  year    = {2025},
  url     = {https://github.com/OmarSharaf/Arabic-NLP-Toolkit},
  version = {2.0.0},
  license = {MIT}
}
```

---

## Author

**Omar S. M. Abdelfatah** — [omarsharaf.me](https://www.omarsharaf.me) · [GitHub](https://github.com/OmarSharaf) · [LinkedIn](https://www.linkedin.com/in/omarsharafaldin/)

Built with care from Egypt 🇪🇬

---

## License

MIT — see [LICENSE](./LICENSE)

```
Copyright (c) 2025 Omar S. M. Abdelfatah
```

<div align="center">

<img src="assets/logo.svg" alt="" width="40" height="40" />

<br/>

**If this project helps you, please ⭐ the repo — it helps Arabic developers discover it.**

[⭐ Star on GitHub](https://github.com/OmarSharaf/Arabic-NLP-Toolkit)

</div>
