<div align="center">

<img src="assets/banner.svg" alt="Arabic NLP Toolkit" width="100%" />

<br/>

<img src="assets/logo.svg" alt="Arabic NLP Toolkit" width="72" height="72" />

# arabic-nlp-toolkit

**Production-grade Arabic natural language processing for dialectal, social, and standard text.**

<p dir="rtl"><strong>معالجة لغوية عربية على مستوى الإنتاج — للهجات والنصوص الرقمية والفصحى</strong></p>

<br/>

[![PyPI](https://img.shields.io/pypi/v/arabic-nlp-toolkit?style=for-the-badge&logo=pypi&logoColor=white&labelColor=1a2332&color=3b82f6)](https://pypi.org/project/arabic-nlp-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/arabic-nlp-toolkit?style=for-the-badge&logo=python&logoColor=white&labelColor=1a2332&color=6366f1)](https://pypi.org/project/arabic-nlp-toolkit/)
[![CI](https://img.shields.io/github/actions/workflow/status/OmarSharaf/Arabic-NLP-Toolkit/ci.yml?style=for-the-badge&logo=github-actions&logoColor=white&labelColor=1a2332&label=CI)](https://github.com/OmarSharaf/Arabic-NLP-Toolkit/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-301%2B-success?style=for-the-badge&labelColor=1a2332)](./tests/)
[![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen?style=for-the-badge&labelColor=1a2332)](./tests/)
[![License](https://img.shields.io/badge/license-MIT-yellow?style=for-the-badge&labelColor=1a2332)](./LICENSE)

<br/>

[**Install**](#installation) · [**Quick Start**](#quick-start) · [**Live Demo**](#live-demo) · [**Capabilities**](#capabilities) · [**API**](./docs/API.md) · [**Contributing**](./CONTRIBUTING.md)

</div>

<br/>

> **Arabic NLP Toolkit** is a typed, offline-first Python library for analyzing real Arabic as it appears in the wild — Egyptian, Gulf, Levantine, Maghrebi, and Modern Standard Arabic — with a single, consistent API and structured Pydantic outputs ready for APIs, analytics, and research pipelines.

---

## At a glance

<table>
<tr>
<td width="25%" align="center">
<h3>8</h3>
<p><strong>Dialects</strong><br/><sub>Confidence-ranked detection</sub></p>
</td>
<td width="25%" align="center">
<h3>15+</h3>
<p><strong>NLP modules</strong><br/><sub>One unified façade</sub></p>
</td>
<td width="25%" align="center">
<h3>301+</h3>
<p><strong>Tests</strong><br/><sub>~89% coverage</sub></p>
</td>
<td width="25%" align="center">
<h3>1</h3>
<p><strong>Core dependency</strong><br/><sub>Pydantic v2 only</sub></p>
</td>
</tr>
</table>

| | |
|:--|:--|
| **Designed for** | Social media, reviews, chat, news, and mixed-register Arabic |
| **Output model** | Frozen Pydantic v2 objects · JSON-serializable · IDE-friendly |
| **Deployment** | No GPU · No model downloads · Runs fully offline |
| **Integration** | Python API · CLI · FastAPI web demo · batch pipelines |

---

## Table of contents

<details open>
<summary><strong>Expand navigation</strong></summary>

- [Why Arabic NLP Toolkit](#why-arabic-nlp-toolkit)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Live Demo](#live-demo)
- [Capabilities](#capabilities)
- [NLP Pipeline](#nlp-pipeline)
- [Supported dialects](#supported-dialects)
- [Command-line interface](#command-line-interface)
- [Architecture](#architecture)
- [Development](#development)
- [Quality & CI/CD](#quality--cicd)
- [Documentation](#documentation)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Author](#author)
- [License](#license)

</details>

---

## Why Arabic NLP Toolkit

Arabic is not a monolith. Hundreds of millions of speakers use **regional dialects** every day — in comments, DMs, product reviews, and customer support — while most NLP stacks still optimize for textbook Modern Standard Arabic.

**Arabic NLP Toolkit** closes that gap with a production-minded toolkit built for practitioners who need reliable results without spinning up GPUs or external APIs.

### What sets it apart

| Principle | How we deliver it |
|-----------|-------------------|
| **Dialect-first** | Eight varieties with ranked confidence scores and Arabic display names |
| **Social-native** | Normalization for mentions, hashtags, URLs, emoji, and Franco-Arabic chat script |
| **Type-safe by default** | Every result is a validated Pydantic model — not ad-hoc dicts |
| **Ship-ready** | Single-call document analysis with `.to_json()` for REST and ETL |
| **Lean core** | Rule-based engine; optional `[ml]` / `[transformers]` extras when you need more |

### Landscape comparison

| Capability | **arabic-nlp-toolkit** | camel-tools | farasa | pyarabic |
|:-----------|:--:|:--:|:--:|:--:|
| Multi-dialect detection (8) | ✓ | — | — | — |
| Dialect-aware sentiment | ✓ | — | — | — |
| Franco-Arabic transliteration | ✓ | — | — | — |
| Keyword extraction | ✓ | — | — | — |
| Text register & quality profiling | ✓ | — | — | — |
| Morphology · POS · stemming | ✓ | ✓ | partial | partial |
| Social text normalization | ✓ | — | — | — |
| Pydantic v2 structured output | ✓ | — | — | — |
| Interactive web demo | ✓ | — | — | — |
| Core runtime dependencies | `pydantic` | many | API | none |

---

## Installation

**Requirements:** Python 3.9 – 3.12

```bash
pip install arabic-nlp-toolkit
```

### Optional extras

```bash
pip install arabic-nlp-toolkit[web]            # FastAPI live demo
pip install arabic-nlp-toolkit[dev]            # pytest, ruff, mypy, build tools
pip install arabic-nlp-toolkit[ml]             # scikit-learn (future ML path)
pip install arabic-nlp-toolkit[transformers]   # HuggingFace (future path)
```

For local development with the full toolchain:

```bash
pip install -e ".[dev,web]"
```

---

## Quick Start

```python
from arabic_nlp import ArabicNLP

nlp = ArabicNLP()

# Detect dialect with confidence scores
dialect = nlp.detect_dialect("ازيك عامل ايه النهارده؟")
print(dialect.dialect, f"{dialect.confidence:.0%}")   # egyptian 93%

# Sentiment with negation handling
sentiment = nlp.sentiment("المنتج ده رائع جداً!")
print(sentiment.label, f"{sentiment.score:.0%}")      # positive 91%

# Full document analysis — one call, structured output
doc = nlp.analyze_document("زيارة محمد صلاح للقاهرة كانت رائعة")
print(doc.summary)
print(doc.to_json())   # drop into APIs, warehouses, or dashboards
```

---

## Live Demo

Explore every module in the browser — no setup beyond a single install.

```bash
pip install -e ".[web]"
python webapp/app.py
```

| | |
|:--|:--|
| **URL** | http://127.0.0.1:8765 |
| **Playground** | Dialect · sentiment · NER · keywords · profiling · POS · stats · transliteration |
| **Profile** | Project overview, pipeline diagram, dialect map, feature catalog |

| Environment variable | Effect |
|---------------------|--------|
| `ARABIC_NLP_NO_BROWSER=1` | Do not auto-open the browser |
| `CI=1` | Same as above (for CI environments) |

→ Full endpoint reference: [docs/WEBAPP.md](./docs/WEBAPP.md)

<p align="center">
  <img src="assets/pipeline.svg" alt="NLP pipeline diagram" width="92%" />
</p>

---

## Capabilities

### Dialect detection

Identify regional variety with ranked alternatives and Arabic labels.

```python
result = nlp.detect_dialect("شلون حالك اليوم؟ وين رحت؟")
print(result.dialect, result.dialect_name_ar)   # gulf · خليجي

for score in result.all_scores[:3]:
    print(f"  {score.dialect}: {score.confidence:.0%}")
```

### Sentiment analysis

Lexicon-driven analysis with negation (`مش`), intensifiers, and optional dialect hints.

```python
nlp.sentiment("المنتج مش كويس")                              # negative
nlp.sentiment("تحفة أوي", dialect=Dialect.EGYPTIAN)          # positive
```

### Named entity recognition

Gazetteer- and pattern-based extraction of persons, locations, and organizations.

```python
entities = nlp.extract_entities("زار محمد صلاح ملعب الأهلي في القاهرة")
print(entities.persons, entities.locations, entities.organizations)
```

### Tokenization & normalization

Offset-aware tokenization and configurable social-text cleanup.

```python
for token in nlp.tokenize("مرحباً بكم! 🇪🇬"):
    print(token.text, token.token_type, token.start, token.end)

nlp.normalize(
    "@user شوف #مصر 🔥",
    remove_mentions=True,
    remove_emojis=True,
)
```

### Transliteration

Bidirectional conversion between Arabic script and Franco-Arabic / Buckwalter.

```python
from arabic_nlp.models import Script

nlp.transliterate("مرحبا", target=Script.FRANCO)
nlp.transliterate("ana mesh 3aref", source=Script.FRANCO, target=Script.ARABIC)
```

### Keywords & text profiling

Extract salient terms and assess register, quality, and recommendations.

```python
keywords = nlp.extract_keywords("الذكاء الاصطناعي يغير التعليم", top_n=5)
profile  = nlp.profile("ازيك 😂 #مصر https://example.com")

print(profile.text_register, profile.quality_score)
print(profile.recommendations)
```

### Morphology, POS & stemming

Classical Arabic NLP building blocks for downstream linguistic tasks.

```python
nlp.morphology("الكاتب")                        # root, pattern, features
nlp.tag_pos("ذهب الولد إلى المدرسة")            # universal POS tags
nlp.stem("الكتاب")                              # كتاب
```

### Document export & batch processing

Configure pipelines once; analyze at scale.

```python
from arabic_nlp.models import ArabicNLPConfig

nlp = ArabicNLP(config=ArabicNLPConfig(
    extract_keywords_on_analyze=True,
    profile_on_analyze=True,
))

doc = nlp.analyze_document("نص للتحليل الكامل")
doc.to_dict()    # REST / ETL
doc.summary      # logging & monitoring

nlp.batch_sentiment(["رائع", "سيء", "عادي"])
nlp.batch_analyze(texts, include_keywords=True)
```

---

## NLP Pipeline

From raw Arabic text to a single structured artifact:

```
  Input
    → Normalize     (diacritics, alef variants, social cleanup)
    → Tokenize      (offset-aware, type-tagged)
    → Analyze       (dialect · sentiment · NER · POS · morphology)
    → Enrich        (keywords · text profile · statistics)
    → Export        (DocumentAnalysis → JSON / summary)
```

<p align="center">
  <img src="assets/dialects.svg" alt="Eight Arabic dialects" width="88%" />
</p>

---

## Supported dialects

| Code | Dialect | العربية |
|:-----|:--------|:--------|
| `msa` | Modern Standard Arabic | فصحى |
| `egyptian` | Egyptian | مصري |
| `gulf` | Gulf | خليجي |
| `levantine` | Levantine | شامي |
| `maghrebi` | Maghrebi | مغاربي |
| `iraqi` | Iraqi | عراقي |
| `yemeni` | Yemeni | يمني |
| `sudanese` | Sudanese | سوداني |

---

## Command-line interface

```bash
arabic-nlp detect-dialect "ازيك عامل ايه"
arabic-nlp sentiment "المنتج رائع"
arabic-nlp keywords "الذكاء الاصطناعي" --top 5
arabic-nlp profile "نص سوشيال ميديا 😂"
arabic-nlp export "نص كامل" --keywords --profile -o json
arabic-nlp analyze "نص للتحليل الشامل"
```

All commands accept `--output json` for scripting and automation.

---

## Architecture

```
arabic_nlp/
├── core.py              # ArabicNLP — unified entry point
├── models.py            # Pydantic v2 schemas & configuration
├── cli/                 # Command-line interface
├── dialects/            # Eight-dialect classifier
├── sentiment/           # Lexicon + negation engine
├── tokenizer/           # Offset-aware tokenizer
├── ner/                 # Gazetteer + pattern NER
├── normalization/       # Script & social normalization
├── transliteration/     # Franco-Arabic · Buckwalter
├── stopwords/           # Per-dialect stopword sets
├── morphology/          # Roots, patterns, features
├── pos/                 # Arabic + Universal POS
├── stemmer/             # Light & aggressive stemming
├── keywords/            # TF-based keyword extraction
├── profiling/           # Register & quality scoring
└── utils/               # Statistics · readability · similarity

webapp/                  # FastAPI interactive demo
assets/                  # Brand & diagram assets (SVG)
tests/                   # 301+ unit & integration tests
```

| Design choice | Rationale |
|:--------------|:----------|
| Lazy module loading | Fast cold start on `import arabic_nlp` |
| Frozen Pydantic models | Immutable, thread-safe, cache-friendly results |
| Rule-based core | Predictable, offline, no model registry |
| Optional ML extras | Clear upgrade path without bloating the default install |

---

## Development

```bash
git clone https://github.com/OmarSharaf/Arabic-NLP-Toolkit.git
cd Arabic-NLP-Toolkit
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev,web]"
pytest
ruff check arabic_nlp/ tests/ webapp/
python webapp/app.py
```

Contribution guidelines: [CONTRIBUTING.md](./CONTRIBUTING.md)

---

## Quality & CI/CD

Every push and pull request is validated via [GitHub Actions](.github/workflows/ci.yml):

| Stage | Scope |
|:------|:------|
| **Lint** | `ruff check` + format enforcement |
| **Test** | `pytest` across Python 3.9–3.12 on Ubuntu & Windows |
| **Build** | `python -m build` + `twine check` |
| **Web smoke** | FastAPI health & dialect API via TestClient |

Tagged releases (`v*.*.*`) trigger automated PyPI publish and GitHub Release creation via [release.yml](.github/workflows/release.yml).

---

## Documentation

| Resource | Description |
|:---------|:------------|
| [docs/API.md](./docs/API.md) | API quick reference |
| [docs/WEBAPP.md](./docs/WEBAPP.md) | Web demo & REST endpoints |
| [docs/PROFILE.md](./docs/PROFILE.md) | Text profiling guide |
| [CHANGELOG.md](./CHANGELOG.md) | Version history |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | Contribution workflow |
| [SECURITY.md](./SECURITY.md) | Vulnerability reporting |
| [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) | Community standards |

---

## Roadmap

| Release | Status | Focus |
|:--------|:-------|:------|
| **v2.0** | Shipped | Morphology, POS, stemming, keywords, profiling, web demo, document export |
| **v2.1** | Planned | ML dialect classifier, AraBERT sentiment, expanded gazetteers |
| **v2.2** | Planned | Transformer NER, summarization, async API |

Track progress: [CHANGELOG.md](./CHANGELOG.md) · [GitHub Issues](https://github.com/OmarSharaf/Arabic-NLP-Toolkit/issues)

---

## Citation

If you use this library in academic or industry work, please cite:

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

**Omar S. M. Abdelfatah**

[omarsharaf.me](https://www.omarsharaf.me) · [GitHub](https://github.com/OmarSharaf) · [LinkedIn](https://www.linkedin.com/in/omarsharafaldin/)

---

## License

Released under the [MIT License](./LICENSE).

```
Copyright (c) 2025 Omar S. M. Abdelfatah
```

---

<div align="center">

<img src="assets/logo.svg" alt="" width="36" height="36" />

<br/><br/>

**Arabic NLP Toolkit** · v2.0.0 · MIT

<br/>

If this project supports your work, a star on GitHub helps other Arabic developers discover it.

<br/>

[![Star on GitHub](https://img.shields.io/github/stars/OmarSharaf/Arabic-NLP-Toolkit?style=for-the-badge&logo=github&label=Star%20this%20repo&labelColor=1a2332&color=3b82f6)](https://github.com/OmarSharaf/Arabic-NLP-Toolkit)

</div>
