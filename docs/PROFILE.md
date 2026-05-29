# Project Profile

<div align="center">

# 🔤 arabic-nlp-toolkit

**Production-ready Arabic NLP for real-world text**

*Dialects · Sentiment · NER · Keywords · Morphology · Web Demo*

[![PyPI](https://img.shields.io/pypi/v/arabic-nlp-toolkit?style=for-the-badge&color=blue)](https://pypi.org/project/arabic-nlp-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/arabic-nlp-toolkit?style=for-the-badge)](https://pypi.org/project/arabic-nlp-toolkit/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-301%2B-success?style=for-the-badge)](./tests/)

</div>

---

## At a glance

| | |
|---|---|
| **Version** | 2.0.0 |
| **Author** | [Omar S. M. Abdelfatah](https://www.omarsharaf.me) |
| **License** | MIT |
| **Python** | 3.9 – 3.12 |
| **Core dependency** | `pydantic` only |
| **Test coverage** | ~89% |

---

## What it does

Built for **MSA and dialectal Arabic** — Egyptian, Gulf, Levantine, Maghrebi, Iraqi, Yemeni, Sudanese — plus social media and Franco-Arabic chat text.

```python
from arabic_nlp import ArabicNLP

nlp = ArabicNLP()
nlp.detect_dialect("ازيك عامل ايه؟")      # Egyptian
nlp.sentiment("المنتج رائع جداً!")        # Positive
nlp.extract_keywords("الذكاء الاصطناعي")  # Ranked keywords
doc = nlp.analyze_document("نص كامل")   # JSON-ready export
```

---

## Module map

```
dialects · sentiment · tokenizer · ner · normalization
transliteration · stopwords · morphology · pos · stemmer
keywords · profiling · utils · cli · webapp
```

---

## Links

- **GitHub:** https://github.com/OmarSharaf/arabic-nlp-toolkit
- **PyPI:** https://pypi.org/project/arabic-nlp-toolkit/
- **Web demo:** `python webapp/app.py` → http://127.0.0.1:8765
- **Social preview:** [assets/social-preview.svg](../assets/social-preview.svg)

---

## Install

```bash
pip install arabic-nlp-toolkit
pip install arabic-nlp-toolkit[web]   # interactive demo
pip install arabic-nlp-toolkit[dev]   # development
```

---

<div align="center">

**Built with care from Egypt 🇪🇬**

</div>
