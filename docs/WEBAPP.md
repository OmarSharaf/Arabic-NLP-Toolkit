# Web Demo

Interactive browser UI for testing all NLP features without writing code.

## Quick start

```bash
pip install -e ".[web]"
python webapp/app.py
```

Opens **http://127.0.0.1:8765** in Google Chrome (or your default browser).

## Pages

| Page | URL | Description |
|------|-----|-------------|
| **Playground** | `/#playground` | Test all NLP tools interactively |
| **Project Profile** | `/#profile` | Attractive project showcase, stats, features, author |

Use the top navigation tabs to switch between views.

To skip auto-open:

```bash
set ARABIC_NLP_NO_BROWSER=1    # Windows
export ARABIC_NLP_NO_BROWSER=1 # Linux/macOS
python webapp/app.py
```

## Features

| Tool | API endpoint |
|------|----------------|
| Full analysis | `POST /api/analyze` |
| Dialect | `POST /api/dialect` |
| Sentiment | `POST /api/sentiment` |
| NER | `POST /api/ner` |
| Tokenize | `POST /api/tokenize` |
| Normalize | `POST /api/normalize` |
| Keywords | `POST /api/keywords` |
| Profile | `POST /api/profile` |
| POS | `POST /api/pos` |
| Stats | `POST /api/stats` |
| Transliterate | `POST /api/transliterate` |

All endpoints accept JSON: `{"text": "نص عربي"}`.

## Development

```bash
pip install -e ".[web]"
uvicorn webapp.app:app --reload --host 127.0.0.1 --port 8765
```

Static files live in `webapp/static/`. The main page is `webapp/index.html`.

## Security note

The demo server is **not hardened for production**. Use only on localhost.
