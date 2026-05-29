"""
Arabic NLP Toolkit — Interactive Web Demo
Run: python webapp/app.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is on path when run directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arabic_nlp import ArabicNLP
from arabic_nlp._version import __version__
from arabic_nlp.models import Script

WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(
    title="Arabic NLP Toolkit",
    description="Interactive demo for arabic-nlp-toolkit",
    version=__version__,
)

nlp = ArabicNLP(lazy=False)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_ASSETS_DIR = _ROOT / "assets"
if _ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class TransliterateRequest(TextRequest):
    target: str = "franco"
    source: str = "arabic"


def _serialize(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, int, float, bool)):
        return {k: _serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return obj


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/api/project")
async def project_profile() -> dict[str, Any]:
    """Project metadata for the profile / about page."""
    return {
        "name": "arabic-nlp-toolkit",
        "version": __version__,
        "tagline": "Production-ready Arabic NLP for real-world text",
        "tagline_ar": "معالجة اللغة العربية الاحترافية للنصوص الحقيقية",
        "author": {
            "name": "Omar S. M. Abdelfatah",
            "website": "https://www.omarsharaf.me",
            "github": "https://github.com/OmarSharaf",
            "email": "omar@omarsharaf.me",
        },
        "links": {
            "github": "https://github.com/OmarSharaf/arabic-nlp-toolkit",
            "pypi": "https://pypi.org/project/arabic-nlp-toolkit/",
            "docs": "https://github.com/OmarSharaf/arabic-nlp-toolkit#readme",
        },
        "stats": {
            "dialects": 8,
            "modules": 14,
            "tests": "301+",
            "coverage": "89%",
            "python": "3.9 – 3.12",
        },
        "features": [
            {
                "icon": "🌍",
                "title": "Dialect detection",
                "title_ar": "كشف اللهجة",
                "desc": "8 Arabic dialects with confidence scores",
            },
            {
                "icon": "💬",
                "title": "Sentiment",
                "title_ar": "تحليل المشاعر",
                "desc": "Negation-aware, dialect-tuned polarity",
            },
            {
                "icon": "🏷️",
                "title": "NER",
                "title_ar": "الكيانات المسماة",
                "desc": "Persons, places, organizations, dates",
            },
            {
                "icon": "🔑",
                "title": "Keywords",
                "title_ar": "كلمات مفتاحية",
                "desc": "Ranked extraction with stopword filtering",
            },
            {
                "icon": "📋",
                "title": "Text profiling",
                "title_ar": "ملف النص",
                "desc": "Register, quality score, recommendations",
            },
            {
                "icon": "🔤",
                "title": "Transliteration",
                "title_ar": "النقل الحرفي",
                "desc": "Franco-Arabic, Buckwalter, Arabic",
            },
            {
                "icon": "🔬",
                "title": "Morphology & POS",
                "title_ar": "صرف ونحو",
                "desc": "Roots, patterns, stemming, POS tags",
            },
            {
                "icon": "📦",
                "title": "Document export",
                "title_ar": "تصدير JSON",
                "desc": "Pydantic v2 pipeline for APIs & ETL",
            },
        ],
        "install": "pip install arabic-nlp-toolkit",
        "license": "MIT",
    }


@app.post("/api/dialect")
async def api_dialect(req: TextRequest) -> dict:
    return _serialize(nlp.detect_dialect(req.text))


@app.post("/api/sentiment")
async def api_sentiment(req: TextRequest) -> dict:
    return _serialize(nlp.sentiment(req.text))


@app.post("/api/tokenize")
async def api_tokenize(req: TextRequest) -> dict:
    tokens = nlp.tokenize(req.text)
    return {"text": req.text, "count": len(tokens), "tokens": _serialize(tokens)}


@app.post("/api/ner")
async def api_ner(req: TextRequest) -> dict:
    return _serialize(nlp.extract_entities(req.text))


@app.post("/api/normalize")
async def api_normalize(req: TextRequest) -> dict:
    return _serialize(nlp.normalize(req.text))


@app.post("/api/transliterate")
async def api_transliterate(req: TransliterateRequest) -> dict:
    script_map = {
        "franco": Script.FRANCO,
        "buckwalter": Script.BUCKWALTER,
        "arabic": Script.ARABIC,
        "ala_lc": Script.ALA_LC,
    }
    src = script_map.get(req.source.lower(), Script.ARABIC)
    tgt = script_map.get(req.target.lower(), Script.FRANCO)
    return _serialize(nlp.transliterate(req.text, source=src, target=tgt))


@app.post("/api/keywords")
async def api_keywords(req: TextRequest) -> dict:
    return _serialize(nlp.extract_keywords(req.text, top_n=10))


@app.post("/api/profile")
async def api_profile(req: TextRequest) -> dict:
    return _serialize(nlp.profile(req.text))


@app.post("/api/analyze")
async def api_analyze(req: TextRequest) -> dict:
    doc = nlp.analyze_document(
        req.text,
        include_keywords=True,
        include_profile=True,
    )
    return doc.to_dict()


@app.post("/api/pos")
async def api_pos(req: TextRequest) -> dict:
    result = nlp.tag_pos(req.text)
    return {
        "tokens": result.tokens,
        "tags": _serialize(result.tags),
        "processing_time_ms": result.processing_time_ms,
    }


@app.post("/api/stats")
async def api_stats(req: TextRequest) -> dict:
    from arabic_nlp.utils import get_readability, get_statistics

    stats = get_statistics(req.text)
    readability = get_readability(req.text)
    return {
        "statistics": _serialize(stats),
        "readability": _serialize(readability),
    }


def _open_chrome(url: str) -> None:
    """Open URL in Google Chrome when available."""
    if os.environ.get("CI") or os.environ.get("ARABIC_NLP_NO_BROWSER"):
        return
    if sys.platform == "win32":
        for path in (
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
            Path.home() / "AppData/Local/Google/Chrome/Application/chrome.exe",
        ):
            if path.exists():
                subprocess.Popen(
                    [str(path), url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return
    webbrowser.open(url)


def main() -> None:
    import uvicorn

    host, port = "127.0.0.1", 8765
    url = f"http://{host}:{port}"

    print("\n  Arabic NLP Toolkit Web Demo")
    print(f"  Open: {url}\n")

    _open_chrome(url)

    uvicorn.run(app, host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
