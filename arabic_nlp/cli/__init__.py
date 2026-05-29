"""
Command-Line Interface for arabic-nlp-toolkit
===============================================
Usage:
    arabic-nlp detect-dialect "ازيك عامل ايه النهارده؟"
    arabic-nlp sentiment "المنتج رائع جداً"
    arabic-nlp tokenize "مرحبا بالعالم"
    arabic-nlp ner "زار محمد صلاح القاهرة"
    arabic-nlp normalize "هَذَا نَصٌّ مُشَكَّل"
    arabic-nlp transliterate "مرحبا" --target franco
    arabic-nlp stem "الكتاب"
    arabic-nlp morph "يكتبون"
    arabic-nlp pos "ذهب الولد إلى المدرسة"
    arabic-nlp stats "أي نص عربي هنا"
    arabic-nlp analyze "النص الكامل للتحليل الشامل"
    arabic-nlp version
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any


def _get_nlp():
    """Lazy import to keep CLI startup fast."""
    from arabic_nlp import ArabicNLP

    return ArabicNLP()


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _print_header(title: str) -> None:
    width = 50
    print("\n" + "─" * width)
    print(f"  {title}")
    print("─" * width)


# ─────────────────────────────────────────────
# COMMAND HANDLERS
# ─────────────────────────────────────────────


def cmd_dialect(text: str, output: str) -> None:
    nlp = _get_nlp()
    result = nlp.detect_dialect(text)

    if output == "json":
        _print_json(
            {
                "dialect": result.dialect.value,
                "dialect_ar": result.dialect_name_ar,
                "confidence": result.confidence,
                "is_arabic": result.is_arabic,
                "all_scores": [
                    {"dialect": s.dialect.value, "confidence": s.confidence}
                    for s in result.all_scores
                ],
                "processing_time_ms": result.processing_time_ms,
            }
        )
    else:
        _print_header("🔍 Dialect Detection — كشف اللهجة")
        print(f"  Text:       {text[:60]}")
        print(f"  Dialect:    {result.dialect.value}  ({result.dialect_name_ar})")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Is Arabic:  {result.is_arabic}")
        print()
        print("  All scores:")
        for score in result.all_scores[:5]:
            bar = "█" * int(score.confidence * 20)
            print(f"    {score.dialect.value:12} {bar:20} {score.confidence:.1%}")
        print(f"\n  ⏱  {result.processing_time_ms:.2f} ms")


def cmd_sentiment(text: str, output: str) -> None:
    nlp = _get_nlp()
    result = nlp.sentiment(text)

    emoji = {"positive": "😊", "negative": "😞", "neutral": "😐", "mixed": "😕"}

    if output == "json":
        _print_json(
            {
                "label": result.label.value,
                "score": result.score,
                "positive_score": result.positive_score,
                "negative_score": result.negative_score,
                "neutral_score": result.neutral_score,
                "processing_time_ms": result.processing_time_ms,
            }
        )
    else:
        _print_header("💬 Sentiment Analysis — تحليل المشاعر")
        print(f"  Text:     {text[:60]}")
        print(f"  Label:    {emoji.get(result.label.value, '')} {result.label.value.upper()}")
        print(f"  Score:    {result.score:.1%}")
        print()
        print("  Breakdown:")
        for label, score in [
            ("Positive", result.positive_score),
            ("Negative", result.negative_score),
            ("Neutral", result.neutral_score),
        ]:
            bar = "█" * int(score * 20)
            print(f"    {label:10} {bar:20} {score:.1%}")
        print(f"\n  ⏱  {result.processing_time_ms:.2f} ms")


def cmd_tokenize(text: str, output: str) -> None:
    nlp = _get_nlp()
    tokens = nlp.tokenize(text)

    if output == "json":
        _print_json(
            [
                {
                    "text": t.text,
                    "type": t.token_type.value,
                    "start": t.start,
                    "end": t.end,
                    "is_arabic": t.is_arabic,
                }
                for t in tokens
            ]
        )
    else:
        _print_header("✂️  Tokenization — التقطيع")
        print(f"  Text:   {text[:60]}")
        print(f"  Tokens: {len(tokens)}\n")
        print(f"  {'#':3}  {'Token':20}  {'Type':15}  {'[start:end]'}")
        print("  " + "─" * 58)
        for i, tok in enumerate(tokens, 1):
            print(f"  {i:3}  {tok.text:20}  {tok.token_type.value:15}  [{tok.start}:{tok.end}]")


def cmd_ner(text: str, output: str) -> None:
    nlp = _get_nlp()
    result = nlp.extract_entities(text)

    if output == "json":
        _print_json(
            {
                "entity_count": len(result.entities),
                "entities": [
                    {
                        "text": e.text,
                        "label": e.label.value,
                        "start": e.start,
                        "end": e.end,
                        "confidence": e.confidence,
                    }
                    for e in result.entities
                ],
                "processing_time_ms": result.processing_time_ms,
            }
        )
    else:
        _print_header("🏷️  Named Entity Recognition — التعرف على الكيانات")
        print(f"  Text: {text[:60]}")
        print(f"  Found {len(result.entities)} entities\n")
        label_emoji = {
            "PERSON": "👤",
            "LOCATION": "📍",
            "ORGANIZATION": "🏢",
            "DATE": "📅",
            "TIME": "⏰",
            "MONEY": "💰",
            "EVENT": "🎯",
        }
        for ent in result.entities:
            emoji = label_emoji.get(ent.label.value, "•")
            print(f"  {emoji} {ent.text:25}  {ent.label.value:15}  ({ent.confidence:.0%})")
        print(f"\n  ⏱  {result.processing_time_ms:.2f} ms")


def cmd_normalize(text: str, output: str, social: bool = False) -> None:
    nlp = _get_nlp()
    if social:
        result = nlp.normalizer.clean_social(text)
    else:
        result = nlp.normalize(text)

    if output == "json":
        _print_json(
            {
                "original": result.original,
                "normalized": result.normalized,
                "changes": result.changes,
                "was_changed": result.was_changed,
                "processing_time_ms": result.processing_time_ms,
            }
        )
    else:
        _print_header("🧹 Normalization — التطبيع")
        print(f"  Original:   {result.original}")
        print(f"  Normalized: {result.normalized}")
        print(f"  Changed:    {result.was_changed}")
        if result.changes:
            print(f"  Steps:      {', '.join(result.changes)}")
        print(f"\n  ⏱  {result.processing_time_ms:.2f} ms")


def cmd_transliterate(text: str, target: str, source: str, output: str) -> None:
    from arabic_nlp.models import Script

    nlp = _get_nlp()

    script_map = {
        "franco": Script.FRANCO,
        "buckwalter": Script.BUCKWALTER,
        "arabic": Script.ARABIC,
        "ala_lc": Script.ALA_LC,
    }

    src = script_map.get(source.lower(), Script.ARABIC)
    tgt = script_map.get(target.lower(), Script.FRANCO)

    result = nlp.transliterate(text, source=src, target=tgt)

    if output == "json":
        _print_json(
            {
                "original": result.original,
                "transliterated": result.transliterated,
                "source_script": result.source_script.value,
                "target_script": result.target_script.value,
                "processing_time_ms": result.processing_time_ms,
            }
        )
    else:
        _print_header("🔤 Transliteration — النقحرة")
        print(f"  Original ({source}):       {result.original}")
        print(f"  Transliterated ({target}): {result.transliterated}")
        print(f"\n  ⏱  {result.processing_time_ms:.2f} ms")


def cmd_stem(text: str, mode: str, output: str) -> None:
    from arabic_nlp.stemmer import ArabicStemmer, StemmerMode

    stemmer = ArabicStemmer()
    m = StemmerMode.AGGRESSIVE if mode == "aggressive" else StemmerMode.LIGHT

    words = text.split()
    results = [stemmer.stem(w, mode=m) for w in words if w.strip()]

    if output == "json":
        _print_json(
            [
                {
                    "original": r.original,
                    "stem": r.stem,
                    "prefixes_removed": list(r.prefixes_removed),
                    "suffixes_removed": list(r.suffixes_removed),
                }
                for r in results
            ]
        )
    else:
        _print_header("✂️  Stemming — الجذع")
        print(f"  Mode: {mode}\n")
        print(f"  {'Original':20}  →  {'Stem':20}  Removed")
        print("  " + "─" * 60)
        for r in results:
            removed = " ".join(list(r.prefixes_removed) + list(r.suffixes_removed))
            print(f"  {r.original:20}  →  {r.stem:20}  [{removed}]")


def cmd_morph(word: str, output: str) -> None:
    from arabic_nlp.morphology import ArabicMorphAnalyzer

    analyzer = ArabicMorphAnalyzer()
    result = analyzer.analyze(word)

    if output == "json":
        _print_json(
            {
                "word": result.word,
                "stem": result.stem,
                "root": result.root,
                "pattern": result.pattern,
                "word_class": result.word_class.value,
                "gender": result.gender.value,
                "number": result.number.value,
                "definiteness": result.definiteness.value,
                "verb_tense": result.verb_tense.value,
                "is_verb": result.is_verb,
                "is_noun": result.is_noun,
                "prefixes": list(result.prefixes),
                "suffixes": list(result.suffixes),
                "confidence": result.confidence,
            }
        )
    else:
        _print_header("🔬 Morphological Analysis — التحليل الصرفي")
        print(f"  Word:         {result.word}")
        print(f"  Stem:         {result.stem}")
        print(f"  Root:         {result.root or '—'}")
        print(f"  Pattern:      {result.pattern or '—'}")
        print(f"  Class:        {result.word_class.value}")
        print(f"  Gender:       {result.gender.value}")
        print(f"  Number:       {result.number.value}")
        print(f"  Definiteness: {result.definiteness.value}")
        if result.is_verb:
            print(f"  Tense:        {result.verb_tense.value}")
            print(f"  Voice:        {result.verb_voice.value}")
        if result.prefixes:
            print(f"  Prefixes:     {' + '.join(result.prefixes)}")
        if result.suffixes:
            print(f"  Suffixes:     {' + '.join(result.suffixes)}")
        print(f"  Confidence:   {result.confidence:.0%}")


def cmd_pos(text: str, output: str) -> None:
    from arabic_nlp.pos import ArabicPOSTagger

    tagger = ArabicPOSTagger()
    result = tagger.tag_sentence(text)

    if output == "json":
        _print_json(
            [
                {
                    "token": r.token,
                    "tag": r.tag.value,
                    "fine_tag": r.fine_tag,
                    "confidence": r.confidence,
                }
                for r in result.tags
            ]
        )
    else:
        _print_header("🏷️  POS Tagging — التشكيل النحوي")
        print(f"  Text: {text[:60]}\n")
        print(f"  {'Token':20}  {'Tag':15}  {'Confidence'}")
        print("  " + "─" * 50)
        for r in result.tags:
            print(f"  {r.token:20}  {r.tag.value:15}  {r.confidence:.0%}")
        print(f"\n  ⏱  {result.processing_time_ms:.2f} ms")


def cmd_stats(text: str, output: str) -> None:
    from arabic_nlp.utils import get_readability, get_statistics

    stats = get_statistics(text)
    readability = get_readability(text)

    if output == "json":
        _print_json(
            {
                "total_words": stats.total_words,
                "unique_words": stats.unique_words,
                "arabic_words": stats.arabic_words,
                "sentences": stats.sentences,
                "avg_word_length": stats.avg_word_length,
                "type_token_ratio": stats.type_token_ratio,
                "arabic_ratio": stats.arabic_ratio,
                "has_diacritics": stats.has_diacritics,
                "readability": {
                    "level": readability.difficulty_level,
                    "score": readability.difficulty_score,
                    "recommendation": readability.recommendation,
                },
                "most_common": stats.most_common_words[:5],
            }
        )
    else:
        _print_header("📊 Text Statistics — إحصائيات النص")
        print(f"  Total words:      {stats.total_words}")
        print(f"  Unique words:     {stats.unique_words}")
        print(f"  Arabic words:     {stats.arabic_words}")
        print(f"  Sentences:        {stats.sentences}")
        print(f"  Avg word length:  {stats.avg_word_length:.1f} chars")
        print(f"  Avg sent length:  {stats.avg_sentence_length:.1f} words")
        print(f"  Type-Token Ratio: {stats.type_token_ratio:.2%}")
        print(f"  Arabic ratio:     {stats.arabic_ratio:.1%}")
        print(f"  Has diacritics:   {stats.has_diacritics}")
        print()
        print(f"  Readability:      {readability.difficulty_level.upper()}")
        print(f"  Recommendation:   {readability.recommendation}")
        if stats.most_common_words:
            print()
            print("  Most common words:")
            for word, count in stats.most_common_words[:5]:
                print(f"    {word:20}  ×{count}")


def cmd_keywords(text: str, output: str, top_n: int) -> None:
    nlp = _get_nlp()
    result = nlp.extract_keywords(text, top_n=top_n)

    if output == "json":
        _print_json(
            {
                "keywords": [k.model_dump() for k in result.keywords],
                "dialect": result.dialect.value,
                "content_tokens": result.content_tokens,
                "processing_time_ms": result.processing_time_ms,
            }
        )
    else:
        _print_header("🔑 Keyword Extraction — استخراج الكلمات المفتاحية")
        print(f"  Text: {text[:60]}\n")
        print(f"  {'Rank':5}  {'Keyword':20}  {'Score':8}  {'Freq'}")
        print("  " + "─" * 50)
        for kw in result.keywords:
            print(f"  {kw.rank:5}  {kw.text:20}  {kw.score:.2f}     {kw.frequency}")
        print(f"\n  ⏱  {result.processing_time_ms:.2f} ms")


def cmd_profile(text: str, output: str) -> None:
    nlp = _get_nlp()
    result = nlp.profile(text)

    if output == "json":
        _print_json(result.model_dump())
    else:
        _print_header("📋 Text Profile — ملف النص")
        print(f"  Text:       {text[:60]}")
        print(f"  Register:   {result.text_register.value}  ({result.register_confidence:.0%})")
        print(f"  Dialect:    {result.dialect.value}  ({result.dialect_confidence:.0%})")
        print(f"  Quality:    {result.quality_score:.0%}")
        print(f"  Arabic:     {result.arabic_ratio:.0%}")
        print(f"  Words:      {result.word_count}  |  Sentences: {result.sentence_count}")
        flags = []
        if result.has_urls:
            flags.append("URLs")
        if result.has_hashtags:
            flags.append("hashtags")
        if result.has_emojis:
            flags.append("emojis")
        if result.has_repeated_chars:
            flags.append("repeated chars")
        if flags:
            print(f"  Flags:      {', '.join(flags)}")
        print("\n  Recommendations:")
        for rec in result.recommendations:
            print(f"    • {rec}")
        print(f"\n  ⏱  {result.processing_time_ms:.2f} ms")


def cmd_export(text: str, output: str, keywords: bool, profile: bool) -> None:
    nlp = _get_nlp()
    doc = nlp.analyze_document(
        text,
        include_keywords=keywords,
        include_profile=profile,
    )
    if output == "json":
        _print_json(doc.to_dict())
    else:
        _print_header("📄 Document Analysis — تحليل المستند")
        s = doc.summary
        print(f"  Dialect:    {s['dialect']} ({s['dialect_confidence']:.0%})")
        print(f"  Sentiment:  {s['sentiment']} ({s['sentiment_score']:.0%})")
        print(f"  Entities:   {s['entities']}")
        print(f"  Tokens:     {s['tokens']}")
        if s.get("keywords"):
            print(f"  Keywords:   {', '.join(s['keywords'])}")
        print(f"\n  ⏱  {s['pipeline_time_ms']:.1f} ms")
        print("\n  Use --output json for full structured export.")


def cmd_analyze(text: str, output: str) -> None:
    nlp = _get_nlp()
    analysis = nlp.analyze(text)

    if output == "json":
        _print_json(
            {
                "dialect": analysis["dialect"].dialect.value,
                "sentiment": analysis["sentiment"].label.value,
                "token_count": len(analysis["tokens"]),
                "entity_count": len(analysis["entities"].entities),
                "normalized": analysis["normalized"].normalized,
                "pipeline_time_ms": analysis["pipeline_time_ms"],
            }
        )
    else:
        _print_header("🔬 Full Pipeline Analysis — التحليل الشامل")
        d = analysis["dialect"]
        s = analysis["sentiment"]
        e = analysis["entities"]
        t = analysis["tokens"]
        n = analysis["normalized"]

        print(f"  Text:        {text[:60]}")
        print()
        print(f"  📍 Dialect:  {d.dialect.value} ({d.dialect_name_ar})  [{d.confidence:.0%}]")
        print(f"  💬 Sentiment:{s.label.value}  [{s.score:.0%}]")
        print(f"  ✂️  Tokens:   {len(t)}")
        print(f"  🏷️  Entities: {len(e.entities)}")
        if e.entities:
            for ent in e.entities[:3]:
                print(f"     • {ent.text}  ({ent.label.value})")
        print(f"  🧹 Normalized: {n.normalized[:60]}")
        print(f"\n  ⏱  Total: {analysis['pipeline_time_ms']:.1f} ms")


def cmd_version() -> None:
    from arabic_nlp._version import __version__, __version_info__

    print(f"arabic-nlp-toolkit v{__version__}")
    print(f"Version info: {__version_info__}")
    print("Author: Omar S. M. Abdelfatah <omar@omarsharaf.me>")
    print("GitHub: https://github.com/OmarSharaf/arabic-nlp-toolkit")


# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arabic-nlp",
        description="🔤 arabic-nlp-toolkit — Arabic NLP from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  arabic-nlp detect-dialect "ازيك عامل ايه النهارده؟"
  arabic-nlp sentiment "المنتج رائع جداً وأنصح به"
  arabic-nlp tokenize "مرحبا بالعالم العربي"
  arabic-nlp ner "زار محمد صلاح ملعب الأهلي في القاهرة"
  arabic-nlp normalize "هَذَا نَصٌّ مُشَكَّل"
  arabic-nlp transliterate "مرحبا" --target franco
  arabic-nlp stem "الكتاب يكتبون" --mode light
  arabic-nlp morph "الكاتب"
  arabic-nlp pos "ذهب الولد إلى المدرسة"
  arabic-nlp stats "أي نص عربي تريد تحليله"
  arabic-nlp analyze "النص الكامل للتحليل الشامل"
  arabic-nlp version
        """,
    )

    def output_arg(p):
        return p.add_argument(
            "--output",
            "-o",
            choices=["text", "json"],
            default="text",
            help="Output format (default: text)",
        )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # detect-dialect
    p = subparsers.add_parser("detect-dialect", aliases=["dialect"], help="Detect Arabic dialect")
    p.add_argument("text", help="Arabic text to analyze")
    output_arg(p)

    # sentiment
    p = subparsers.add_parser("sentiment", help="Analyze sentiment")
    p.add_argument("text", help="Arabic text to analyze")
    output_arg(p)

    # tokenize
    p = subparsers.add_parser("tokenize", aliases=["tok"], help="Tokenize text")
    p.add_argument("text", help="Arabic text to tokenize")
    output_arg(p)

    # ner
    p = subparsers.add_parser("ner", help="Named entity recognition")
    p.add_argument("text", help="Arabic text to analyze")
    output_arg(p)

    # normalize
    p = subparsers.add_parser("normalize", aliases=["norm"], help="Normalize Arabic text")
    p.add_argument("text", help="Arabic text to normalize")
    p.add_argument("--social", action="store_true", help="Apply social media cleanup")
    output_arg(p)

    # transliterate
    p = subparsers.add_parser("transliterate", aliases=["translit"], help="Transliterate text")
    p.add_argument("text", help="Text to transliterate")
    p.add_argument("--source", default="arabic", choices=["arabic", "franco", "buckwalter"])
    p.add_argument("--target", default="franco", choices=["franco", "buckwalter", "arabic"])
    output_arg(p)

    # stem
    p = subparsers.add_parser("stem", help="Stem Arabic words")
    p.add_argument("text", help="Text to stem (space-separated words)")
    p.add_argument("--mode", default="light", choices=["light", "aggressive"])
    output_arg(p)

    # morph
    p = subparsers.add_parser("morph", help="Morphological analysis")
    p.add_argument("word", help="Single Arabic word to analyze")
    output_arg(p)

    # pos
    p = subparsers.add_parser("pos", help="POS tagging")
    p.add_argument("text", help="Arabic sentence to tag")
    output_arg(p)

    # stats
    p = subparsers.add_parser("stats", help="Text statistics")
    p.add_argument("text", help="Arabic text to analyze")
    output_arg(p)

    # analyze
    p = subparsers.add_parser("analyze", help="Full pipeline analysis")
    p.add_argument("text", help="Arabic text to analyze")
    output_arg(p)

    # keywords
    p = subparsers.add_parser("keywords", aliases=["kw"], help="Extract keywords")
    p.add_argument("text", help="Arabic text to analyze")
    p.add_argument("--top", type=int, default=10, help="Number of keywords (default: 10)")
    output_arg(p)

    # profile
    p = subparsers.add_parser("profile", help="Profile text register and quality")
    p.add_argument("text", help="Arabic text to profile")
    output_arg(p)

    # export
    p = subparsers.add_parser("export", help="Structured document analysis (JSON-ready)")
    p.add_argument("text", help="Arabic text to analyze")
    p.add_argument("--keywords", action="store_true", help="Include keyword extraction")
    p.add_argument("--profile", action="store_true", help="Include text profile")
    output_arg(p)

    # version
    subparsers.add_parser("version", help="Show version info")

    return parser


def main() -> int:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    try:
        cmd = args.command

        if cmd in ("detect-dialect", "dialect"):
            cmd_dialect(args.text, args.output)
        elif cmd == "sentiment":
            cmd_sentiment(args.text, args.output)
        elif cmd in ("tokenize", "tok"):
            cmd_tokenize(args.text, args.output)
        elif cmd == "ner":
            cmd_ner(args.text, args.output)
        elif cmd in ("normalize", "norm"):
            cmd_normalize(args.text, args.output, getattr(args, "social", False))
        elif cmd in ("transliterate", "translit"):
            cmd_transliterate(args.text, args.target, args.source, args.output)
        elif cmd == "stem":
            cmd_stem(args.text, args.mode, args.output)
        elif cmd == "morph":
            cmd_morph(args.word, args.output)
        elif cmd == "pos":
            cmd_pos(args.text, args.output)
        elif cmd == "stats":
            cmd_stats(args.text, args.output)
        elif cmd == "analyze":
            cmd_analyze(args.text, args.output)
        elif cmd in ("keywords", "kw"):
            cmd_keywords(args.text, args.output, getattr(args, "top", 10))
        elif cmd == "profile":
            cmd_profile(args.text, args.output)
        elif cmd == "export":
            cmd_export(
                args.text,
                args.output,
                getattr(args, "keywords", False),
                getattr(args, "profile", False),
            )
        elif cmd == "version":
            cmd_version()

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
