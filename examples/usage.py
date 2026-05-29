"""
arabic-nlp-toolkit — Full Usage Examples
=========================================
Run this file directly to see all features in action:
    python examples/usage.py
"""

from arabic_nlp import ArabicNLP
from arabic_nlp.models import Dialect, Script
from arabic_nlp.stemmer import ArabicStemmer, StemmerMode
from arabic_nlp.morphology import ArabicMorphAnalyzer
from arabic_nlp.pos import ArabicPOSTagger
from arabic_nlp.utils import get_statistics, get_readability, compute_similarity


def separator(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print('═' * 60)


def main():
    from arabic_nlp.models import ArabicNLPConfig

    nlp = ArabicNLP(
        config=ArabicNLPConfig(
            extract_keywords_on_analyze=False,
            profile_on_analyze=False,
        )
    )

    # ─────────────────────────────────────────────
    # 1. DIALECT DETECTION
    # ─────────────────────────────────────────────
    separator("1️⃣  DIALECT DETECTION — كشف اللهجة")

    samples = [
        ("Egyptian",   "ازيك عامل ايه النهارده يا صاحبي؟"),
        ("Gulf",       "شلون حالك اليوم؟ وين رحت البارحة؟"),
        ("Levantine",  "شو بدك هلق؟ رح روح عالبيت"),
        ("Maghrebi",   "واش راك؟ بزاف ناس هنا"),
        ("MSA",        "إن الذي يجتهد في عمله سينجح بإذن الله"),
    ]

    for label, text in samples:
        result = nlp.detect_dialect(text)
        print(f"\n  [{label}]")
        print(f"  Text:       {text}")
        print(f"  Dialect:    {result.dialect.value}  ({result.dialect_name_ar})")
        print(f"  Confidence: {result.confidence:.1%}")

    # ─────────────────────────────────────────────
    # 2. SENTIMENT ANALYSIS
    # ─────────────────────────────────────────────
    separator("2️⃣  SENTIMENT ANALYSIS — تحليل المشاعر")

    reviews = [
        "المنتج ده رائع جداً وبحبه أوي! أنصح بيه بشدة",
        "الخدمة وحشة جداً ومش راضي عنها خالص، مهدر للوقت",
        "المنتج عادي ومعقول، لا ممتاز ولا سيء",
        "مش كويس خالص، مضيعة للفلوس",
    ]

    for review in reviews:
        r = nlp.sentiment(review)
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐", "mixed": "😕"}
        print(f"\n  {emoji.get(r.label.value, '')} [{r.label.value.upper():10}] {r.score:.0%}")
        print(f"  \"{review[:50]}...\"" if len(review) > 50 else f"  \"{review}\"")

    # ─────────────────────────────────────────────
    # 3. TOKENIZATION
    # ─────────────────────────────────────────────
    separator("3️⃣  TOKENIZATION — التقطيع")

    text = "زار محمد صلاح @Ahly ملعب الأهلي في القاهرة! 🇪🇬"
    tokens = nlp.tokenize(text)
    print(f"\n  Text: {text}")
    print(f"\n  {'Token':20}  {'Type':15}  [start:end]")
    print("  " + "─" * 50)
    for tok in tokens:
        print(f"  {tok.text:20}  {tok.token_type.value:15}  [{tok.start}:{tok.end}]")

    # ─────────────────────────────────────────────
    # 4. NAMED ENTITY RECOGNITION
    # ─────────────────────────────────────────────
    separator("4️⃣  NAMED ENTITY RECOGNITION — التعرف على الكيانات")

    text = "زار محمد صلاح ملعب الأهلي في القاهرة يوم الجمعة بمبلغ 500000 جنيه"
    result = nlp.extract_entities(text)
    print(f"\n  Text: {text}")
    print(f"\n  Found {len(result.entities)} entities:\n")
    label_emoji = {"PERSON": "👤", "LOCATION": "📍", "ORGANIZATION": "🏢",
                   "DATE": "📅", "TIME": "⏰", "MONEY": "💰"}
    for ent in result.entities:
        emoji = label_emoji.get(ent.label.value, "•")
        print(f"  {emoji}  {ent.text:25}  {ent.label.value:15}  [{ent.confidence:.0%}]")

    # ─────────────────────────────────────────────
    # 5. NORMALIZATION
    # ─────────────────────────────────────────────
    separator("5️⃣  NORMALIZATION — التطبيع")

    tests = [
        ("Diacritics", "هَذَا نَصٌّ بِالتَّشْكِيل"),
        ("Tatweel",    "جميـــل جداااً"),
        ("Social",     "@ahmed شوف ده 🔥 http://example.com #مصر"),
    ]

    for label, text in tests:
        r = nlp.normalize(text, remove_urls=True, remove_mentions=True,
                          remove_hashtags=True, remove_emojis=True)
        print(f"\n  [{label}]")
        print(f"  Before: {r.original}")
        print(f"  After:  {r.normalized}")
        if r.changes:
            print(f"  Steps:  {', '.join(r.changes)}")

    # ─────────────────────────────────────────────
    # 6. TRANSLITERATION
    # ─────────────────────────────────────────────
    separator("6️⃣  TRANSLITERATION — النقحرة")

    translit_tests = [
        ("مرحبا كيف حالك",  Script.ARABIC,     Script.FRANCO),
        ("بسم الله",         Script.ARABIC,     Script.BUCKWALTER),
        ("ana mesh 3aref",  Script.FRANCO,     Script.ARABIC),
    ]

    for text, src, tgt in translit_tests:
        r = nlp.transliterate(text, source=src, target=tgt)
        print(f"\n  {src.value:12} → {tgt.value:12}")
        print(f"  Input:  {r.original}")
        print(f"  Output: {r.transliterated}")

    # ─────────────────────────────────────────────
    # 7. STEMMING
    # ─────────────────────────────────────────────
    separator("7️⃣  STEMMING — الجذع")

    stemmer = ArabicStemmer()
    words = ["الكتاب", "يكتبون", "مدرسة", "والمدرستين", "مدرستهم"]

    print(f"\n  {'Word':20}  {'Light Stem':20}  {'Aggressive Stem'}")
    print("  " + "─" * 65)
    for word in words:
        light = stemmer.stem(word, mode=StemmerMode.LIGHT).stem
        aggr  = stemmer.stem(word, mode=StemmerMode.AGGRESSIVE).stem
        print(f"  {word:20}  {light:20}  {aggr}")

    # ─────────────────────────────────────────────
    # 8. MORPHOLOGICAL ANALYSIS
    # ─────────────────────────────────────────────
    separator("8️⃣  MORPHOLOGICAL ANALYSIS — التحليل الصرفي")

    morph = ArabicMorphAnalyzer()
    words = ["الكاتب", "يكتبون", "مدرسة", "والكتابة"]

    for word in words:
        r = morph.analyze(word)
        print(f"\n  Word: {r.word}")
        print(f"    Stem:    {r.stem}")
        print(f"    Root:    {r.root or '—'}")
        print(f"    Class:   {r.word_class.value}")
        print(f"    Gender:  {r.gender.value}")
        print(f"    Number:  {r.number.value}")
        print(f"    Definite:{r.definiteness.value}")
        if r.prefixes: print(f"    Prefixes:{', '.join(r.prefixes)}")
        if r.suffixes: print(f"    Suffixes:{', '.join(r.suffixes)}")

    # ─────────────────────────────────────────────
    # 9. POS TAGGING
    # ─────────────────────────────────────────────
    separator("9️⃣  POS TAGGING — التشكيل النحوي")

    tagger = ArabicPOSTagger()
    sentences = [
        "ذهب الولد إلى المدرسة",
        "المنتج رائع جداً",
        "هل ذهبت إلى البيت؟",
    ]

    for sent in sentences:
        result = tagger.tag_sentence(sent)
        print(f"\n  Sentence: {sent}")
        for pos in result.tags:
            print(f"    {pos.token:15} → {pos.tag.value:15} [{pos.confidence:.0%}]")

    # ─────────────────────────────────────────────
    # 10. TEXT STATISTICS & READABILITY
    # ─────────────────────────────────────────────
    separator("🔟  TEXT STATISTICS — إحصائيات النص")

    text = """
    كتب الروائي المصري نجيب محفوظ أعمالاً خالدة تُعدّ من روائع الأدب العربي.
    حصل على جائزة نوبل للآداب عام 1988 وكان أول كاتب عربي يحظى بهذا التكريم.
    تميّزت أعماله بالعمق الفكري والتصوير الدقيق للمجتمع المصري.
    """

    stats = get_statistics(text)
    readability = get_readability(text)

    print(f"\n  Text (excerpt): {text.strip()[:80]}...")
    print(f"\n  Words:          {stats.total_words}")
    print(f"  Unique words:   {stats.unique_words}")
    print(f"  Sentences:      {stats.sentences}")
    print(f"  Avg word len:   {stats.avg_word_length:.1f} chars")
    print(f"  TTR:            {stats.type_token_ratio:.2%}")
    print(f"  Arabic ratio:   {stats.arabic_ratio:.1%}")
    print(f"  Readability:    {readability.difficulty_level}")
    print(f"  Recommendation: {readability.recommendation}")

    # ─────────────────────────────────────────────
    # 11. TEXT SIMILARITY
    # ─────────────────────────────────────────────
    separator("1️⃣1️⃣  TEXT SIMILARITY — تشابه النصوص")

    pairs = [
        ("المنتج رائع جداً",       "المنتج ممتاز جداً"),
        ("السماء زرقاء",            "البحر أزرق"),
        ("مرحبا بكم في مصر",       "مرحبا بكم في مصر"),
    ]

    for t1, t2 in pairs:
        sim = compute_similarity(t1, t2)
        print(f"\n  A: {t1}")
        print(f"  B: {t2}")
        print(f"  Jaccard: {sim.jaccard:.2f}  Cosine: {sim.cosine:.2f}  Overlap: {sim.overlap:.2f}")

    # ─────────────────────────────────────────────
    # 12. KEYWORDS & TEXT PROFILING (NEW)
    # ─────────────────────────────────────────────
    separator("1️⃣2️⃣  KEYWORDS & PROFILING — الكلمات المفتاحية وملف النص")

    kw_text = "الذكاء الاصطناعي يغير مستقبل التعليم والاقتصاد في العالم العربي"
    kw = nlp.extract_keywords(kw_text, top_n=5)
    print(f"\n  Text: {kw_text}")
    print("  Top keywords:")
    for k in kw.keywords:
        print(f"    {k.rank}. {k.text}  (score={k.score:.2f}, freq={k.frequency})")

    social = "ازيك يا صاحبي 😂 شوف #مصر https://example.com"
    profile = nlp.profile(social)
    print(f"\n  Social text: {social[:50]}...")
    print(f"  Register: {profile.text_register.value}  |  Quality: {profile.quality_score:.0%}")
    print(f"  Tip: {profile.recommendations[0]}")

    doc = nlp.analyze_document(kw_text, include_keywords=True, include_profile=True)
    print(f"\n  Document export summary: {doc.summary}")

    # ─────────────────────────────────────────────
    # 13. FULL PIPELINE
    # ─────────────────────────────────────────────
    separator("1️⃣3️⃣  FULL PIPELINE — التحليل الشامل")

    text = "زيارة محمد صلاح لملعب الأهلي في القاهرة كانت رائعة جداً"
    analysis = nlp.analyze(text)

    print(f"\n  Text: {text}\n")
    print(f"  🌍 Dialect:   {analysis['dialect'].dialect.value}  ({analysis['dialect'].dialect_name_ar})")
    print(f"  💬 Sentiment: {analysis['sentiment'].label.value}  [{analysis['sentiment'].score:.0%}]")
    print(f"  ✂️  Tokens:    {len(analysis['tokens'])}")
    print(f"  🏷️  Entities:  {len(analysis['entities'].entities)}")
    for ent in analysis['entities'].entities:
        print(f"     • {ent.text}  ({ent.label.value})")
    print(f"  🧹 Normalized: {analysis['normalized'].normalized}")
    print(f"\n  ⏱  Pipeline time: {analysis['pipeline_time_ms']:.1f} ms")

    print("\n\n✅ All examples completed successfully!\n")


if __name__ == "__main__":
    main()
