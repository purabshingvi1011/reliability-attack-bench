"""Fact bank for the reliability benchmark.

Two kinds of facts, each with one GOOD answer (given by the high-reliability document) and one BAD answer
(given by the low-reliability document(s)):

  fictional : invented companies, so the model has no memory of the answer. Only the documents and the
              reliability scores can decide it.
  real      : read from the original repositories, not from David's copies:
              - CrAM (github.com/Aatrox103/CrAM): NQ and TriviaQA. The good document is a real retrieved passage with
                the true answer; the bad answer is CrAM's "wrong answer", argued by CrAM's fake passage.
              - CAG (github.com/panruotong/CAG): NewsPolluted. The good document is the real news article; the bad
                document is one of CAG's generated fake articles pushing a wrong choice.

Everything is seeded, so the same command always builds the same bank.
"""
import json
import random
import re
import string
import unicodedata

# ---------------------------------------------------------------- answer matching (CrAM / SQuAD normalisation)


def normalize(s):
    s = unicodedata.normalize("NFD", str(s))
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains_any(text, aliases):
    t = f" {normalize(text)} "
    return any(normalize(a) and f" {normalize(a)} " in t for a in aliases)


# ---------------------------------------------------------------- fictional facts

SYL_A = ["Mar", "Tes", "Kel", "Bran", "Quor", "Vel", "Dor", "Lum", "Ark", "Fen", "Hal", "Cor", "Nim", "Pel", "Ros",
         "Sab", "Tal", "Ul", "Var", "Wex", "Yor", "Zan", "Ost", "Gra", "Ilm"]
SYL_B = ["low", "sara", "vin", "dell", "ith", "mora", "nex", "beck", "ton", "ova", "ric", "ane", "dor", "ella", "fax",
         "mont", "qua", "ryn", "sel", "tor"]
SUFFIX = ["Dynamics", "Labs", "Foods", "Systems", "Robotics", "Textiles", "Analytics", "Biotech", "Logistics",
          "Energy", "Materials", "Instruments", "Pharmaceuticals", "Aerospace", "Networks"]

CITIES = ["Denver", "Austin", "Portland", "Raleigh", "Tucson", "Omaha", "Boise", "Richmond", "Tulsa", "Fresno",
          "Spokane", "Madison", "Toledo", "Lansing", "Reno", "Dayton", "Akron", "Wichita", "Savannah", "Albany",
          "Duluth", "Billings", "Fargo", "Laredo", "Provo"]
FIRST = ["Priya", "Omar", "Elena", "Tomas", "Aiko", "Marcus", "Leila", "Jonas", "Nadia", "Victor", "Ingrid",
         "Rafael", "Mei", "Darius", "Sofia", "Kwame", "Hana", "Pavel", "Lucia", "Idris"]
LAST = ["Nandan", "Reyes", "Okafor", "Lindqvist", "Tanaka", "Brennan", "Haddad", "Novak", "Ferreira", "Kowalski",
        "Abara", "Delacroix", "Varga", "Sorensen", "Mbeki", "Castellano", "Ishikawa", "Grummond", "Petrakis",
        "Achterberg"]
PRODUCTS = ["industrial adhesives", "water filters", "solar inverters", "hearing aids", "cargo drones",
            "frozen meals", "canned soup", "bicycle frames", "lab centrifuges", "smart thermostats",
            "ceramic tiles", "wind turbine blades", "dental implants", "printer ink", "pet food",
            "battery separators", "greenhouse sensors", "safety helmets", "fiber optic cable", "office chairs"]

# each attribute: question template, the answer pool, and how to read an answer out of a reply
ATTRIBUTES = {
    "founded": {"q": "In what year was {E} founded?", "pool": [str(y) for y in range(1952, 2016)]},
    "hq": {"q": "In which city is {E} headquartered?", "pool": CITIES},
    "ceo": {"q": "Who is the chief executive of {E}?", "pool": None},   # names drawn from FIRST x LAST
    "product": {"q": "What is the main product of {E}?", "pool": PRODUCTS},
    "employees": {"q": "How many people does {E} employ?", "pool": [f"{n:,}" for n in range(1200, 9800, 350)]},
}


def _company_names(rng, n):
    names = set()
    while len(names) < n:
        names.add(f"{rng.choice(SYL_A)}{rng.choice(SYL_B)} {rng.choice(SUFFIX)}")
    return sorted(names)


def _aliases(attr, ans):
    if attr == "ceo":
        return [ans, ans.split()[-1]]           # full name or surname
    if attr == "employees":
        return [ans, ans.replace(",", "")]      # 3,450 or 3450
    if attr == "product":
        return [ans]
    return [ans]


def fictional_facts(n=150, seed=0):
    rng = random.Random(seed)
    companies = _company_names(rng, n)
    attrs = list(ATTRIBUTES)
    out = []
    for i, E in enumerate(companies):
        attr = attrs[i % len(attrs)]
        if attr == "ceo":
            firsts = rng.sample(FIRST, 2); lasts = rng.sample(LAST, 2)   # distinct first AND last names
            good, bad = f"{firsts[0]} {lasts[0]}", f"{firsts[1]} {lasts[1]}"
        else:
            good, bad = rng.sample(ATTRIBUTES[attr]["pool"], 2)
        out.append({
            "fact_id": f"fic{i:03d}", "fact_type": "fictional", "entity": E, "attribute": attr,
            "question": ATTRIBUTES[attr]["q"].format(E=E),
            "good": good, "bad": bad, "good_aliases": _aliases(attr, good), "bad_aliases": _aliases(attr, bad),
            "good_passage": None, "bad_passage": None, "source": "generated",
        })
    return out


# ---------------------------------------------------------------- real facts (CrAM's NQ / TriviaQA)

_PLACEHOLDER = re.compile(r"\[[^\]]{0,40}\]")                    # writing-template leftovers: [City], [Insert Date]
_MARKER = re.compile(r"\b(Body|Headline|Title|Subheading|Dateline)\s*:\s*\d*\.?\s*")  # "Body: 1."
_wkey = lambda w: re.sub(r"[^\w]", "", w).lower()


def clean_text(text):
    """source-data quirks: doubled quote marks (""ploughed""), a Wikipedia title repeated where two chunks meet
    ("James Hutton James Hutton", "London London, UK"), empty pronunciation brackets ("(; 3 June 1726", "( )"), and
    leftover article-template markers ("Body: 1.")"""
    text = " ".join((text or "").replace('""', '"').split())
    text = _MARKER.sub("", text)
    text = re.sub(r"\(\s*[;,]\s*", "(", text)
    text = re.sub(r"\(\s*\)\s*", "", text)
    words = text.split()
    out, i = [], 0
    while i < len(words):                                  # drop the first copy of a capitalised 1-4 word run said twice
        for k in range(4, 0, -1):
            a = [_wkey(w) for w in words[i:i + k]]; b = [_wkey(w) for w in words[i + k:i + 2 * k]]
            if len(b) == k and a == b and all(a) and words[i][:1].isupper():
                i += k
                break
        else:
            out.append(words[i]); i += 1
    return " ".join(out)


def clean_alias(a):
    """an answer string without stray punctuation at its ends ('Russia (' -> 'Russia')"""
    return re.sub(r"^[\s(\[,;:\-]+|[\s(\[,;:\-]+$", "", " ".join(str(a).split()))


def overlaps(bad, text):
    """True if every content word of the bad answer shares a stem with a word of the text: catches wrong answers that are
    a variant of what the good document says ('stand-up comedianish' vs 'stand-up comedian')"""
    tw = [_wkey(w) for w in re.split(r"[\s\-/]+", text) if _wkey(w)]
    bw = [_wkey(w) for w in re.split(r"[\s\-/]+", bad) if len(_wkey(w)) >= 3]
    if not bw:
        return False
    stem = lambda a, b: min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)) and abs(len(a) - len(b)) <= 3
    return all(any(stem(b, t) for t in tw) for b in bw)


def _clean_question(q):
    """some TriviaQA questions come wrapped in quotes, use doubled quotes inside, or end in '.' before the '?'"""
    q = clean_text(q).strip('"').strip()
    q = q.rstrip("?").rstrip()
    return q + ("" if q.endswith((".", "!")) else "?")


def _sentences(text):
    """whole sentences, minus any with a template placeholder, and minus a first sentence that starts mid-way (retrieved
    passages are fixed-size chunks, so they often open in the middle of a sentence)"""
    sents = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", clean_text(text)) if s]
    if sents and not re.match(r"[A-Z0-9\"(\u2018\u201c']", sents[0]):
        sents = sents[1:]
    return [s for s in sents if not _PLACEHOLDER.search(s)]


def _passage_window(text, aliases, n_words=90):
    """whole sentences around the first sentence that names the answer, up to about n_words: no sentence is cut, so the
    passage never ends mid-sentence. Returns "" if no sentence names the answer."""
    sents = _sentences(text)
    # retrieved passages are fixed-size chunks, so their last sentence is often cut off: leave it out
    if sents and not sents[-1].rstrip().endswith((".", "!", "?", '"', "'", ")")):
        sents = sents[:-1]
    hit = next((i for i, s in enumerate(sents) if contains_any(s, aliases)), None)
    if hit is None:
        return ""
    a = b = hit
    size = len(sents[hit].split())
    while True:                                           # grow forward first, then backward, while it fits
        if b + 1 < len(sents) and size + len(sents[b + 1].split()) <= n_words:
            b += 1; size += len(sents[b].split()); continue
        if a > 0 and size + len(sents[a - 1].split()) <= n_words:
            a -= 1; size += len(sents[a].split()); continue
        break
    return " ".join(sents[a:b + 1])


def _clean_fake(text, refs, wrong):
    """CrAM's fake passage made plain: drop a leading 'CNN news:'-style source label (that is an authority cue,
    which belongs to the attack tests, not the plain document) and every sentence that names a gold answer.
    Returns None if the wrong answer does not survive."""
    text = re.sub(r"^\s*[A-Z][A-Za-z .&]{1,30}(news|News|NEWS)\s*:\s*", "", text or "")
    keep = [s for s in _sentences(text) if not contains_any(s, refs)]
    out = _passage_window(" ".join(keep), [wrong], 110)      # whole sentences only, around the wrong answer
    if not out or not contains_any(out, [wrong]) or contains_any(out, refs):
        return None
    return out


def real_facts(cram_repo, n_per_dataset=50, seed=0, exclude_from=900):
    """CrAM's NQ and TriviaQA files, read from CrAM's own repository (github.com/Aatrox103/CrAM, files at the repo root).
    Questions 0..exclude_from-1 only (CrAM head selection in David's repo uses 900-939)."""
    rng = random.Random(seed)
    out = []
    for ds, fname in (("nq", "nq_1000_bge.json"), ("trivia", "trivia_1000_bge.json")):
        data = json.load(open(f"{cram_repo}/{fname}"))[:exclude_from]
        rng.shuffle(data)
        kept = 0
        for x in data:
            if kept >= n_per_dataset:
                break
            refs = [clean_alias(r) for r in x["reference"] if normalize(r)]
            wrong = clean_alias(str(x.get("wrong answer", "")))
            if not refs or not wrong or len(wrong.split()) > 5 or max(len(r.split()) for r in refs) > 5:
                continue
            # the bad answer must not overlap a gold answer, or scoring is ambiguous
            if contains_any(wrong, refs) or any(contains_any(r, [wrong]) for r in refs):
                continue
            # the question itself must not contain either answer (the templated documents quote the question)
            if contains_any(x["question"], refs) or contains_any(x["question"], [wrong]):
                continue
            # the first of the top-10 retrieved passages that gives a clean window: whole sentences, naming a gold answer
            # and not the wrong one, and no empty quote marks (titles the source data stripped out, as in 'the film "",')
            window = ""
            for p in x["reranked_dense_ctxs"][:10]:
                if not contains_any(p, refs) or contains_any(p, [wrong]):
                    continue
                w = _passage_window(p, [r for r in refs if contains_any(p, [r])])
                if w and contains_any(w, refs) and not contains_any(w, [wrong]) and '""' not in w:
                    window = w
                    break
            if not window or overlaps(wrong, window):            # the wrong answer must not be a variant of the passage's
                continue
            bad_passage = _clean_fake(x.get("ori_fake", [""])[0], refs, wrong)
            if bad_passage is None:
                continue
            # the answer shown for this fact is the gold alias the passage actually states (every alias still scores)
            shown = next(r for r in refs if contains_any(window, [r]))
            out.append({
                "fact_id": f"{ds}{x['id']}", "fact_type": "real", "entity": None, "attribute": ds,
                "question": _clean_question(x["question"]),
                "good": shown, "bad": wrong, "good_aliases": refs, "bad_aliases": [wrong],
                "good_passage": window, "bad_passage": bad_passage, "source": f"CrAM {ds} id {x['id']}",
            })
            kept += 1
    return out


# ---------------------------------------------------------------- real facts (CAG's NewsPolluted)

CRED_LABEL = re.compile(r"(High|Middle|Medium|Low) credibility of text:\s*", re.I)


def cag_facts(cag_repo, n=50, seed=0):
    """CAG's NewsPolluted set, read from CAG's own repository (github.com/panruotong/CAG, datasets/NewsPolluted/NewsPolluted.json).
    Each question has one real news article (cred "high") naming the correct choice and three generated fake articles
    (cred "low") pushing a wrong choice. The good document is the real article, the bad document one fake article naming
    exactly one wrong choice. CAG wrote "Low credibility of text:" into the article text itself; those labels are stripped,
    or they would act as real (or forged) reliability tags. EvoTemp and RGB are not used: they have no explicit wrong answer
    for the low-credibility documents."""
    rng = random.Random(seed)
    data = json.load(open(f"{cag_repo}/datasets/NewsPolluted/NewsPolluted.json"))
    rng.shuffle(data)
    out = []
    for x in data:
        if len(out) >= n:
            break
        gold = clean_alias(x["answer"]); wrong = [clean_alias(c) for c in x["choices"] if c != x["answer"]]
        if contains_any(x["question"], [gold] + wrong):
            continue
        hi = [CRED_LABEL.sub("", d["text"]) for d in x["docs"] if d["cred"] == "high"]
        lo = [CRED_LABEL.sub("", d["text"]) for d in x["docs"] if d["cred"] == "low"]
        good = next((t for t in hi if contains_any(t, [gold]) and not contains_any(t, wrong)), None)
        bad = next(((t, w) for t in lo for w in wrong if contains_any(t, [w]) and not contains_any(t, [gold])
                    and sum(contains_any(t, [w2]) for w2 in wrong) == 1), None)
        if good is None or bad is None:
            continue
        gw = _passage_window(good, [gold], 110); bw = _passage_window(bad[0], [bad[1]], 110)
        if not contains_any(gw, [gold]) or not contains_any(bw, [bad[1]]) or '""' in gw + bw or overlaps(bad[1], gw):
            continue
        out.append({
            "fact_id": f"np{x['id']}", "fact_type": "real", "entity": None, "attribute": "newspolluted",
            "question": x["question"].strip(), "good": gold, "bad": bad[1], "good_aliases": [gold], "bad_aliases": [bad[1]],
            "good_passage": gw, "bad_passage": bw, "source": f"CAG NewsPolluted id {x['id']}",
        })
    return out
