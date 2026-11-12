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

def _passage_window(text, aliases, n_words=90):
    """a window of about n_words around the first mention of the answer"""
    words = text.split()
    norm_words = [normalize(w) for w in words]
    first = normalize(aliases[0]).split()[0] if aliases and normalize(aliases[0]) else ""
    idx = next((k for k, w in enumerate(norm_words) if w == first), 0)
    a = max(0, idx - n_words // 2)
    return " ".join(words[a:a + n_words])


def _clean_fake(text, refs, wrong):
    """CrAM's fake passage made plain: drop a leading 'CNN news:'-style source label (that is an authority cue,
    which belongs to the attack tests, not the plain document) and every sentence that names a gold answer.
    Returns None if the wrong answer does not survive."""
    text = re.sub(r"^\s*[A-Z][A-Za-z .&]{1,30}(news|News|NEWS)\s*:\s*", "", text or "")
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    keep = [s for s in sents if not contains_any(s, refs)]
    out = " ".join(keep).strip()
    if not out or not contains_any(out, [wrong]) or contains_any(out, refs):
        return None
    words = out.split()
    return " ".join(words[:110])


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
            refs = [r for r in x["reference"] if normalize(r)]
            wrong = str(x.get("wrong answer", "")).strip()
            if not refs or not wrong or len(wrong.split()) > 5 or max(len(r.split()) for r in refs) > 5:
                continue
            # the bad answer must not overlap a gold answer, or scoring is ambiguous
            if contains_any(wrong, refs) or any(contains_any(r, [wrong]) for r in refs):
                continue
            # the question itself must not contain either answer (the templated documents quote the question)
            if contains_any(x["question"], refs) or contains_any(x["question"], [wrong]):
                continue
            passage = next((p for p in x["reranked_dense_ctxs"][:10] if contains_any(p, refs) and not contains_any(p, [wrong])), None)
            if passage is None:
                continue
            window = _passage_window(passage, [r for r in refs if contains_any(passage, [r])])
            if not contains_any(window, refs) or contains_any(window, [wrong]):
                continue
            bad_passage = _clean_fake(x.get("ori_fake", [""])[0], refs, wrong)
            if bad_passage is None:
                continue
            out.append({
                "fact_id": f"{ds}{x['id']}", "fact_type": "real", "entity": None, "attribute": ds,
                "question": x["question"].strip().rstrip("?") + "?",
                "good": refs[0], "bad": wrong, "good_aliases": refs, "bad_aliases": [wrong],
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
        gold = x["answer"]; wrong = [c for c in x["choices"] if c != gold]
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
        if not contains_any(gw, [gold]) or not contains_any(bw, [bad[1]]):
            continue
        out.append({
            "fact_id": f"np{x['id']}", "fact_type": "real", "entity": None, "attribute": "newspolluted",
            "question": x["question"].strip(), "good": gold, "bad": bad[1], "good_aliases": [gold], "bad_aliases": [bad[1]],
            "good_passage": gw, "bad_passage": bw, "source": f"CAG NewsPolluted id {x['id']}",
        })
    return out
