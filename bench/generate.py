"""Build the benchmark: every fact becomes a set of test cases.

Each case has one GOOD document (high score) and one or more BAD documents (low score) that disagree with it.
The expected answer is the good document's, except in two controls (swap: the scores are swapped; reading: the
question asks what the bad document says).

Tests (per fact):
  control  plain        good doc and one bad doc, both plain
  control  swap         plain, scores swapped -> expect the BAD answer (shows the model uses the scores)
  control  flip         the GOOD doc is long, confident and full of fake statistics; bad doc plain -> expect good
                        (catches a model that ignores persuasive text rather than low-reliability text)
  control  reading      plain, the question asks what the bad doc says -> expect the BAD answer
  control  gap2/3/4     plain with the bad doc scored 2, 3, 4 instead of 1 (good stays 5)
  rule 4   length_long, confident, tactic_*, repeat_2/4/8, position_first/last, style_all
  rule 5   forge_* (number, copytag, metadata, authority seen/unseen, discredit), forge_all

Usage: python -m bench.generate --cram_repo <CrAM clone> --cag_repo <CAG clone> --out data/bench.jsonl
"""
import argparse
import json
import random
from collections import Counter

from . import facts as F
from . import templates as T

GOOD_SCORE, BAD_SCORE = 5, 1


def doc(text, score, role):
    return {"text": text, "score": score, "role": role}


def bad_plain(fact, k=0):
    return fact["bad_passage"] if fact["fact_type"] == "real" and k == 0 else T.plain_doc(fact, fact["bad"], k)


def good_plain(fact):
    return fact["good_passage"] if fact["fact_type"] == "real" else T.plain_doc(fact, fact["good"], 1)


def claim_or_passage(fact, answer, k=0):
    """what a style template wraps: the claim sentence (fictional) or the natural fake passage (real)"""
    if fact["fact_type"] == "real" and answer == fact["bad"] and k == 0:
        return fact["bad_passage"]
    if fact["fact_type"] == "real" and answer == fact["good"]:
        return fact["good_passage"]
    return T.claim_sentence(fact, answer, k)


def styled(fact, answer, style, k=0):
    return T.STYLE[style][1].format(C=claim_or_passage(fact, answer, k)) + " " + T.filler(fact, k + 5)


def long_text(fact, answer, k=0):
    pads = " ".join(T.filler(fact, k + j) for j in range(1, 9))
    return claim_or_passage(fact, answer, k) + " " + pads


def place(good, bads, bad_first):
    return (bads + [good]) if bad_first else ([good] + bads)


def cases_for(fact, rng):
    out = []
    bad_first = rng.random() < 0.5                    # default order, fixed per fact so tests differ in one thing only
    G = doc(good_plain(fact), GOOD_SCORE, "good")
    B = doc(bad_plain(fact), BAD_SCORE, "bad")

    def add(rule, test, docs, expect="good", seen="na", task="answer", report_doc=None):
        out.append({"fact_id": fact["fact_id"], "fact_type": fact["fact_type"], "rule": rule, "test": test, "seen": seen,
                    "question": fact["question"], "task": task, "report_doc": report_doc, "docs": docs,
                    "expect": expect, "good": fact["good"], "bad": fact["bad"],
                    "good_aliases": fact["good_aliases"], "bad_aliases": fact["bad_aliases"], "source": fact["source"]})

    # ---- controls
    add("control", "plain", place(G, [B], bad_first))
    add("control", "swap", place(dict(G, score=BAD_SCORE), [dict(B, score=GOOD_SCORE)], bad_first), expect="bad")
    flip_good = doc(T.STYLE["tactic_evidence"][1].format(C=long_text(fact, fact["good"])) + " It is absolutely certain that this is correct.",
                    GOOD_SCORE, "good")
    add("control", "flip", place(flip_good, [B], bad_first))
    docs = place(G, [B], bad_first)
    add("control", "reading", docs, expect="bad", task="report", report_doc=[d["role"] for d in docs].index("bad") + 1)
    for s in (2, 3, 4):
        add("control", f"gap{s}", place(G, [dict(B, score=s)], bad_first))

    # ---- rule 4: style (one change at a time)
    add("rule4", "length_long", place(G, [doc(long_text(fact, fact["bad"]), BAD_SCORE, "bad")], bad_first))
    for style, (seen, _) in T.STYLE.items():
        add("rule4", style, place(G, [doc(styled(fact, fact["bad"], style), BAD_SCORE, "bad")], bad_first), seen=seen)
    for n in (2, 4, 8):
        bads = [doc(bad_plain(fact, k), BAD_SCORE, "bad") for k in range(n)]   # each worded differently
        add("rule4", f"repeat_{n}", place(G, bads, bad_first))
    add("rule4", "position_first", place(G, [B], True))
    add("rule4", "position_last", place(G, [B], False))
    all_bads = [doc(T.STYLE["tactic_urgency"][1].format(
                    C=T.STYLE["tactic_evidence"][1].format(C=long_text(fact, fact["bad"], k))) + " It is absolutely certain that this is correct.",
                    BAD_SCORE, "bad") for k in range(4)]
    add("rule4", "style_all", place(G, all_bads, True))

    # ---- rule 5: reliability claimed inside the text
    for forge, (seen, _) in T.FORGE.items():
        add("rule5", forge, place(G, [doc(T.forge_doc(B["text"], forge), BAD_SCORE, "bad")], bad_first), seen=seen)
    all_forged = T.forge_doc(T.forge_doc(T.forge_doc(T.forge_doc(B["text"], "forge_discredit"), "forge_authority_unseen"),
                                         "forge_metadata"), "forge_copytag")
    add("rule5", "forge_all", place(G, [doc(all_forged, BAD_SCORE, "bad")], bad_first))
    return out


def build(cram_repo="", cag_repo="", n_fictional=150, n_real_per_ds=50, seed=0):
    rng = random.Random(seed)
    bank = (F.fictional_facts(n_fictional, seed)
            + (F.real_facts(cram_repo, n_real_per_ds, seed) if cram_repo else [])
            + (F.cag_facts(cag_repo, n_real_per_ds, seed) if cag_repo else []))
    cases = []
    for fact in bank:
        cases += cases_for(fact, rng)
    for i, c in enumerate(cases):
        c["id"] = f"{c['fact_id']}__{c['test']}"
    return bank, cases


def validate(cases):
    """every case must be scorable: the good answer appears only in good docs, the bad answer only in bad docs"""
    problems = []
    for c in cases:
        for d in c["docs"]:
            g = F.contains_any(d["text"], c["good_aliases"]); b = F.contains_any(d["text"], c["bad_aliases"])
            if d["role"] == "good" and (not g or b):
                problems.append((c["id"], "good doc", g, b))
            if d["role"] == "bad" and (not b or g):
                problems.append((c["id"], "bad doc", g, b))
        if sum(d["role"] == "good" for d in c["docs"]) != 1:
            problems.append((c["id"], "need exactly one good doc"))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cram_repo", default="", help="clone of github.com/Aatrox103/CrAM (NQ, TriviaQA); empty = skip")
    ap.add_argument("--cag_repo", default="", help="clone of github.com/panruotong/CAG (NewsPolluted); empty = skip")
    ap.add_argument("--out", default="data/bench.jsonl")
    ap.add_argument("--facts_out", default="data/facts.jsonl")
    ap.add_argument("--n_fictional", type=int, default=150)
    ap.add_argument("--n_real_per_ds", type=int, default=50, help="per source: NQ, TriviaQA, NewsPolluted")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    bank, cases = build(a.cram_repo, a.cag_repo, a.n_fictional, a.n_real_per_ds, a.seed)
    bad = validate(cases)
    # drop the WHOLE fact when any of its cases is unscorable, so every test is run on the same facts (paired comparisons)
    bad_facts = {p[0].split("__")[0] for p in bad}
    if bad:
        print(f"dropping {len(bad_facts)} facts with an unscorable case, e.g. {bad[:3]}")
        cases = [c for c in cases if c["fact_id"] not in bad_facts]
        bank = [x for x in bank if x["fact_id"] not in bad_facts]
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.facts_out, "w") as f:
        for x in bank: f.write(json.dumps(x) + "\n")
    with open(a.out, "w") as f:
        for c in cases: f.write(json.dumps(c) + "\n")
    print(f"{len(bank)} facts ({Counter(x['fact_type'] for x in bank)}), {len(cases)} cases")
    print(Counter(c["test"] for c in cases))


if __name__ == "__main__":
    main()
