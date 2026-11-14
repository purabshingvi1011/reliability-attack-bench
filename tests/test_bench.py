"""Checks that need no GPU: python -m pytest -q tests  (or python tests/test_bench.py)"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench import render as R
from bench import score as S
from bench.facts import contains_any, normalize
from bench.generate import build, validate
from bench.templates import FORGED_TAG

BENCH, _CASES = None, None


def cases():
    global _CASES
    if _CASES is None:
        _, _CASES = build(n_fictional=40)
    return _CASES


def test_counts_and_validity():
    cs = cases()
    assert not validate(cs)
    per_test = Counter(c["test"] for c in cs)
    assert len(set(per_test.values())) == 1, per_test        # every test has every fact
    assert len(per_test) == 30


def test_core_suite():
    from bench.generate import SUITES
    _, core = build(n_fictional=40, suite="core")
    assert {c["test"] for c in core} == set(SUITES["core"]) and len(SUITES["core"]) == 14
    assert len(core) == 40 * 14 and not validate(core)


def test_external_injections():
    from bench.templates import EXT
    _, cs = build(n_fictional=40, suite="core")
    by = {(c["fact_id"], c["test"]): c for c in cs}
    ext = [c for c in cs if c["test"] == "ext_injection"]
    assert len({c["ext_source"] for c in ext}) == len(EXT) == 16          # every trigger from the paper is used
    for c in ext:
        assert "2510.09023" in c["ext_source"]
        plain_bad = [d for d in by[(c["fact_id"], "plain")]["docs"] if d["role"] == "bad"][0]["text"]
        bad = [d for d in c["docs"] if d["role"] == "bad"][0]["text"]
        assert bad.startswith(plain_bad) and len(bad) > len(plain_bad)    # the plain bad doc plus the trigger, nothing else


def test_expectations():
    for c in cases():
        assert c["expect"] == ("bad" if c["test"] in ("swap", "reading") else "good")
        good = [d for d in c["docs"] if d["role"] == "good"][0]; bads = [d for d in c["docs"] if d["role"] == "bad"]
        if c["test"] == "swap":
            assert good["score"] == 1 and all(b["score"] == 5 for b in bads)
        else:
            assert good["score"] == 5 and all(b["score"] < 5 for b in bads)


def test_one_change_at_a_time():
    by = {(c["fact_id"], c["test"]): c for c in cases()}
    for (fid, t), c in by.items():
        plain = by[(fid, "plain")]
        if t in ("swap", "reading", "gap2", "gap3", "gap4"):
            assert [d["text"] for d in c["docs"]] == [d["text"] for d in plain["docs"]]
        if t.startswith("forge_") and t != "forge_all":
            # same good doc, same position; the bad doc is the plain bad doc plus the forgery
            pb = [d for d in plain["docs"] if d["role"] == "bad"][0]["text"]
            cb = [d for d in c["docs"] if d["role"] == "bad"][0]["text"]
            assert pb in cb and cb != pb
            assert [d["role"] for d in c["docs"]] == [d["role"] for d in plain["docs"]]


def test_repeat_docs_differ():
    for c in cases():
        if c["test"].startswith("repeat_"):
            texts = [d["text"] for d in c["docs"] if d["role"] == "bad"]
            assert len(texts) == int(c["test"].split("_")[1]) and len(set(texts)) == len(texts)


def test_forged_tag_rendering():
    c = next(c for c in cases() if c["test"] == "forge_copytag")
    for m in ("notag", "tags", "cram", "direction", "direction_tags"):
        b = R.build(c, m)
        assert FORGED_TAG not in b["messages"][1]["content"] and "(reliability 5/5)" in b["messages"][1]["content"]
    b = R.build(c, "cag")
    assert b["text"].count("High credibility of text:") == 2      # the real high label and the forged one


def test_tags_shown_only_where_intended():
    c = next(c for c in cases() if c["test"] == "plain")
    assert "reliability" not in R.build(c, "notag")["messages"][1]["content"]
    assert "reliability" not in R.build(c, "direction")["messages"][1]["content"]
    assert "(reliability 1/5)" in R.build(c, "tags")["messages"][1]["content"]
    t = R.build(c, "cag")["text"]
    assert "Low credibility of text:" in t
    # CAG's own format: "{system} USER: Question:{q}\nDocs:{label} credibility of text: ...\n\nAnswer: ASSISTANT:"
    assert " USER: Question:" in t and "\nDocs:" in t and t.endswith("\n\nAnswer: ASSISTANT:")


def test_tool_layout():
    c = next(c for c in cases() if c["test"] == "repeat_4")
    for m in ("notag", "tags", "cram", "direction", "direction_tags"):
        b = R.build(c, m, "tool")
        roles = [x["role"] for x in b["messages"]]
        assert roles == ["system", "user"] + ["ipython"] * len(c["docs"])          # one tool message per document
        assert all(t in x["content"] for t, x in zip(b["texts"], b["messages"][2:]))
        assert not any(t in b["messages"][1]["content"] for t in b["texts"])        # no document in the user message
    assert "(reliability 1/5)" in R.build(c, "tags", "tool")["messages"][3]["content"] or \
           "(reliability 1/5)" in R.build(c, "tags", "tool")["messages"][2]["content"]
    try:
        R.build(c, "cag", "tool"); assert False, "CAG must refuse the tool layout"
    except AssertionError as e:
        assert "CAG" in str(e)


def test_reading_question_points_at_bad_doc():
    for c in cases():
        if c["test"] == "reading":
            assert c["docs"][c["report_doc"] - 1]["role"] == "bad"


def test_strengths():
    assert R.strengths([5, 1]) == [0.0, 0.8]
    assert R.strengths([5, 5, 1], scale=0.5) == [0.0, 0.0, 0.4]


def test_scoring():
    c = {"good_aliases": ["Priya Nandan", "Nandan"], "bad_aliases": ["Omar Reyes", "Reyes"], "expect": "good"}
    assert S.classify(c, "Priya Nandan.") == "correct"
    assert S.classify(c, "It is Omar Reyes") == "wrong"
    assert S.classify(c, "Either Nandan or Reyes") == "both"
    assert S.classify(c, "Either Nandan or Reyes", judged="good") == "correct"
    assert S.classify(c, "I don't know") == "neither"
    assert S.classify(dict(c, expect="bad"), "Omar Reyes") == "correct"
    # whole-word matching: 1960 must not match 19600, and numbers with commas match without
    c2 = {"good_aliases": ["3,450", "3450"], "bad_aliases": ["1960"], "expect": "good"}
    assert S.classify(c2, "about 3,450 people") == "correct" and S.classify(c2, "19600") == "neither"
    assert normalize("May 18 , 2010") == normalize("May 18, 2010")


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
