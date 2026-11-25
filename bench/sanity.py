"""Quick pass/fail checks on real run files, to catch a broken setup before spending GPU time on the full run.

Run after a small trial (e.g. --limit 140 per method):
  python -m bench.sanity --runs "runs/*.jsonl"

Each check prints PASS, WARN or FAIL with the number behind it. A FAIL means something is likely wrong with the setup (prompt,
model, hook), not with the method being tested; fix it before the full run.
"""
import argparse
import glob
import json
import statistics
from collections import defaultdict

from .score import classify


def rate(xs, what):
    return sum(o == what for o in xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/bench.jsonl")
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(a.bench)}
    runs = defaultdict(list)
    for pat in a.runs:
        for p in sorted(glob.glob(pat)):
            if p.endswith("judge.json"):
                continue
            for l in open(p):
                r = json.loads(l)
                if r["id"] in cases:
                    runs[(r.get("layout", "inline"), r["method"])].append((cases[r["id"]], r["output"]))
    if not runs:
        print("no replies found"); return
    out = []

    def check(name, ok, warn, detail):
        out.append(("PASS" if ok else "WARN" if warn else "FAIL", name, detail))

    for (layout, method), recs in sorted(runs.items()):
        tag = f"{layout}/{method}"
        res = [(c, classify(c, o), o) for c, o in recs]
        by = lambda t, ft=None: [r for c, r, _ in res if c["test"] == t and (ft is None or c["fact_type"] == ft)]
        n = len(res)
        neither = rate([r for _, r, _ in res], "neither")
        check(f"{tag}: replies name an answer", neither < 0.15, neither < 0.3,
              f"{neither:.0%} of {n} replies name neither answer (blank, refusal or garbled?)")
        words = statistics.median(len(o.split()) for _, _, o in res)
        check(f"{tag}: replies are short", words <= 25, words <= 60, f"median {words:.0f} words")
        rd = by("reading")
        if rd:
            v = rate(rd, "correct"); check(f"{tag}: reading task", v >= 0.8, v >= 0.6, f"{v:.0%} report the named document correctly")
        pl = by("plain")
        if method in ("notag", "direction_mean"):
            pf = by("plain", "fictional")
            if pf:
                g = rate(pf, "correct") + 0.5 * rate(pf, "both")
                check(f"{tag}: no hidden cue (invented facts, plain)", 0.25 <= g <= 0.75, 0.15 <= g <= 0.85,
                      f"picks the good document {g:.0%} of the time; with no scores (or equal strengths) this should be near 50%")
        elif pl:
            v = rate(pl, "correct")
            check(f"{tag}: plain cases", v >= 0.7, v >= 0.5, f"{v:.0%} correct")
            sw = by("swap")
            if sw and method in ("tags", "direction_tags", "cag"):
                s = rate(sw, "correct"); check(f"{tag}: swap flips the answer", s >= 0.6, s >= 0.4, f"{s:.0%} follow the swapped scores")
    base = {lay: [r for c, r, _ in [(c, classify(c, o), o) for c, o in runs[(lay, "notag")]] if c["test"] == "plain"]
            for lay in ("inline", "tool") if (lay, "notag") in runs}
    for (layout, method), recs in sorted(runs.items()):
        if method in ("cram", "direction", "direction_floor") and layout in base:
            pl = [classify(c, o) for c, o in recs if c["test"] == "plain"]
            if pl and base[layout]:
                m, b = rate(pl, "wrong"), rate(base[layout], "wrong")
                check(f"{layout}/{method}: lowers bad answers vs no tags", m <= b, m <= b + 0.05,
                      f"bad-answer rate {m:.0%} vs {b:.0%} with no tags (plain cases)")
    for st, name, detail in out:
        print(f"{st:5}  {name}: {detail}")
    fails = sum(s == "FAIL" for s, _, _ in out)
    print(f"\n{fails} FAIL, {sum(s == 'WARN' for s, _, _ in out)} WARN, {sum(s == 'PASS' for s, _, _ in out)} PASS")


if __name__ == "__main__":
    main()
