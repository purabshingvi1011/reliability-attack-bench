"""Settle replies that name BOTH answers: which one does the reply commit to as its final answer?

Only "both" replies are judged (string matching decides the rest). The judge is a local chat model reading the
question, the two candidate answers and the reply, and answering with one letter. Spot-check about 50 verdicts by
hand before trusting it, and report how many replies needed the judge.

Usage: python -m bench.judge --bench data/bench.jsonl --runs runs/*.jsonl --model meta-llama/Llama-3.1-8B-Instruct --out runs/judge.jsonl
"""
import argparse
import glob
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .facts import contains_any

PROMPT = ("A reply to a question is shown below. Which answer does the reply give as its final answer?\n\n"
          "Question: {q}\nReply: {r}\n\n(A) {a}\n(B) {b}\n(C) it does not commit to either\n\n"
          "Answer with one letter: A, B or C.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/bench.jsonl"); ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--out", default="runs/judge.jsonl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(a.bench)}
    todo = []
    for p in [p for pat in a.runs for p in sorted(glob.glob(pat))]:
        if p == a.out: continue
        for l in open(p):
            r = json.loads(l); c = cases.get(r["id"])
            if c and contains_any(r["output"], c["good_aliases"]) and contains_any(r["output"], c["bad_aliases"]):
                todo.append((c, r))
    print(f"{len(todo)} replies name both answers")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device).eval()
    letters = [tok.encode(x, add_special_tokens=False)[0] for x in ("A", "B", "C")]
    with open(a.out, "w") as f:
        for c, r in todo:
            q = c["question"] if c["task"] == "answer" else f"According to Document {c['report_doc']}, {c['question']}"
            msg = [{"role": "user", "content": PROMPT.format(q=q, r=r["output"], a=c["good"], b=c["bad"])}]
            ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            ids = {k: v.to(a.device) for k, v in ids.items()}
            with torch.no_grad():
                logits = model(**ids).logits[0, -1, letters]
            v = ["good", "bad", "neither"][int(logits.argmax())]
            f.write(json.dumps({"id": r["id"], "method": r["method"], "layout": r.get("layout", "inline"),
                                "model": r.get("model", ""), "verdict": v}) + "\n")
    print("DONE", a.out)


if __name__ == "__main__":
    main()
