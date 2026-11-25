"""Run one method on the benchmark and save every reply.

Examples (on a GPU node; see README for the full list):
  python -m bench.run_eval --method notag     --model meta-llama/Llama-3.1-8B-Instruct --out runs/llama31_notag.jsonl
  python -m bench.run_eval --method tags      --model meta-llama/Llama-3.1-8B-Instruct --out runs/llama31_tags.jsonl
  python -m bench.run_eval --method direction --model meta-llama/Llama-3.1-8B-Instruct \
      --vectors <repo>/direction/results/models/direction_llama31-inline/vectors.pt --out runs/llama31_direction.jsonl
  python -m bench.run_eval --method cram      --model meta-llama/Llama-3.1-8B-Instruct \
      --heads <repo>/cram/data/heads/heads_scores_mean_nq_llama31-inline.json --out runs/llama31_cram.jsonl
  python -m bench.run_eval --method cag       --model ruotong-pan/CAG-7b --max_new 200 --out runs/cag7b.jsonl

Resumes automatically: cases already in --out are skipped.
"""
import argparse
import json
import math
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import render as R
from .hooks import CramHook, LayerVectors, load_heads

METHODS = ("notag", "tags", "cag", "cram", "direction", "direction_tags", "direction_floor", "direction_mean")


def prompt_string(tok, built):
    """the chat-templated prompt. Tool results ("ipython" messages) are written as raw text: the Llama-3.1 template would
    JSON-quote a string tool result, while Meta's reference format, and David's seplib/chatfmt.render, write it as is."""
    if built["kind"] == "raw":
        return built["text"]
    raw, msgs = [], []
    for i, m in enumerate(built["messages"]):
        if m["role"] == "ipython":
            msgs.append({"role": "user", "content": f"@@IPY{i}@@"}); raw.append((i, m["content"]))
        else:
            msgs.append(m)
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    for i, c in raw:
        hdr = f"<|start_header_id|>user<|end_header_id|>\n\n@@IPY{i}@@"
        assert chat.count(hdr) == 1, "the tool layout expects the Llama-3 chat header format"
        chat = chat.replace(hdr, f"<|start_header_id|>ipython<|end_header_id|>\n\n{c.strip()}")
    return chat


def doc_spans(prompt, texts):
    """character span of each document text in the prompt, found left to right"""
    spans, pos = [], 0
    for t in texts:
        a = prompt.find(t, pos)
        if a < 0:
            raise RuntimeError("document text not found in the rendered prompt")
        spans.append((a, a + len(t))); pos = a + len(t)
    return spans


def token_values(offsets, spans, values):
    """per-token value: the value of the document whose characters the token overlaps, else 0"""
    out = torch.zeros(len(offsets))
    for j, (o0, o1) in enumerate(offsets):
        if o1 <= o0:
            continue
        for (a, b), v in zip(spans, values):
            if o1 > a and o0 < b:
                out[j] = v; break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/bench.jsonl")
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--layout", default="inline", choices=R.LAYOUTS,
                    help="inline: documents inside the user message; tool: each document a tool result. Use the direction "
                         "vectors and CrAM heads made for the same layout (David's *-ipython files for tool)")
    ap.add_argument("--model", required=True, help="Hugging Face id or local path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vectors", default="", help="direction: vectors.pt from David's repo")
    ap.add_argument("--dir_scale", type=float, default=1.0, help="direction: multiplies every strength")
    ap.add_argument("--floor", type=float, default=0.75, help="direction_floor / direction_mean: lowest strength")
    ap.add_argument("--heads", default="", help="cram: head list JSON")
    ap.add_argument("--n_heads", type=int, default=894, help="cram: top heads used (CrAM: 894 for NQ, 506 for TriviaQA)")
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (smoke test)")
    ap.add_argument("--tests", default="", help="comma-separated test names only")
    ap.add_argument("--double_bos", action="store_true",
                    help="tokenise the chat-templated string with add_special_tokens=True (a second BOS), as David's current "
                         "harness does; use it with vectors trained before his simplify-data fix")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()

    cases = [json.loads(l) for l in open(a.bench)]
    if a.tests: cases = [c for c in cases if c["test"] in set(a.tests.split(","))]
    if a.limit: cases = cases[: a.limit]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    todo = [c for c in cases if c["id"] not in done]
    print(f"{len(cases)} cases, {len(done)} already done, {len(todo)} to run", flush=True)
    if not todo: return

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype),
                                                 attn_implementation="eager" if a.method == "cram" else "sdpa").to(a.device).eval()
    layers = model.model.layers
    dvec = cram = None
    if a.method.startswith("direction"):
        assert a.vectors, "--vectors is required for the direction"
        dvec = LayerVectors(layers, t_init=torch.load(a.vectors, map_location="cpu")["t"]).to(a.device)
    if a.method == "cram":
        assert a.heads, "--heads is required for CrAM"
        cram = CramHook(layers, load_heads(a.heads, a.n_heads))
    eos = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|eot_id|>")] if "<|eot_id|>" in tok.get_vocab() else [])

    items = []
    for c in todo:
        b = R.build(c, a.method, a.layout)
        p = prompt_string(tok, b)
        items.append((c, p, doc_spans(p, b["texts"]), b["scores"]))
    items.sort(key=lambda x: len(x[1]))
    add_special = a.double_bos or a.method == "cag"   # raw (non-template) prompts need the BOS from the tokenizer

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fout = open(a.out, "a"); t0 = time.time()
    for s0 in range(0, len(items), a.bs):
        batch = items[s0: s0 + a.bs]
        enc = tok([p for _, p, _, _ in batch], return_tensors="pt", padding=True, add_special_tokens=add_special,
                  return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        enc = {k: v.to(a.device) for k, v in enc.items()}
        if dvec is not None:
            dm = torch.stack([token_values(offs[j].tolist(), sp, R.strengths(sc, a.dir_scale, R.strength_mode(a.method), a.floor)) for j, (_, _, sp, sc) in enumerate(batch)])
            dvec.set(torch.ones(len(batch)), dm)
        if cram is not None:
            def logw(sc):
                mx = max(sc); return [math.log(s / mx) if s > 0 else -1e4 for s in sc]
            cram.rows = torch.stack([token_values(offs[j].tolist(), sp, logw(sc)) for j, (_, _, sp, sc) in enumerate(batch)])
        with torch.no_grad():
            g = model.generate(**enc, do_sample=False, max_new_tokens=a.max_new, eos_token_id=eos, pad_token_id=tok.pad_token_id)
        if dvec is not None: dvec.clear()
        if cram is not None: cram.rows = None
        T = enc["input_ids"].shape[1]
        for j, (c, p, _, _) in enumerate(batch):
            out = tok.decode(g[j, T:], skip_special_tokens=True).strip()
            fout.write(json.dumps({"id": c["id"], "method": a.method, "layout": a.layout, "model": a.model, "output": out}) + "\n")
        fout.flush()
        n = min(s0 + a.bs, len(items))
        print(f"{n}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    print("DONE", a.out)


if __name__ == "__main__":
    main()
