"""Pre-flight checks for reliability_bench on CPU with a tiny random Llama. usage: python preflight.py <bench_dir> <david_repo> <tmp>"""
import ast
import json
import os
import subprocess
import sys

import torch

BENCH, DAVID, TMP = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, BENCH)
os.makedirs(TMP, exist_ok=True)
from bench import render as R                                     # noqa: E402
from bench.hooks import CramHook, LayerVectors                    # noqa: E402
from bench.run_eval import doc_spans, prompt_string, token_values  # noqa: E402
from bench.templates import FORGED_TAG                            # noqa: E402

PROBLEMS = []
def ok(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond: PROBLEMS.append(msg)

cases = [json.loads(l) for l in open(f"{BENCH}/data/bench.jsonl")]
tests = sorted({c["test"] for c in cases})
sample = []
for src in ("generated", "CrAM nq", "CrAM trivia", "CAG NewsPolluted"):
    for t in tests:
        sample += [c for c in cases if c["test"] == t and c["source"].startswith(src)][:2]

# ---------------------------------------------------------------- 1. tiny model with a Llama-3.1-style chat template
print("1. tiny model and tokenizer")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402
from tokenizers.processors import TemplateProcessing                        # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402
SPECIALS = ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
# the parts of Meta's Llama-3.1 template that matter here: BOS, the system block with the knowledge-date preamble, a header
# per message, trimmed content, and tool results ("ipython") written with tojson (so a string comes out JSON-quoted)
TEMPLATE = ("{{- bos_token }}{%- set sys = messages[0]['content'] if messages[0]['role'] == 'system' else '' %}"
            "{{- '<|start_header_id|>system<|end_header_id|>\\n\\nCutting Knowledge Date: December 2023\\nToday Date: 26 Jul 2024\\n\\n' + sys + '<|eot_id|>' }}"
            "{%- for m in messages %}{%- if m['role'] == 'system' %}{%- elif m['role'] in ['ipython', 'tool'] %}"
            "{{- '<|start_header_id|>ipython<|end_header_id|>\\n\\n' + (m['content'] | tojson) + '<|eot_id|>' }}"
            "{%- else %}{{- '<|start_header_id|>' + m['role'] + '<|end_header_id|>\\n\\n' + (m['content'] | trim) + '<|eot_id|>' }}{%- endif %}{%- endfor %}"
            "{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}")
texts = [c["question"] for c in cases[:3000]] + [d["text"] for c in cases[:3000] for d in c["docs"]]
tk = Tokenizer(models.BPE()); tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=3000, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
bos = tk.token_to_id("<|begin_of_text|>")
tk.post_processor = TemplateProcessing(single="<|begin_of_text|> $A", special_tokens=[("<|begin_of_text|>", bos)])
tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<|begin_of_text|>", eos_token="<|end_of_text|>")
tok.chat_template = TEMPLATE
cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=8192, bos_token_id=bos, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
torch.manual_seed(0)
MD = f"{TMP}/tiny"
LlamaForCausalLM(cfg).save_pretrained(MD); tok.save_pretrained(MD)
T = torch.randn(4, 64) * 0.5
torch.save({"t": T}, f"{MD}/vectors.pt")
json.dump([[1.0 - 0.01 * i, [l, h]] for i, (l, h) in enumerate((l, h) for l in range(4) for h in range(4))], open(f"{MD}/heads.json", "w"))
tok = PreTrainedTokenizerFast.from_pretrained(MD)
ok(tok.chat_template is not None, "chat template saved with the tokenizer")

# ---------------------------------------------------------------- 2. prompts
print("2. prompts, for every test x source x method x layout")
n_prompts = 0
for c in sample:
    for layout in ("inline", "tool"):
        for m in ("notag", "tags", "cram", "direction", "direction_tags", "cag"):
            if m == "cag" and layout == "tool":
                continue
            b = R.build(c, m, layout); p = prompt_string(tok, b); n_prompts += 1
            if FORGED_TAG in p: ok(False, f"placeholder left in prompt: {c['id']} {m}")
            try:
                sp = doc_spans(p, b["texts"])
            except RuntimeError:
                ok(False, f"document not found in prompt: {c['id']} {m} {layout}"); continue
            if c["test"] == "forge_copytag" and m != "cag":
                if p.count("(reliability 5/5)") < 1: ok(False, f"forged tag missing: {c['id']} {m}")
            if layout == "tool":
                if p.count("<|start_header_id|>ipython<|end_header_id|>") != len(c["docs"]):
                    ok(False, f"tool layout: wrong number of tool messages {c['id']} {m}")
                if '"Document 1' in p or '\\n' in p:
                    ok(False, f"tool layout: JSON-quoted tool result {c['id']} {m}")
            shows_tags = "(reliability 1/5)" in p or "(reliability 5/5)" in p.replace(FORGED_TAG, "")
            if m in ("notag", "cram", "direction") and c["test"] != "forge_copytag" and shows_tags:
                ok(False, f"scores leaked into a no-tag prompt: {c['id']} {m}")
            if m in ("tags", "direction_tags") and "(reliability" not in p:
                ok(False, f"tags missing: {c['id']} {m}")
            if m == "cag" and not (" USER: Question:" in p and "\nDocs:" in p and p.endswith("\n\nAnswer: ASSISTANT:")):
                ok(False, f"CAG prompt shape: {c['id']}")
            if m in ("direction", "direction_tags", "cram"):
                enc = tok(p, add_special_tokens=True, return_offsets_mapping=True)
                vals = R.strengths(b["scores"]) if m != "cram" else [__import__("math").log(s / max(b["scores"])) for s in b["scores"]]
                tv = token_values(enc["offset_mapping"], sp, vals)
                marked = tok.decode([t for t, v in zip(enc["input_ids"], tv) if v != 0]).replace(" ", "")
                want = "".join(t for t, d in zip(b["texts"], c["docs"]) if R.strengths(b["scores"])[c["docs"].index(d)] > 0).replace(" ", "")
                if marked != want: ok(False, f"marking does not cover exactly the low-scored documents: {c['id']} {m} {layout}")
ok(not [x for x in PROBLEMS if "prompt" in x or "marking" in x or "tool" in x], f"{n_prompts} prompts built and checked")

# ---------------------------------------------------------------- 3. our hooks vs David's originals
print("3. our hooks give the same numbers as David's code")
sys.path.insert(0, DAVID)
from seplib.trust import LayerVectors as DavidLV  # noqa: E402
src = open(f"{DAVID}/cram/scripts/cram_eval.py").read()
node = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "CramHook")
ns = {"torch": torch}; exec(compile(ast.Module([node], []), "cram_eval", "exec"), ns); DavidCram = ns["CramHook"]

def logits_with(kind, impl, enc, rows):
    model = LlamaForCausalLM.from_pretrained(MD, dtype=torch.float32, attn_implementation="eager" if kind == "cram" else "sdpa").eval()
    if kind == "dir":
        h = (LayerVectors(model.model.layers, t_init=T) if impl == "ours" else DavidLV(model.model.layers, t_init=T))
        h.set(torch.ones(rows.shape[0]), rows)
    else:
        heads = {0: [0, 2], 2: [1, 3], 3: [0]}
        h = CramHook(model.model.layers, heads) if impl == "ours" else DavidCram(model, heads)
        h.rows = rows
    with torch.no_grad():
        out = model(**enc).logits
        g = model.generate(**enc, do_sample=False, max_new_tokens=6, pad_token_id=tok.pad_token_id)
    return out, g

tok.padding_side = "left"; tok.pad_token = tok.eos_token
two = [c for c in sample if c["test"] == "repeat_4"][:1] + [c for c in sample if c["test"] == "plain"][:1]
ps = [prompt_string(tok, R.build(c, "direction")) for c in two]
enc = tok(ps, return_tensors="pt", padding=True, return_offsets_mapping=True); offs = enc.pop("offset_mapping")
for kind in ("dir", "cram"):
    rows = []
    for j, c in enumerate(two):
        b = R.build(c, "direction"); sp = doc_spans(ps[j], b["texts"])
        vals = R.strengths(b["scores"]) if kind == "dir" else [__import__("math").log(s / max(b["scores"])) for s in b["scores"]]
        rows.append(token_values(offs[j].tolist(), sp, vals))
    rows = torch.stack(rows)
    lo, go = logits_with(kind, "ours", dict(enc), rows); ld, gd = logits_with(kind, "david", dict(enc), rows)
    base, _ = logits_with(kind, "ours", dict(enc), torch.zeros_like(rows))
    d = (lo - ld).abs().max().item(); eff = (lo - base).abs().max().item()
    name = "direction" if kind == "dir" else "CrAM"
    ok(d < 1e-4, f"{name}: our output matches David's (max difference {d:.1e})")
    ok(torch.equal(go, gd), f"{name}: same generated tokens as David's, prompt pass and decoding")
    ok(eff > 1e-3, f"{name}: the hook actually changes the model (effect {eff:.2f})")

# ---------------------------------------------------------------- 4. batch size does not change results
print("4. batching (left padding) does not change results")
batch = [c for c in sample if c["source"].startswith(("CrAM", "CAG"))][:6]
for m in ("direction", "cram"):
    model = LlamaForCausalLM.from_pretrained(MD, dtype=torch.float32, attn_implementation="eager" if m == "cram" else "sdpa").eval()
    hook = LayerVectors(model.model.layers, t_init=T) if m == "direction" else CramHook(model.model.layers, {0: [0, 2], 2: [1, 3], 3: [0]})
    def last_logits(cs):
        bs = [R.build(c, m) for c in cs]; ps = [prompt_string(tok, b) for b in bs]
        e = tok(ps, return_tensors="pt", padding=True, return_offsets_mapping=True); o = e.pop("offset_mapping")
        rows = []
        for j, b in enumerate(bs):
            vals = R.strengths(b["scores"]) if m == "direction" else [__import__("math").log(s / max(b["scores"])) for s in b["scores"]]
            rows.append(token_values(o[j].tolist(), doc_spans(ps[j], b["texts"]), vals))
        rows = torch.stack(rows)
        if m == "direction": hook.set(torch.ones(len(cs)), rows)
        else: hook.rows = rows
        with torch.no_grad(): out = model(**e).logits[:, -1]
        if m == "direction": hook.clear()
        else: hook.rows = None
        return out
    one = torch.cat([last_logits([c]) for c in batch]); many = last_logits(batch)
    d = (one - many).abs().max().item()
    ok(d < 1e-3, f"{m}: batch of 6 vs one at a time, max difference {d:.1e}")

# ---------------------------------------------------------------- 5. the real command line, every method and layout
print("5. command line: every method in both layouts, resume, judge, scoring, sanity")
sub = [c for c in cases if c["fact_id"] in {cases[0]["fact_id"], [x for x in cases if x["source"].startswith("CAG")][0]["fact_id"]}]
open(f"{TMP}/bench_small.jsonl", "w").write("".join(json.dumps(c) + "\n" for c in sub))
env = dict(os.environ, PYTHONPATH=BENCH)
def run(args):
    r = subprocess.run([sys.executable, "-m"] + args, cwd=BENCH, env=env, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-600:]
runs = f"{TMP}/runs"; os.makedirs(runs, exist_ok=True)
for layout in ("inline", "tool"):
    for m in ("notag", "tags", "direction", "direction_tags", "direction_floor", "direction_mean", "cram", "cag"):
        if m == "cag" and layout == "tool": continue
        extra = ["--vectors", f"{MD}/vectors.pt"] if m.startswith("direction") else ["--heads", f"{MD}/heads.json", "--n_heads", "6"] if m == "cram" else []
        out = f"{runs}/{layout}_{m}.jsonl"
        code, log = run(["bench.run_eval", "--bench", f"{TMP}/bench_small.jsonl", "--method", m, "--layout", layout, "--model", MD,
                         "--out", out, "--bs", "8", "--max_new", "6", "--dtype", "float32", "--double_bos"] + extra)
        n = sum(1 for _ in open(out)) if os.path.exists(out) else 0
        lay = {json.loads(l).get("layout") for l in open(out)} if n else set()
        ok(code == 0 and n == len(sub) and lay == {layout}, f"{layout}/{m}: {n}/{len(sub)} replies" + ("" if code == 0 else " ERROR " + log))
code, log = run(["bench.run_eval", "--bench", f"{TMP}/bench_small.jsonl", "--method", "notag", "--layout", "inline", "--model", MD,
                 "--out", f"{runs}/inline_notag.jsonl", "--dtype", "float32"])
ok(code == 0 and "0 to run" in log and sum(1 for _ in open(f"{runs}/inline_notag.jsonl")) == len(sub), "a rerun skips finished cases (resume)")
code, log = run(["bench.judge", "--bench", f"{TMP}/bench_small.jsonl", "--runs", f"{runs}/*.jsonl", "--model", MD, "--out", f"{runs}/judge.json"])
ok(code == 0, "judge runs" + ("" if code == 0 else " " + log))
code, log = run(["bench.score", "--bench", f"{TMP}/bench_small.jsonl", "--runs", f"{runs}/inline_*.jsonl", "--judge", f"{runs}/judge.json", "--out", f"{TMP}/t.md"])
ok(code == 0 and "## 1. Headline" in open(f"{TMP}/t.md").read(), "score writes the tables")
code, log = run(["bench.sanity", "--bench", f"{TMP}/bench_small.jsonl", "--runs", f"{runs}/*.jsonl"])
ok(code == 0 and "PASS" in log + "FAIL", "sanity checker runs")

# ---------------------------------------------------------------- 6. sizes
print("6. prompt sizes (CAG-7B reads at most 4,096 tokens)")
longest = max((len(R.build(c, "cag")["text"]), c["id"]) for c in cases)
ok(longest[0] / 3.5 < 3800, f"longest CAG prompt about {longest[0] / 3.5:.0f} tokens ({longest[1]})")
longest = max((sum(len(d["text"]) for d in c["docs"]), c["id"]) for c in cases)
print(f"  info longest documents: about {longest[0] / 3.5:.0f} tokens ({longest[1]})")

print("\nRESULT:", "all checks passed" if not PROBLEMS else f"{len(PROBLEMS)} problem(s)")
for p in PROBLEMS: print("  -", p)
