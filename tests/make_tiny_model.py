"""A tiny random Llama with a Llama-3-style chat template, for testing the pipeline on CPU (no download).
Its replies are gibberish; the point is that prompts, token marking, hooks, generation and scoring all run.

python tests/make_tiny_model.py --bench data/bench.jsonl --out /tmp/tiny_model
"""
import argparse
import json
import os

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

SPECIALS = ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
TEMPLATE = ("{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n{{ m['content'] }}<|eot_id|>"
            "{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bench", default="data/bench.jsonl"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    texts = []
    for l in open(a.bench):
        c = json.loads(l); texts += [c["question"]] + [d["text"] for d in c["docs"]]
    texts += ["Answer the question using the documents below. Document reliability High Middle Low credibility of text"]
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=2000, special_tokens=SPECIALS,
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<|begin_of_text|>", eos_token="<|end_of_text|>")
    tok.chat_template = TEMPLATE
    # a BOS when add_special_tokens=True, as Llama's tokenizer does
    from tokenizers.processors import TemplateProcessing
    bos = tk.token_to_id("<|begin_of_text|>")
    tok._tokenizer.post_processor = TemplateProcessing(single="<|begin_of_text|> $A", special_tokens=[("<|begin_of_text|>", bos)])
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=8192, bos_token_id=bos, eos_token_id=tok.eos_token_id,
                      pad_token_id=tok.eos_token_id)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    os.makedirs(a.out, exist_ok=True); model.save_pretrained(a.out); tok.save_pretrained(a.out)
    torch.save({"t": torch.randn(4, 64) * 0.5}, f"{a.out}/vectors.pt")
    json.dump([[1.0 - 0.01 * i, [l, h]] for i, (l, h) in enumerate((l, h) for l in range(4) for h in range(4))],
              open(f"{a.out}/heads.json", "w"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
