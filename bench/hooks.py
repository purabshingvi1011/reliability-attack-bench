"""The two interventions under test, copied from David's repository so this package runs on its own.

LayerVectors : seplib/trust.py (the learned direction). Vector files from direction/results/models/<cell>/vectors.pt
               load with torch.load(path)["t"].
CramHook     : cram/scripts/cram_eval.py (CrAM, Deng et al. AAAI 2025): adds log(score/max) to the attention logits of each
               document's tokens in the selected heads, for every query, at the prompt pass and every decoding step.
               Needs attn_implementation="eager".
Keep these in sync with the repo if David changes them.
"""
import torch
import torch.nn as nn


class LayerVectors(nn.Module):
    def __init__(self, layers, t_init=None, d=None):
        super().__init__()
        L = len(layers)
        t0 = torch.as_tensor(t_init, dtype=torch.float32) if t_init is not None else torch.zeros(L, d)
        assert t0.shape[0] == L, f"vector file has {t0.shape[0]} layers, model has {L}"
        self.t = nn.Parameter(t0.clone(), requires_grad=False); self.betas = None; self.dmask = None
        self.handles = [layers[l].register_forward_hook(self._make_hook(l)) for l in range(L)]

    def set(self, betas, dmask):
        self.betas = torch.as_tensor(betas, dtype=torch.float32); self.dmask = torch.as_tensor(dmask, dtype=torch.float32)

    def clear(self):
        self.betas = None; self.dmask = None

    def _make_hook(self, l):
        def hook(mod, args, out):
            if self.betas is None or self.dmask is None: return None
            h = out[0] if isinstance(out, tuple) else out
            if self.dmask.shape != h.shape[:2]: return None          # decoding steps (length 1) are left unchanged
            add = (self.betas.to(h.device)[:, None, None] * self.dmask.to(h.device)[..., None]) * self.t[l].to(h.device)[None, None, :]
            h = h + add.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook


class CramHook:
    def __init__(self, layers, heads):
        """heads: {layer: [head, ...]}"""
        self.rows, self.handles = None, []
        for layer, hs in heads.items():
            attn = layers[int(layer)].self_attn
            self.handles.append(attn.register_forward_pre_hook(self._mk(sorted(hs)), with_kwargs=True))

    def _mk(self, hs):
        def hook(mod, args, kwargs):
            if self.rows is None: return None
            h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            B, Q = h.shape[:2]; am = kwargs.get("attention_mask"); cp = kwargs.get("cache_position")
            T = self.rows.shape[1]
            K = am.shape[-1] if am is not None else int(cp[-1]) + 1
            if am is None:
                causal = cp[:, None] >= torch.arange(K, device=h.device)[None, :]
                am = torch.where(causal, 0.0, torch.finfo(h.dtype).min).to(h.dtype)[None, None].expand(B, 1, Q, K)
            elif am.dtype == torch.bool:
                am = torch.where(am, 0.0, torch.finfo(h.dtype).min).to(h.dtype)
            H = mod.config.num_attention_heads
            out = am.expand(B, H, Q, K).clone()
            r = self.rows.to(out.device, out.dtype)[:, None, None, :]
            sel = out[:, hs, :, :T]
            out[:, hs, :, :T] = torch.where(sel > torch.finfo(out.dtype).min / 2, sel + r, sel)
            kwargs["attention_mask"] = out
            return args, kwargs
        return hook


def load_heads(path, n):
    """CrAM head list file (cram/data/heads/heads_scores_mean_<ds>_<cell>.json): [[score, [layer, head]], ...], best first"""
    import json
    hs = json.load(open(path))[:n]
    heads = {}
    for _, (l, h) in hs:
        heads.setdefault(int(l), []).append(int(h))
    return heads
