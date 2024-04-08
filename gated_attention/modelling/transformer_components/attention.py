from collections import defaultdict

import torch as t
from fancy_einsum import einsum as es
from torch import Tensor, nn

from gated_attention.modelling.transformer_components.config import (
    SimplifiedAttnOnlyModelConfig,
)


class Attention(nn.Module):
    def __init__(self, cfg: SimplifiedAttnOnlyModelConfig):
        super().__init__()
        self.cfg = cfg

        t.manual_seed(seed=cfg.seed)

        def empty_weights(*shape):
            return nn.init.xavier_normal_(
                nn.Parameter(t.empty(shape, device=self.cfg.device))
            )

        def zero_weights(*shape):
            return nn.init.xavier_normal_(
                nn.Parameter(t.zeros(shape, device=self.cfg.device))
            )

        self.W_Q, self.W_K, self.W_V = (
            empty_weights(cfg.n_heads, cfg.d_model, cfg.d_head) for _ in range(3)
        )
        self.W_O = empty_weights(cfg.n_heads, cfg.d_head, cfg.d_model)

        self.b_Q, self.b_K, self.b_V = (
            zero_weights(cfg.n_heads, cfg.d_head) for _ in range(3)
        )
        if self.cfg.output_bias:
            self.b_O = t.zeros(cfg.d_model, device=self.cfg.device)

        def init_weights(*weights):
            (nn.init.xavier_normal_(w) for w in weights)

        init_weights(self.W_Q, self.W_K, self.W_V, self.W_O)
        self.register_buffer(
            "IGNORE", t.tensor(-1e5, dtype=t.float32, device=self.cfg.device)
        )
        self.cache: dict = defaultdict(list)
        self.attn_activation_cache = self.cache

    def forward(self, normalized_resid_pre: Tensor):
        # batch = batch, po = position, dm = d_model, dh = d_head, n_heads = n_heads
        # pq = pos_q, pk = pos_k
        dh = self.cfg.d_head

        r = normalized_resid_pre  # [batch, pos, d_model]
        assert r.ndim == 3 and r.shape[-1] == self.cfg.d_model

        def qkv(r, w):
            return es(
                "Batch Pos D_MODEL,  n_heads D_MODEL d_heaD -> Batch Pos n_heads d_heaD",
                r,
                w,
            )

        w_qkv = (self.W_Q, self.W_K, self.W_V)

        q_pre, k_pre, v_pre = (qkv(r, w) for w in w_qkv)

        q = q_pre + self.b_Q
        k = k_pre + self.b_K
        v = v_pre + self.b_V

        scores = es(
            "batch Pos_q n_heads D_HEAD,  batch pos_K n_heads D_HEAD -> batch n_heads Pos_q pos_K",
            q,
            k,
        )

        scores = scores / (dh**0.5)
        mask = t.triu(
            t.ones(scores.size(-2), scores.size(-1), device=r.device),
            diagonal=1,
        ).bool()
        attn_scores_masked = scores.masked_fill(mask, self.IGNORE)
        attn_pattern = attn_scores_masked.softmax(-1)

        z = es(
            "batch POS_K n_heads D_head,  batch n_heads pos_Q POS_K -> batch pos_Q n_heads D_head",
            v,
            attn_pattern,
        )

        attn_out_pre_per_head = es(
            "Batch Pos_q n_heads D_HEAD,  n_heads D_HEAD d_modeL -> Batch Pos_q n_heads d_modeL",
            z,
            self.W_O,
        )

        attn_out_pre = attn_out_pre_per_head.sum(dim=2)

        attn_out = attn_out_pre

        if self.cfg.output_bias:
            attn_out = attn_out + self.b_O

        def add_cache():
            self.cache["q_pre"] = q_pre
            self.cache["k_pre"] = k_pre
            self.cache["q"] = q
            self.cache["k"] = k
            self.cache["attn_scores_unmasked"] = scores
            self.cache["attn_scores"] = attn_scores_masked
            self.cache["pattern"] = attn_pattern
            self.cache["v_pre"] = v_pre
            self.cache["v"] = v
            self.cache["z"] = z
            self.cache["attn_out_pre_per_head"] = attn_out_pre_per_head
            self.cache["attn_out_pre"] = attn_out_pre
            self.cache["attn_out"] = attn_out

        add_cache()

        return attn_out
