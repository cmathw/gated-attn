import math
from copy import deepcopy

import torch as t
from fancy_einsum import einsum as es
from jaxtyping import Float, Int
from torch import Tensor, nn

from gated_attention.modelling.traditional_transformer import (
    VerySimpleAttnOnlyTransformer,
)
from gated_attention.modelling.transformer_components.embed_unembed import (
    Embed,
    Unembed,
)


class ModifiedAttention(nn.Module):
    def __init__(self, orig_model, expansion_factor, seed, use_duplicate_weights=False):
        super().__init__()
        original_n_heads = orig_model.cfg.n_heads
        self.n_heads = expansion_factor * original_n_heads
        self.d_model = orig_model.cfg.d_model
        self.device = orig_model.cfg.device
        self.d_head = orig_model.cfg.d_head
        self.d_gate = 1

        # Retrieve original weights (ignoring output bias)
        orig_W_Q = orig_model.attn.W_Q.clone()
        orig_W_K = orig_model.attn.W_K.clone()
        orig_W_V = orig_model.attn.W_V.clone()
        orig_W_O = orig_model.attn.W_O.clone()

        orig_b_Q = orig_model.attn.b_Q.clone()
        orig_b_K = orig_model.attn.b_K.clone()
        orig_b_V = orig_model.attn.b_V.clone()

        # Duplicate weights and biases according to expansion_factor
        duplicated_W_Q = t.cat([orig_W_Q] * expansion_factor, dim=0)
        duplicated_W_K = t.cat([orig_W_K] * expansion_factor, dim=0)
        duplicated_W_V = t.cat([orig_W_V] * expansion_factor, dim=0)
        duplicated_W_O = t.cat([orig_W_O] * expansion_factor, dim=0)

        duplicated_b_Q = t.cat([orig_b_Q] * expansion_factor, dim=0)
        duplicated_b_K = t.cat([orig_b_K] * expansion_factor, dim=0)
        duplicated_b_V = t.cat([orig_b_V] * expansion_factor, dim=0)

        # Create noise tensors with Xavier normal initialization
        noise_W_Q = t.empty_like(duplicated_W_Q)
        noise_W_K = t.empty_like(duplicated_W_K)
        noise_W_V = t.empty_like(duplicated_W_V)
        noise_W_O = t.empty_like(duplicated_W_O)

        noise_b_Q = t.empty_like(duplicated_b_Q)
        noise_b_K = t.empty_like(duplicated_b_K)
        noise_b_V = t.empty_like(duplicated_b_V)

        t.manual_seed(seed=seed)
        nn.init.xavier_normal_(noise_W_Q)
        nn.init.xavier_normal_(noise_W_K)
        nn.init.xavier_normal_(noise_W_V)
        nn.init.xavier_normal_(noise_W_O)
        nn.init.xavier_normal_(noise_b_Q)
        nn.init.xavier_normal_(noise_b_K)
        nn.init.xavier_normal_(noise_b_V)

        if use_duplicate_weights:
            # Add Xavier normal noise to the duplicated weights and biases
            duplicated_W_Q += noise_W_Q
            duplicated_W_K += noise_W_K
            duplicated_W_V += noise_W_V
            duplicated_W_O += noise_W_O

            duplicated_b_Q += noise_b_Q
            duplicated_b_K += noise_b_K
            duplicated_b_V += noise_b_V
        else:
            duplicated_W_Q = noise_W_Q
            duplicated_W_K = noise_W_K
            duplicated_W_V = noise_W_V
            duplicated_W_O = noise_W_O

            duplicated_b_Q = noise_b_Q
            duplicated_b_K = noise_b_K
            duplicated_b_V = noise_b_V

        # Set duplicated weights as frozen
        self.W_Q = nn.Parameter(duplicated_W_Q, requires_grad=True)
        self.W_K = nn.Parameter(duplicated_W_K, requires_grad=True)
        self.W_V = nn.Parameter(duplicated_W_V, requires_grad=True)
        self.W_O = nn.Parameter(duplicated_W_O, requires_grad=True)

        self.b_Q = nn.Parameter(duplicated_b_Q, requires_grad=True)
        self.b_K = nn.Parameter(duplicated_b_K, requires_grad=True)
        self.b_V = nn.Parameter(duplicated_b_V, requires_grad=True)

        # Initialize gates as before
        self.W_gate_Q = nn.Parameter(
            t.empty(
                (self.n_heads, self.d_model, self.d_gate),
                dtype=t.float32,
                requires_grad=True,
            )
        )
        self.W_gate_K = nn.Parameter(
            t.empty(
                (self.n_heads, self.d_model, self.d_gate),
                dtype=t.float32,
                requires_grad=True,
            )
        )
        self.b_gate_Q = nn.Parameter(
            t.zeros((self.n_heads, self.d_gate), dtype=t.float32, requires_grad=True)
        )
        self.b_gate_K = nn.Parameter(
            t.zeros((self.n_heads, self.d_gate), dtype=t.float32, requires_grad=True)
        )

        nn.init.orthogonal_(self.W_gate_Q)
        nn.init.orthogonal_(self.W_gate_K)

        self.cache = {}

    def forward(
        self, normalised_resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:

        W_V_normed = nn.functional.normalize(self.W_V, p=2, dim=1)
        W_O_normed = nn.functional.normalize(self.W_O, p=2, dim=2)

        r = normalised_resid_pre

        queries = (
            es(
                "n_heads d_model d_head, batch pos_q d_model -> batch pos_q n_heads d_head",
                self.W_Q,
                r,
            )
            + self.b_Q
        )
        query_gate = (
            es(
                "n_heads d_model d_gate, batch pos_q d_model -> batch pos_q n_heads d_gate",
                self.W_gate_Q,
                r,
            )
            + self.b_gate_Q
        )

        keys = (
            es(
                "n_heads d_model d_head, batch pos_k d_model -> batch pos_k n_heads d_head",
                self.W_K,
                r,
            )
            + self.b_K
        )
        key_gate = (
            es(
                "n_heads d_model d_gate, batch pos_k d_model -> batch pos_k n_heads d_gate",
                self.W_gate_K,
                r,
            )
            + self.b_gate_K
        )

        values = (
            es(
                "n_heads d_model d_head, batch pos_k d_model -> batch pos_k n_heads d_head",
                W_V_normed,
                r,
            )
            + self.b_V
        )

        attn_scores = es(
            "batch pos_q n_heads d_head, batch pos_k n_heads d_head -> batch n_heads pos_q pos_k",
            queries,
            keys,
        )

        attn_gate = es(
            "batch pos_q n_heads d_head, batch pos_k n_heads d_head -> batch n_heads pos_q pos_k",
            query_gate,
            key_gate,
        )

        attn_gate = t.clamp(attn_gate, min=0, max=1)

        # attn_gate = t.nn.ReLU()(attn_gate)
        # attn_gate = t.nn.Tanh()(attn_gate)

        reg_attn_gate = t.norm(attn_gate, p=0.6, dim=1).mean()

        # Scale attn scores
        scaled_attn = attn_scores / math.sqrt(self.d_head)
        masked_attn = self.apply_causal_mask(scaled_attn)
        softmax_attn = masked_attn.softmax(-1)

        gated_attn = es(
            "batch n_heads pos_q pos_k, batch n_heads pos_q pos_k -> batch n_heads pos_q pos_k",
            attn_gate,
            softmax_attn,
        )
        pre_z = es(
            "batch n_heads pos_q pos_k, batch pos_k n_heads d_head -> batch n_heads pos_q pos_k d_head",
            gated_attn,
            values,
        )
        z = pre_z.sum(3)
        attn_out_pre_PER_HEAD = es(
            "batch n_heads pos_q d_head, n_heads d_head d_model -> batch pos_q n_heads d_model",
            z,
            W_O_normed,
        )
        output = attn_out_pre_PER_HEAD.sum(dim=2)

        # Cache intermediate activations
        self.cache["normalized_resid_pre"] = r
        self.cache["queries"] = queries
        self.cache["keys"] = keys
        self.cache["reg_attn_gate"] = reg_attn_gate
        self.cache["attn_score"] = attn_scores
        self.cache["attn_gate"] = attn_gate
        self.cache["pattern"] = gated_attn
        self.cache["values"] = values
        self.cache["z"] = z
        self.cache["attn_out_pre_per_head"] = attn_out_pre_PER_HEAD
        self.cache["attn_output"] = output
        return output

    def apply_causal_mask(self, attn_scores: t.Tensor):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        # Generate a mask for the upper triangular part
        batch_size, n_heads, query_length, key_length = attn_scores.size()
        mask = t.triu(
            t.ones((query_length, key_length), device=attn_scores.device), diagonal=1
        ).bool()
        # Expand mask to match the dimensions of attn_scores
        mask = mask[None, None, :, :].expand(
            batch_size, n_heads, -1, -1
        )  # [batch, n_heads, query_pos, key_pos]
        # Apply the mask by setting masked positions to a large negative value
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        return attn_scores


class ModifiedAttentionTransformer(nn.Module):
    def __init__(
        self,
        orig_model: VerySimpleAttnOnlyTransformer,
        modified_attn: ModifiedAttention,
    ) -> None:
        super().__init__()
        self.cfg = deepcopy(orig_model.cfg)
        self.cfg.n_heads = modified_attn.n_heads
        self.device = orig_model.cfg.device
        self.cache: dict = {}
        self.embed = Embed(self.cfg)
        self.attn = modified_attn
        self.unembed = Unembed(self.cfg)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        # resid_pre, resid_post, unembed=output, embed
        resid = self.embed(tokens)
        self.cache["embed"] = resid.detach().clone()
        resid = self.attn(resid) + resid
        self.cache.update(self.attn.cache)
        self.cache["resid_post"] = resid.detach().clone()
        logits = self.unembed(resid)
        self.cache["unembed"] = logits.detach().clone()
        return logits
