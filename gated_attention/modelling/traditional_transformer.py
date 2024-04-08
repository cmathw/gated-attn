import torch as t
from jaxtyping import Float, Int
from torch import Tensor, nn

from gated_attention.modelling.transformer_components.attention import Attention
from gated_attention.modelling.transformer_components.config import (
    SimplifiedAttnOnlyModelConfig,
)
from gated_attention.modelling.transformer_components.embed_unembed import (
    Embed,
    Unembed,
)


class VerySimpleAttnOnlyTransformer(nn.Module):
    """
    SimplifiedAttnOnlyTransformer is a simplified transformer with the following features:
        1. Single layer
        2. Does not use Layer Norm
        3. Attention Only
        4. Uses One Hot Encoding for Embed and Unembed
    """

    def __init__(self, cfg: SimplifiedAttnOnlyModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = "cuda:0" if t.cuda.is_available() else "cpu"
        self.cache: dict = {}
        self.embed = Embed(self.cfg)
        self.attn = Attention(self.cfg)
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

    class training_cfg:
        batch_size: int = 8  # 8
        num_epochs: int = 1
        max_steps: float = 1e6  # float("inf")
        log_every: int = 200
        lr: float = 5e-4
        weight_decay: float = 0
