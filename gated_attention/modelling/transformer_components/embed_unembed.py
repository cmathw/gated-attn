import torch as t
from fancy_einsum import einsum as es
from jaxtyping import Float, Int
from torch import Tensor, nn

from gated_attention.modelling.transformer_components.config import (
    SimplifiedAttnOnlyModelConfig,
)


class Embed(nn.Module):
    def __init__(self, cfg: SimplifiedAttnOnlyModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        assert cfg.d_vocab == cfg.d_model
        self.W_E = t.eye(cfg.d_model).to(cfg.device)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        embed = self.W_E[tokens, :]
        return embed


class Unembed(nn.Module):
    def __init__(self, cfg: SimplifiedAttnOnlyModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        assert cfg.d_vocab <= cfg.d_model
        self.W_U = t.eye(cfg.d_model).to(self.cfg.device)

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return es(
            "batch posn d_model, d_model d_vocab -> batch posn d_vocab",
            normalized_resid_final,
            self.W_U,
        )
