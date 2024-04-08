from dataclasses import dataclass, field
from typing import Optional

import torch as t

from gated_attention.dataset.toy_datasets import OVIncoherentDatasetConfig

DEVICE = "cuda:0" if t.cuda.is_available() else "cpu"


@dataclass
class SimplifiedAttnOnlyModelConfig:
    dataset_cfg: OVIncoherentDatasetConfig = field(
        default_factory=OVIncoherentDatasetConfig
    )
    device: str = DEVICE
    init_range: float = 0.02
    n_heads: int = 4
    d_head: int = 1
    seed: int = 0
    output_bias: bool = True

    def __post_init__(
        self, dataset_cfg: Optional[OVIncoherentDatasetConfig] = None
    ) -> None:
        if dataset_cfg is not None:
            self.dataset_cfg = OVIncoherentDatasetConfig()
        self.d_vocab: int = self.dataset_cfg.n_tokens
        self.d_model: int = self.dataset_cfg.n_tokens
