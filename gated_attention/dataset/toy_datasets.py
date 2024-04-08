from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch as t
from jaxtyping import Int
from torch import Tensor
from tqdm import tqdm


@dataclass
class OVIncoherentDatasetConfig:
    n_positions: int = 11
    A_toks: Tensor = t.tensor([0])
    B_toks: Tensor = t.tensor(range(1, 6))
    device: str = "cuda:0" if t.cuda.is_available() else "cpu"
    n_batches: int = 100
    n_samples_per_batch: int = 1000
    seed: int = 0

    def __post_init__(self) -> None:
        self._validate_config()
        self._calculate_variables()

    def _validate_config(self) -> None:
        assert isinstance(self.n_batches, int)
        assert isinstance(self.n_samples_per_batch, int)
        assert isinstance(self.n_positions, int) and self.n_positions >= 3

    def _calculate_variables(self) -> None:
        self.C_toks = self._calculate_C_toks()
        self.BOS_tok: Tensor = t.tensor([self.C_toks[-1] + 1])
        self.n_skiptrigram = len(self.B_toks)
        self.ABC_toks = t.cat((self.A_toks, self.B_toks, self.C_toks, self.BOS_tok))
        self.B_TO_C = self._create_B_TO_C_mapping(self.C_toks)
        self.n_total_samples = self.n_batches * self.n_samples_per_batch
        self.n_tokens = self.ABC_toks.shape[0]

    def _calculate_C_toks(self) -> Tensor:
        return self.B_toks + self.B_toks.shape[0]

    def _create_B_TO_C_mapping(self, C_toks: Tensor) -> Dict[Any, Union[int, float]]:
        B_TO_C = {}
        for b_index, b in enumerate(self.B_toks):
            B_TO_C[b.item()] = C_toks[b_index].item()
        return B_TO_C


@dataclass
class OVIncoherentTask:
    cfg: OVIncoherentDatasetConfig = field(default_factory=OVIncoherentDatasetConfig)

    def generate_sample(
        self, cfg: OVIncoherentDatasetConfig
    ) -> tuple[Tensor, list[dict[str, int]]]:
        sample_tok_data = []
        n_zeros = 0
        n_trigrams_in = 0
        sample: list[Any] = [cfg.BOS_tok]
        rand_int = t.randint(low=0, high=cfg.n_tokens, size=(1,))
        while len(sample) < cfg.n_positions:
            sample.append(rand_int.item())
            if rand_int == 0:
                n_zeros += 1
                a_pos = len(sample) - 1
            if n_zeros > n_trigrams_in:
                if 0 < sample[-1] < (cfg.n_skiptrigram + 1):
                    rand_int = t.tensor(sample[-1] + cfg.n_skiptrigram)
                    n_trigrams_in += 1
                    b_pos = len(sample) - 1
                    sample_tok_data.append(
                        {"a_pos": a_pos, "b_pos": b_pos, "b_tok": sample[-1]}
                    )
                else:
                    rand_int = t.randint(low=1, high=cfg.n_tokens, size=(1,))
            else:
                rand_int = t.randint(low=0, high=cfg.n_tokens, size=(1,))
        sample_tensor = t.tensor(sample, dtype=t.int)
        return sample_tensor, sample_tok_data

    def generate_batches(
        self,
        cfg: OVIncoherentDatasetConfig,
    ) -> tuple[
        Int[Tensor, "n_batches n_samples_per_batch n_pos"],
        dict[Any, list[dict[str, int]]],
    ]:
        data: Int[Tensor, "n_batches n_samples_per_batch n_pos"] = t.empty(
            cfg.n_batches,
            cfg.n_samples_per_batch,
            cfg.n_positions,
            dtype=t.int32,
            device=cfg.device,
        )
        tok_data: dict[Any, list[dict[str, int]]] = defaultdict(list)
        t.manual_seed(cfg.seed)
        range_n_batches = tqdm(range(cfg.n_batches), leave=False)
        for batch_index in range_n_batches:
            for sample_index in range(cfg.n_samples_per_batch):
                sample, sample_tok_data = self.generate_sample(cfg)
                data[batch_index, sample_index] = sample
                tok_data[batch_index, sample_index] = sample_tok_data
        return data, tok_data

    def __post_init__(self, cfg: Optional[OVIncoherentDatasetConfig] = None) -> None:
        if cfg is not None:
            self.cfg = OVIncoherentDatasetConfig()
        self.data, self.tok_data = self.generate_batches(self.cfg)
