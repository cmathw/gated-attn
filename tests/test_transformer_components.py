from dataclasses import dataclass
from typing import Optional

import pytest
import torch as t
from torch import nn

from gated_attention.dataset.toy_datasets import OVIncoherentDatasetConfig
from gated_attention.modelling.traditional_transformer import (
    VerySimpleAttnOnlyTransformer,
)
from gated_attention.modelling.transformer_components.attention import Attention
from gated_attention.modelling.transformer_components.config import (
    SimplifiedAttnOnlyModelConfig,
)
from gated_attention.modelling.transformer_components.embed_unembed import (
    Embed,
    Unembed,
)

DEVICE = "cuda:0" if t.cuda.is_available() else "cpu"


# Test Components
@pytest.fixture(scope="module")
def cfg():
    @dataclass
    class model_cfg:
        dataset_cfg: Optional[OVIncoherentDatasetConfig]
        d_vocab: int
        n_heads: int
        d_head: int
        d_model: int
        init_range: float
        device: str
        output_bias: bool
        seed: int

    return model_cfg(
        dataset_cfg=None,
        d_vocab=512,
        n_heads=2,
        d_head=64,
        d_model=512,
        init_range=0.1,
        device=DEVICE,
        output_bias=True,
        seed=0,
    )


@pytest.fixture(scope="module")
def input_tokens():
    return t.randint(0, 512, (32, 100)).to(device=DEVICE)


@pytest.fixture(scope="module")
def normalized_resid_pre():
    return t.randn(32, 100, 512).to(device=DEVICE)


@pytest.fixture(scope="module")
def unembed_out():
    return t.randn(32, 100, 512).to(device=DEVICE)


# Embed
@pytest.fixture(scope="module")
def embed(cfg):
    return Embed(cfg)


def test_embed_shape(embed, input_tokens, normalized_resid_pre):
    output = embed(input_tokens)
    assert output.shape == normalized_resid_pre.shape


# Attention
@pytest.fixture(scope="module")
def attention(cfg):
    return Attention(cfg)


def test_attn_shape(attention, normalized_resid_pre):
    output = attention(normalized_resid_pre)
    assert output.shape == normalized_resid_pre.shape


# Unembed
@pytest.fixture(scope="module")
def unembed(cfg):
    return Unembed(cfg)


def test_unembed_shape(unembed, normalized_resid_pre, unembed_out):
    output = unembed(normalized_resid_pre)
    assert output.shape == unembed_out.shape


# Test Components
@pytest.fixture(scope="module")
def cfg2():
    @dataclass
    class model_cfg:
        dataset_cfg: Optional[OVIncoherentDatasetConfig]
        d_vocab: int
        n_heads: int
        d_head: int
        d_model: int
        n_layers: int
        init_range: float
        device: str
        output_bias: bool
        seed: int

    return model_cfg(
        dataset_cfg=None,
        d_vocab=5,
        n_heads=10,
        d_head=10,
        d_model=5,
        n_layers=1,
        init_range=0.1,
        device=DEVICE,
        output_bias=True,
        seed=0,
    )


# Memorize dataset
@pytest.fixture(scope="module")
def transformer(cfg2):
    return VerySimpleAttnOnlyTransformer(cfg2)


def test_transformer_shape(transformer):
    dataset = t.randint(0, 5, (100, 100))
    output = transformer(dataset)
    assert output.shape == (dataset.shape[0], dataset.shape[1], 5)


@pytest.mark.slow
def test_memorize_dataset(transformer):
    dataset = t.randint(0, 5, (5, 5))
    dataset = dataset.to(DEVICE)
    optimizer = t.optim.Adam(transformer.parameters())
    for _ in range(2500):
        optimizer.zero_grad()
        logits = transformer(dataset)
        loss = nn.functional.cross_entropy(
            input=logits.reshape(-1, 5), target=dataset.reshape(-1)
        )
        loss.backward()
        optimizer.step()
    assert loss < 1e-3


# Test Default Config
@pytest.fixture(scope="module")
def default_model_config():
    default_config = SimplifiedAttnOnlyModelConfig()
    return VerySimpleAttnOnlyTransformer(default_config)


def test_default_model_init(default_model_config):
    default_dataset_config = OVIncoherentDatasetConfig()
    assert default_model_config.cfg.dataset_cfg == default_dataset_config
    assert default_model_config.cfg.device == DEVICE
    assert default_model_config.cfg.init_range == 0.02
    assert default_model_config.cfg.n_heads == 4
    assert default_model_config.cfg.d_head == 1
    assert default_model_config.cfg.d_vocab == default_dataset_config.n_tokens
    assert default_model_config.cfg.d_model == default_dataset_config.n_tokens
