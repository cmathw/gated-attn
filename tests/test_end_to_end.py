import random

import numpy as np
import pytest
import torch as t
from safetensors.torch import load_model

from gated_attention.dataset.toy_datasets import OVIncoherentDatasetConfig
from gated_attention.experiments.analysis_utils import check_no_attn_super
from gated_attention.modelling.modified_attention import (
    ModifiedAttention,
    ModifiedAttentionTransformer,
)
from gated_attention.modelling.traditional_transformer import (
    SimplifiedAttnOnlyModelConfig,
)
from gated_attention.train.train_traditional import (
    OV_Incoherent_Training_Config,
    create_model,
    train_model,
)


def setup_test_environment(seed):
    t.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = "cuda" if t.cuda.is_available() else "cpu"
    if device == "cuda":
        t.cuda.manual_seed_all(seed)
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False
    return device


@pytest.fixture
def device():
    return setup_test_environment(seed=0)


@pytest.fixture
def model_and_config(device):
    dataset_cfg = OVIncoherentDatasetConfig(
        B_toks=t.tensor([1, 2, 3, 4, 5]),
        device=device,
    )
    model_cfg = SimplifiedAttnOnlyModelConfig(
        dataset_cfg=dataset_cfg,
        n_heads=4,
        d_head=1,
        output_bias=False,
        device=device,
        seed=3,
    )
    train_cfg = OV_Incoherent_Training_Config(model_cfg, dataset_cfg, lr=0.005)
    return create_model(train_cfg), dataset_cfg, train_cfg


@pytest.mark.slow
def test_traditional_training_end_to_end(device, model_and_config):
    model, _, train_cfg = model_and_config
    train_cfg.epochs = 100
    train_cfg.save_file_name = "tests/test_models/toy_model_conventional_4H_5ST.st"
    train_cfg.save_file_overwrite = True

    model.to(device)

    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    trained_model, _ = train_model(model, training_cfg=train_cfg)

    repr_dict = check_no_attn_super(trained_model, True)

    assert len(repr_dict["Distributed Repr"]) > 0


@pytest.mark.slow
def test_gated_attention_load_and_check_end_to_end(device, model_and_config):
    model, _, _ = model_and_config
    modified_attn_block = ModifiedAttention(
        model, expansion_factor=2, seed=0, use_duplicate_weights=False
    )
    PATH = "tests/test_models/gated_block_8H_5ST.st"
    load_model(model=modified_attn_block, filename=PATH)
    modified_model = ModifiedAttentionTransformer(
        orig_model=model, modified_attn=modified_attn_block
    )
    modified_model.to(device)
    gated_attn_repr_dict = check_no_attn_super(modified_model, True)

    assert gated_attn_repr_dict["Head 0"] == [5]
    assert gated_attn_repr_dict["Head 1"] == []
    assert gated_attn_repr_dict["Head 2"] == []
    assert gated_attn_repr_dict["Head 3"] == [4]
    assert gated_attn_repr_dict["Head 4"] == [2]
    assert gated_attn_repr_dict["Head 5"] == [1]
    assert gated_attn_repr_dict["Head 6"] == [3]
    assert gated_attn_repr_dict["Head 7"] == []
    assert gated_attn_repr_dict["Distributed Repr"] == []
    assert gated_attn_repr_dict["Not Learnt"] == []
