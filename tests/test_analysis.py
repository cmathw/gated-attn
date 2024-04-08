import pytest
import torch as t

from gated_attention.experiments.analysis_utils import check_no_attn_super
from gated_attention.modelling.traditional_transformer import (
    SimplifiedAttnOnlyModelConfig,
    VerySimpleAttnOnlyTransformer,
)


@pytest.fixture(scope="module")
def untrained_model():
    cfg = SimplifiedAttnOnlyModelConfig()
    model = VerySimpleAttnOnlyTransformer(cfg)
    return model


DEVICE = "cuda:0" if t.cuda.is_available() else "cpu"


def test_check_untrained_no_learnt_dict(untrained_model):
    encoding_dict = check_no_attn_super(untrained_model, True)
    assert encoding_dict["Distributed Repr"] == []


def test_check_untrained_no_learnt_bool(untrained_model):
    assert check_no_attn_super(untrained_model, False)
