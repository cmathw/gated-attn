import torch as t
from analysis_utils import check_no_attn_super
from safetensors.torch import load_model

from gated_attention.dataset.toy_datasets import (
    OVIncoherentDatasetConfig,
    OVIncoherentTask,
)
from gated_attention.modelling.traditional_transformer import (
    SimplifiedAttnOnlyModelConfig,
    VerySimpleAttnOnlyTransformer,
)

DEVICE = "cuda:0" if t.cuda.is_available() else "cpu"

dataset_cfg = OVIncoherentDatasetConfig(
    B_toks=t.tensor([1, 2, 3]),
    device=DEVICE,
)

model_cfg = SimplifiedAttnOnlyModelConfig(
    dataset_cfg=dataset_cfg, n_heads=2, d_head=1, device=DEVICE, output_bias=False
)
dataset = OVIncoherentTask(cfg=dataset_cfg)
orig_model = VerySimpleAttnOnlyTransformer(model_cfg)
# modified_attn_block = ModifiedAttention(
#     orig_model, expansion_factor=2, seed=0, use_duplicate_weights=False
# )

PATH = "saved_models/toy_model_2H_3ST_MANUAL.st"
load_model(model=orig_model, filename=PATH)

# modified_model = ModifiedAttentionTransformer(
#     orig_model=orig_model, modified_attn=modified_attn_block
# )


# modified_model.to(DEVICE)
orig_model.to(DEVICE)

print(check_no_attn_super(orig_model, True))
