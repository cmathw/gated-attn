import torch as t
import yaml
from safetensors.torch import load_model

from gated_attention.dataset.toy_datasets import (
    OVIncoherentDatasetConfig,
    OVIncoherentTask,
)
from gated_attention.experiments.analysis_utils import check_no_attn_super
from gated_attention.modelling.modified_attention import ModifiedAttentionTransformer
from gated_attention.modelling.traditional_transformer import (
    SimplifiedAttnOnlyModelConfig,
    VerySimpleAttnOnlyTransformer,
)
from gated_attention.train.train_gated_attention import train_modified_attn_block

"""
Note: In config.yaml we have listed all of the hyperparameters that we used to train the gated attention
blocks in the post. We include this script to replicate these results but due to the randomness inherit to the 
training process, the results may not be exactly the same. We include the gated attention blocks outlined in
the post in the saved_models directory for reference. These can be loaded in with `loading_gated_attention_model.py`
"""

if __name__ == "__main__":

    # Load the configuration from YAML
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Change this to experiment of interest (all other params will adjust below)
    experiment_name = "4 heads, 5 skip trigrams"

    if experiment_name in config["experiments"]:
        exp_config = config["experiments"][experiment_name]
        n_orig_heads = exp_config["n_orig_heads"]
        b_tok_range = exp_config["b_tok_range"]
        epochs = exp_config["epochs"]
        n_batches = exp_config["n_batches"]
        n_samples_per_batch = exp_config["n_samples_per_batch"]
        expansion_factor = exp_config["expansion_factor"]
        alpha_reg_attn_gate = exp_config["alpha_reg_attn_gate"]
        lr = exp_config["lr"]
        output_bias = exp_config["output_bias"]
        t_manual_seed = exp_config["t_manual_seed"]
        dataset_seed = exp_config["dataset_seed"]
        model_seed = exp_config["model_seed"]
        path = exp_config["path"]

    else:
        print(f"Experiment named '{experiment_name}' not found in config.")

    t.manual_seed(t_manual_seed)

    DEVICE = "cuda:0" if t.cuda.is_available() else "cpu"

    dataset_cfg = OVIncoherentDatasetConfig(
        B_toks=t.arange(1, b_tok_range + 1), device=DEVICE, seed=dataset_seed
    )

    model_cfg = SimplifiedAttnOnlyModelConfig(
        dataset_cfg=dataset_cfg,
        n_heads=n_orig_heads,
        d_head=1,
        device=DEVICE,
        output_bias=output_bias,
    )
    dataset = OVIncoherentTask(cfg=dataset_cfg)
    orig_model = VerySimpleAttnOnlyTransformer(model_cfg)
    load_model(model=orig_model, filename=path)

    print(check_no_attn_super(orig_model, True))

    modified_attn_block = train_modified_attn_block(
        orig_model=orig_model,
        dataset=dataset,
        n_epochs=epochs,
        expansion_factor=expansion_factor,
        alpha_reg_attn_gate=alpha_reg_attn_gate,
        lr=lr,
        seed=model_seed,
    )

    modified_model = ModifiedAttentionTransformer(
        orig_model=orig_model, modified_attn=modified_attn_block
    )

    print(check_no_attn_super(modified_model, True))
