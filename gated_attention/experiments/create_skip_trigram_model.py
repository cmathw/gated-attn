import torch as t

from gated_attention.dataset.toy_datasets import OVIncoherentDatasetConfig
from gated_attention.experiments.analysis_utils import check_no_attn_super
from gated_attention.modelling.traditional_transformer import (
    SimplifiedAttnOnlyModelConfig,
)
from gated_attention.train.train_traditional import (
    OV_Incoherent_Training_Config,
    create_model,
    train_model,
)

if __name__ == "__main__":
    t.manual_seed(0)
    dataset_cfg = OVIncoherentDatasetConfig(
        B_toks=t.tensor([1, 2, 3]), device="cuda:0", seed=0
    )
    model_cfg = SimplifiedAttnOnlyModelConfig(
        dataset_cfg=dataset_cfg,
        n_heads=2,
        d_head=1,
        output_bias=False,
        device="cuda:0",
        seed=3,
    )

    train_cfg = OV_Incoherent_Training_Config(model_cfg, dataset_cfg, lr=1e-3)

    orig_model = create_model(train_cfg)
    print(f"Device: {orig_model.device}")
    print(f"Seed: {orig_model.cfg.seed}")
    train_cfg.epochs = 1_000
    train_cfg.save_file_name = f"saved_models/toy_model_{model_cfg.n_heads}H_{dataset_cfg.n_skiptrigram}ST_{model_cfg.seed}.st"
    train_cfg.save_file_overwrite = True

    print(orig_model.device)
    orig_model.to("cuda:0")
    model, _ = train_model(orig_model, training_cfg=train_cfg)

    print(check_no_attn_super(model, True))
