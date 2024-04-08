import torch as t
import torch.nn as nn

from gated_attention.dataset.toy_datasets import OVIncoherentTask
from gated_attention.modelling.modified_attention import ModifiedAttention
from gated_attention.modelling.traditional_transformer import (
    VerySimpleAttnOnlyTransformer,
)


def train_modified_attn_block(
    orig_model: VerySimpleAttnOnlyTransformer,
    dataset: OVIncoherentTask,
    expansion_factor: int,
    n_epochs: int,
    alpha_reg_attn_gate: int,
    seed: int = 0,
    lr: float = 5e-4,
) -> ModifiedAttention:
    train_loss_list = []
    indices = []
    modified_attn_block = ModifiedAttention(
        orig_model=orig_model, expansion_factor=expansion_factor, seed=seed
    ).to(orig_model.cfg.device)
    dataset.cfg.B_TO_C
    print(f"Training model with {modified_attn_block.n_heads} heads.")
    dataset.data = dataset.data.long().to(orig_model.device)

    # Initial and final learning rates
    optimizer = t.optim.Adam(modified_attn_block.parameters(), lr=lr)

    for epoch in range(n_epochs):
        batch_index = None
        progress_bar = range(dataset.cfg.n_batches)
        for batch_index in progress_bar:
            optimizer.zero_grad()
            batch = dataset.data[batch_index]
            orig_model(batch)
            normalised_resid_pre = orig_model.cache["embed"].clone()
            original_attn_out = orig_model.cache["attn_out"].clone()
            modified_output = modified_attn_block(normalised_resid_pre)
            mse_loss = nn.MSELoss(reduction="mean")
            model_output_flat = modified_output.view(-1, modified_output.size(-1))
            target_flat = original_attn_out.view(-1, original_attn_out.size(-1))
            reg_attn_gate = (
                alpha_reg_attn_gate * modified_attn_block.cache["reg_attn_gate"]
            )
            reconstruction_loss = mse_loss(model_output_flat, target_flat)
            loss = reconstruction_loss + reg_attn_gate
            loss.backward(retain_graph=True)
            optimizer.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} reconstruction loss: {reconstruction_loss.item()} reg attn gate: {reg_attn_gate.item()}"
            )
            print(f"Epoch {epoch} loss: {loss.item()}")

        del batch_index
        indices.append(epoch)
        train_loss_list.append(loss.item())

    return modified_attn_block
