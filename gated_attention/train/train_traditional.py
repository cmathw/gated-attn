import os
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Callable, Dict

import torch as t
from safetensors.torch import load_model, save_model
from torch import Tensor

from gated_attention.dataset.dataset_utils import check_valid_ds
from gated_attention.dataset.toy_datasets import (
    OVIncoherentDatasetConfig,
    OVIncoherentTask,
)
from gated_attention.modelling.traditional_transformer import (
    SimplifiedAttnOnlyModelConfig,
    VerySimpleAttnOnlyTransformer,
)

SAVED_DIRECTORY = ""


def get_log_probs(
    batch,  # [n_samples, n_pos]
    model: VerySimpleAttnOnlyTransformer,
):
    assert batch.ndim == 2
    input_data = batch[:, :-1]  # [n_samples, n_pos-1]
    logits = model(input_data)  # [n_samples, n_pos-1, n_logits]
    log_probs = logits.log_softmax(dim=-1)  # [n_samples, n_pos-1, n_logits]
    return log_probs


def get_loss(
    batch,  # [n_samples, n_pos]
    model: VerySimpleAttnOnlyTransformer,
):
    correct_output = batch[:, 1:]  # [n_samples, n_pos-1]
    correct_output_logit_indices = correct_output.unsqueeze(
        -1
    )  # [n_samples, n_pos-1, 1]

    log_probs = get_log_probs(batch, model)  # [n_samples, n_pos-1, n_logits]
    log_probs_per_position = log_probs.gather(
        dim=-1, index=correct_output_logit_indices
    ).squeeze(-1)
    return -log_probs_per_position.mean()  # Float


# Create a store of error count, total count, accuracy
Score = float
learnt_tokens_factory: Callable = set
n_learnt_tokens_factoy: Callable = int
metrics_factory: Callable = lambda: defaultdict(Score)
b_token_factory: Callable = lambda: defaultdict(BToken)


@dataclass
class BToken:
    metrics: Dict[str, Score] = field(default_factory=metrics_factory)

    def __repr__(self):
        return f"BToken(Metrics={dict(self.metrics)})"


@dataclass
class Epoch:
    b_token: Dict[int, BToken] = field(default_factory=b_token_factory)
    metrics: Dict[str, Score] = field(default_factory=metrics_factory)
    learnt_tokens: set = field(default_factory=learnt_tokens_factory)
    n_learnt_tokens: int = field(default_factory=n_learnt_tokens_factoy)

    def __repr__(self):
        # return f"Epoch(BToken={dict(self.b_token)}, LearntTokens={self.learnt_tokens}, NLearntTokens={self.n_learnt_tokens})"
        return f"Epoch(Metrics={dict(self.metrics)})"

    def __len__(self):
        return len(self.b_token)


def update_correct_b_tok_predictions(
    correct_b_tok_prediction_count: list,
    validation_dataset: OVIncoherentTask,
    batch_index: int,
    B_TO_C: Dict,
    max_prob_predicted_tokens: Tensor,
) -> list:
    for sample_index in range(validation_dataset.data.shape[1]):
        b_list = validation_dataset.tok_data[batch_index, sample_index]
        for tok_data_list in b_list:
            if tok_data_list["b_pos"] == validation_dataset.data.shape[2] - 1:
                continue
            correct_token = B_TO_C[tok_data_list["b_tok"]]
            predicted_token = max_prob_predicted_tokens[
                sample_index, tok_data_list["b_pos"]
            ]
            correct_b_tok_prediction_count.append(
                (correct_token == predicted_token).item()
            )
    return correct_b_tok_prediction_count


@dataclass
class OV_Incoherent_Training_Config:
    model_cfg: SimplifiedAttnOnlyModelConfig
    dataset_cfg: OVIncoherentDatasetConfig

    epochs: int = 5
    lr: float = 0.001
    weight_decay: float = 0.0

    save_file_name: str = ""
    save_file_overwrite: bool = False

    def __post_init__(self):
        assert isinstance(self.model_cfg, SimplifiedAttnOnlyModelConfig)

        self.save_file_name = self.save_file_name
        self.device = self.model_cfg.device


def create_model(
    training_cfg: OV_Incoherent_Training_Config,
) -> VerySimpleAttnOnlyTransformer:
    m_cfg = training_cfg.model_cfg

    print("Creating model...")
    model = VerySimpleAttnOnlyTransformer(m_cfg)
    return model


def train_model(
    model: VerySimpleAttnOnlyTransformer, training_cfg: OV_Incoherent_Training_Config
):
    d_cfg = training_cfg.dataset_cfg
    d_cfg.device = model.device

    optimizer = t.optim.Adam(
        model.parameters(),
        lr=training_cfg.lr,
        weight_decay=training_cfg.weight_decay,
    )

    save_file_path = SAVED_DIRECTORY + training_cfg.save_file_name

    if (
        training_cfg.save_file_name
        and os.path.isfile(save_file_path)
        and not training_cfg.save_file_overwrite
    ):
        load_model(model=model, filename=save_file_path)
        return model, None

    print("Generating data...")
    training_dataset = OVIncoherentTask(d_cfg)
    # dataset.tok_data[batch_index, sample_index] = [{"a_pos": a_pos, "b_pos": b_pos, "b_tok": b_tok}, ...]
    validation_cfg = replace(
        d_cfg, n_samples_per_batch=d_cfg.n_samples_per_batch // 2, seed=42
    )
    validation_dataset = OVIncoherentTask(cfg=validation_cfg)

    print("Checking Datasets...")
    train_ds_valid = check_valid_ds(training_dataset, min_examples=1000)
    val_ds_valid = check_valid_ds(validation_dataset, min_examples=100)
    print(f"Training Dataset Valid: {train_ds_valid}")
    print(f"Validation Dataset Valid: {val_ds_valid}")
    assert train_ds_valid
    assert val_ds_valid

    B_TO_C = training_dataset.cfg.B_TO_C

    train_loss_list = []
    indices = []
    across_epoch_val_loss_list = []
    across_epoch_accuracy_list = []

    # Training
    for epoch in range(training_cfg.epochs):
        batch_index = None
        progress_bar = range(training_dataset.cfg.n_batches)
        for batch_index in progress_bar:
            optimizer.zero_grad()
            batch = training_dataset.data[batch_index].to(
                device="cuda:0"
            )  # [n_samples, n_pos]
            correct_outputs = batch[:, 1:]  # [n_samples, n_pos-1]
            correct_outputs_logit_indices = correct_outputs.unsqueeze(-1).to(
                t.int64
            )  # [n_samples, n_pos-1, 1]
            input_data = batch[:, :-1].long()  # [n_samples, n_pos-1]
            logits = model(input_data)  # [n_samples, n_pos-1, n_logits]
            log_probs = logits.log_softmax(dim=-1)  # [n_samples, n_pos-1, n_logits]
            max_prob_predicted_tokens = log_probs.argmax(-1)  # [n_samples, n_pos-1]

            log_probs_per_position = log_probs.gather(
                dim=-1, index=correct_outputs_logit_indices
            ).squeeze(-1)

            loss = -log_probs_per_position.mean()  # [n_samples, n_pos-1]
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:02}, Train Loss = {loss:.4f}")

        del batch_index
        indices.append(epoch)
        train_loss_list.append(loss.item())

        # Validation
        with t.inference_mode():
            correct_b_tok_predictions: list = []
            val_loss_list = []
            for batch_index, batch in enumerate(validation_dataset.data):
                batch = validation_dataset.data[batch_index]
                batch.to(training_cfg.device)
                correct_outputs = batch[:, 1:]  # [n_samples, n_pos-1]
                correct_outputs_logit_indices = correct_outputs.unsqueeze(
                    -1
                )  # [n_samples, n_pos-1, 1]
                input_data = batch[:, :-1].long()  # [n_samples, n_pos-1]
                logits = model(input_data)  # [n_samples, n_pos-1, n_logits]
                logs_for_predicted_logits = logits.log_softmax(
                    dim=-1
                )  # [n_samples, n_pos-1, n_logits]
                max_prob_predicted_tokens = logs_for_predicted_logits.argmax(
                    -1
                )  # [samples, pos]
                correct_b_tok_predictions = update_correct_b_tok_predictions(
                    correct_b_tok_predictions,
                    validation_dataset,
                    batch_index,
                    B_TO_C,
                    max_prob_predicted_tokens,
                )
                val_loss_list.append(-logs_for_predicted_logits.mean().item())

        accuracy = sum(correct_b_tok_predictions) / len(correct_b_tok_predictions)
        val_loss = sum(val_loss_list) / len(val_loss_list)
        across_epoch_val_loss_list.append(val_loss)
        across_epoch_accuracy_list.append(accuracy)
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch:02}, Validation Loss = {val_loss:.8f}, Accuracy = {accuracy:2f} <-- Only on Skip Trigrams"
            )

    if training_cfg.save_file_name and (
        training_cfg.save_file_overwrite or not os.path.isfile(save_file_path)
    ):
        # t.save(model.state_dict(), save_file_path)
        save_model(model, save_file_path)

    return model, 0
