from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from itertools import combinations
from typing import Dict, Union

import pygraphviz as pgv
import torch as t
from IPython.display import Image, display
from torch import Tensor
from tqdm import tqdm

from gated_attention.dataset.toy_datasets import OVIncoherentTask
from gated_attention.modelling.modified_attention import ModifiedAttentionTransformer
from gated_attention.modelling.traditional_transformer import (
    VerySimpleAttnOnlyTransformer,
)
from gated_attention.train.train_traditional import BToken


def generate_combinations_excluding_number(numbers, exclude):
    """
    Generate all combinations of numbers from a list excluding a specific number.

    Parameters:
    - numbers: List[int], the original list of numbers.
    - exclude: int, the number to be excluded from combinations.

    Returns:
    - List[List[int]], a list of all combinations excluding the specified number.
    """
    # Remove the number to be excluded
    filtered_numbers = [num for num in numbers if num != exclude]

    # Generate all combinations of the filtered list
    all_combinations = []
    for r in range(1, len(filtered_numbers) + 1):
        all_combinations += list(combinations(filtered_numbers, r))

    # Convert tuples in the list to lists
    all_combinations = [list(comb) for comb in all_combinations]

    return all_combinations


def generate_all_combinations(numbers):
    """
    Generate all combinations of numbers from a list.

    Parameters:
    - numbers: List[int], the original list of numbers.

    Returns:
    - List[List[int]], a list of all combinations of the numbers.
    """
    # Generate all combinations of the list
    all_combinations = []
    for r in range(1, len(numbers) + 1):
        all_combinations += list(combinations(numbers, r))

    # Convert tuples in the list to lists
    all_combinations = [list(comb) for comb in all_combinations]

    return all_combinations


def recursively_remove_all_other_heads_alt(
    representation: int,
    head_of_interest: int,
    val_dataset: OVIncoherentTask,
    model: VerySimpleAttnOnlyTransformer,
    verbose: bool = False,
):
    """
    Given an attention head we think is solely responsible for a representation. Retain this head and recursively
    remove all combinations of all other heads that exist within the block. If the representation is always
    retained, then we know that this head is acting solely in producing a given representation.

    Args:
        repr (int): The representation that we expect the head is encoding.
        head_of_interest (int): The head in which is encoding the representation.
        modified_model (VerySimpleAttnOnlyTransformer): The one layer model in which the head exists
    """

    corrects = 0
    incorrects = 0
    total = 0

    # dataset.tok_data[batch_index, sample_index] = [{"a_pos": a_pos, "b_pos": b_pos, "b_tok": b_tok}, ...]
    # val_dataset.data[0] <-- single batch
    single_batch = val_dataset.data[0]
    for sample_idx, sample in enumerate(single_batch):

        # Only look at samples with single skip trigram present for simplicity (some will have 0, some will have > 1)
        if len(val_dataset.tok_data[0, sample_idx]) != 1:
            continue

        # Get b_token from the sample
        b_token = val_dataset.tok_data[0, sample_idx][0]["b_tok"]

        # If the token is not the representation we are looking for, skip
        if b_token != representation:
            continue

        # Run the sample through the model
        sample = sample.to(model.device).unsqueeze(0)  # [1, sample, pos]
        model(sample)

        # val_dataset.tok_data = [batch_idx, sample_idx][{"a_pos": a_pos, "b_pos": b_pos, "b_tok": b_tok}, ...]

        # Get b_token position and correct token
        b_token_pos = val_dataset.tok_data[0, sample_idx][0]["b_pos"]
        correct_token = b_token + val_dataset.cfg.n_skiptrigram

        list_of_all_heads = [x for x in range(model.cfg.n_heads)]
        all_combinations_beside_HOI = generate_combinations_excluding_number(
            list_of_all_heads, head_of_interest
        )
        combination_correct = 0

        for heads_to_remove in all_combinations_beside_HOI:
            combined_head_outputs = t.zeros_like(
                model.cache["attn_out_pre_per_head"][0, -2, 0]
            )
            for head in range(model.cfg.n_heads):
                if head not in heads_to_remove:
                    combined_head_outputs += model.cache["attn_out_pre_per_head"][
                        0, b_token_pos, head
                    ]
            if t.argmax(combined_head_outputs).item() == correct_token:
                combination_correct += 1

        if len(all_combinations_beside_HOI) == combination_correct:
            corrects += 1

        else:
            incorrects += 1

        total += 1
    if verbose:
        print(f"Corrects: {corrects}, Incorrects: {incorrects}, Total: {total}")
    if corrects / total > 0.99:
        return True
    else:
        return False


def update_b_tok_info(
    model: VerySimpleAttnOnlyTransformer,
    dataset: OVIncoherentTask,
    B_TO_C: Dict,
    max_prob_predicted_tokens,
) -> Dict:
    b_tok_info: Dict[int, BToken] = defaultdict(BToken)
    for tok_data_entry_idx, tok_data_entries in dataset.tok_data.items():
        sample_index = tok_data_entry_idx[1]
        for sample_tok_data in tok_data_entries:
            if sample_tok_data["b_pos"] == dataset.cfg.n_positions - 1:
                continue
            b_token = sample_tok_data["b_tok"]
            b_pos = sample_tok_data["b_pos"]
            predicted_token = max_prob_predicted_tokens[sample_index, b_pos]
            correct_token = B_TO_C[b_token]
            head_outputs = model.cache["attn_out_pre_per_head"][
                sample_index, b_pos
            ]  # [head_idx, d_model]
            total_count = 1 + b_tok_info[b_token].metrics["total_count"]
            accumulated_head_output = (
                head_outputs + b_tok_info[b_token].metrics["accumulated_head_output"]
            )
            error = (correct_token != predicted_token).item()
            error_count = error + b_tok_info[b_token].metrics["error_count"]
            total_count = 1 + b_tok_info[b_token].metrics["total_count"]

            b_tok_info[b_token].metrics[
                "accumulated_head_output"
            ] = accumulated_head_output
            b_tok_info[b_token].metrics["total_count"] = total_count
            b_tok_info[b_token].metrics["error_count"] = error_count
            b_tok_info[b_token].metrics["accuracy"] = (
                total_count - error_count
            ) / total_count
            b_tok_info[b_token].metrics["average_head_output"] = (
                accumulated_head_output / total_count
            )
            b_tok_info[b_token].metrics["summed_head_output"] = (
                b_tok_info[b_token].metrics["average_head_output"].sum(0)
            )

    return b_tok_info


def check_no_attn_super(
    model: Union[VerySimpleAttnOnlyTransformer, ModifiedAttentionTransformer],
    return_dict: bool,
):
    """ """
    b_tok_info = BToken()
    model_dataset_cfg = model.cfg.dataset_cfg
    dataset_cfg_for_check = replace(
        model_dataset_cfg, n_batches=1, n_samples_per_batch=100_000, seed=1
    )
    val_dataset = OVIncoherentTask(cfg=dataset_cfg_for_check)
    val_dataset.data = val_dataset.data.long()
    B_TO_C = val_dataset.cfg.B_TO_C
    n_heads = model.cfg.n_heads
    b_toks = model.cfg.dataset_cfg.B_toks
    logits = model(val_dataset.data.squeeze())
    log_probs = logits.log_softmax(dim=-1)
    max_prob_predicted_tokens = log_probs.argmax(-1)
    b_tok_info = update_b_tok_info(
        model=model,
        dataset=val_dataset,
        B_TO_C=B_TO_C,
        max_prob_predicted_tokens=max_prob_predicted_tokens,
    )
    set_learnt_tokens = set()
    for b_tok in model.cfg.dataset_cfg.B_toks:
        b_tok = b_tok.item()
        if b_tok_info[b_tok].metrics["accuracy"] > 0.99:
            set_learnt_tokens.add(b_tok)

    head_to_encodings: dict[str, list] = {}

    for x in range(n_heads):
        head_to_encodings[f"Head {x}"] = []
    head_to_encodings["Distributed Repr"] = []
    head_to_encodings["Not Learnt"] = []
    all_encodings = []
    print(f"Learnt Tokens: {set_learnt_tokens}")

    model_dataset_cfg = model.cfg.dataset_cfg
    dataset_cfg_for_recursive = replace(
        model_dataset_cfg, n_batches=1, n_samples_per_batch=1000, seed=42
    )
    val_dataset_recursive = OVIncoherentTask(cfg=dataset_cfg_for_recursive)

    for b_tok in tqdm(set_learnt_tokens, total=len(set_learnt_tokens), leave=False):
        correct_c_tok = b_tok + val_dataset.cfg.n_skiptrigram
        for head_idx in range(n_heads):

            # Comparing effect of removing any combination of all other heads
            if recursively_remove_all_other_heads_alt(
                representation=b_tok,
                head_of_interest=head_idx,
                val_dataset=val_dataset_recursive,
                model=model,
            ):
                head_to_encodings[f"Head {head_idx}"].append(b_tok)
                all_encodings.append(b_tok)

        if b_tok not in all_encodings:
            summed_head_outputs_for_b_tok = b_tok_info[b_tok].metrics[
                "summed_head_output"
            ]
            if t.argmax(summed_head_outputs_for_b_tok) == correct_c_tok:
                head_to_encodings["Distributed Repr"].append(b_tok)
                all_encodings.append(b_tok)

    for b_tok in b_toks:
        b_tok = b_tok.item()
        if b_tok not in set_learnt_tokens:
            head_to_encodings["Not Learnt"].append(b_tok)

    if return_dict:
        return head_to_encodings

    if not head_to_encodings["Distributed Repr"]:
        return True
    else:
        return False


def auto_forward_graph(
    model: VerySimpleAttnOnlyTransformer,
    example: Tensor,
    show_per_head: bool = True,
    save: bool = False,
) -> None:
    """
    Creates a graph of the forward pass of the model.
    """
    logits = model(example)
    graph = pgv.AGraph(directed=True, strict=True)
    t.set_printoptions(precision=2, sci_mode=False)
    graph.add_node("input", label=f"{example.data}", shape="ellipse", color="blue")
    graph.add_node("logits", label=f"{logits.data}", shape="ellipse", color="green")
    graph.add_node(
        "output", label=f"{logits.argmax(dim=-1).data}", shape="ellipse", color="blue"
    )
    for key, value in dict(model.named_parameters()).items():
        if "attn.W_" in key:
            for head_idx in range(model.cfg.n_heads):
                graph.add_node(
                    key + f"_head_{head_idx}",
                    label=f"{key}_head_{head_idx}: {value.data[head_idx]}",
                    shape="box",
                    color="red",
                )
        elif "attn.b_" in key and "attn.b_O" not in key:
            for head_idx in range(model.cfg.n_heads):
                graph.add_node(
                    key + f"_head_{head_idx}",
                    label=f"{key}_head_{head_idx}: {value.data[head_idx]}",
                    shape="box",
                    color="red",
                )
        else:
            graph.add_node(key, label=f"{key}: {value.data}", shape="box", color="red")

    per_head_activation_names = ["q", "k", "v", "z", "q_pre", "k_pre", "v_pre"]
    per_head_activation_names_alt = ["attn_scores_unmasked", "attn_scores", "pattern"]

    for activation_name, activation in model.cache.items():
        if activation_name in per_head_activation_names:
            for head in range(model.cfg.n_heads):
                graph.add_node(
                    activation_name + f"_head_{head}",
                    label=f"{activation_name}_head_{head}: {activation.data[:, :, head, :]}",
                    shape="ellipse",
                    color="green",
                )
        elif activation_name in per_head_activation_names_alt:
            for head in range(model.cfg.n_heads):
                graph.add_node(
                    activation_name + f"_head_{head}",
                    label=f"{activation_name}_head_{head}: {activation.data[:, head, :, :]}",
                    shape="ellipse",
                    color="green",
                )
        else:
            if "attn_out_pre_per_head" not in activation_name:
                graph.add_node(
                    activation_name,
                    label=f"{activation_name}: {activation.data}",
                    shape="ellipse",
                    color="green",
                )

    if show_per_head:
        for head in range(model.cfg.n_heads):
            per_head_output = (
                model.cache["z"][:, :, head, :] @ model.attn.W_O[head, :, :]
            )
            graph.add_node(
                "per_head_output" + f"_head_{head}",
                label=f"per_head_output_head_{head}: {per_head_output.data}",
                shape="ellipse",
                color="green",
            )

    graph.add_edge("input", "embed")
    for head_idx in range(model.cfg.n_heads):
        graph.add_edge("embed", f"attn.W_Q_head_{head_idx}")
        graph.add_edge("embed", f"attn.W_K_head_{head_idx}")
        graph.add_edge("embed", f"attn.W_V_head_{head_idx}")
        graph.add_edge(f"attn.W_Q_head_{head_idx}", f"q_pre_head_{head_idx}")
        graph.add_edge(f"attn.W_K_head_{head_idx}", f"k_pre_head_{head_idx}")
        graph.add_edge(f"attn.W_V_head_{head_idx}", f"v_pre_head_{head_idx}")
        graph.add_edge(f"q_pre_head_{head_idx}", f"attn.b_Q_head_{head_idx}")
        graph.add_edge(f"k_pre_head_{head_idx}", f"attn.b_K_head_{head_idx}")
        graph.add_edge(f"v_pre_head_{head_idx}", f"attn.b_V_head_{head_idx}")
        graph.add_edge(f"attn.b_Q_head_{head_idx}", f"q_head_{head_idx}")
        graph.add_edge(f"attn.b_K_head_{head_idx}", f"k_head_{head_idx}")
        graph.add_edge(f"attn.b_V_head_{head_idx}", f"v_head_{head_idx}")
        graph.add_edge(f"q_head_{head_idx}", f"attn_scores_unmasked_head_{head_idx}")
        graph.add_edge(f"k_head_{head_idx}", f"attn_scores_unmasked_head_{head_idx}")
        graph.add_edge(
            f"attn_scores_unmasked_head_{head_idx}", f"attn_scores_head_{head_idx}"
        )
        graph.add_edge(f"attn_scores_head_{head_idx}", f"pattern_head_{head_idx}")
        graph.add_edge(f"v_head_{head_idx}", f"z_head_{head_idx}")
        graph.add_edge(f"pattern_head_{head_idx}", f"z_head_{head_idx}")
        graph.add_edge(f"z_head_{head_idx}", f"attn.W_O_head_{head_idx}")
        if show_per_head:
            graph.add_edge(
                f"attn.W_O_head_{head_idx}", f"per_head_output_head_{head_idx}"
            )
            graph.add_edge(f"per_head_output_head_{head_idx}", f"attn_out_pre")
        else:
            graph.add_edge(f"attn.W_O_head_{head_idx}", f"attn_out_pre")
    graph.add_edge("attn_out_pre", "attn.b_O")
    graph.add_edge("attn.b_O", "attn_out")
    graph.add_edge("attn_out", "resid_post")
    graph.add_edge("embed", "resid_post")
    graph.add_edge("resid_post", "unembed")
    graph.add_edge("unembed", "logits")
    graph.add_edge("logits", "output")
    timestamp = datetime.now().strftime("%H-%M-%S--%d-%m-%Y")
    graph_path = f"forward_graph_{timestamp}.png"
    display(Image(graph.draw(format="png", prog="dot")))
    if save:
        graph.draw(graph_path, prog="dot")
