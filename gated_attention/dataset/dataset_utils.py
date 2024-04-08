from collections import defaultdict

import torch as t

from gated_attention.dataset.toy_datasets import OVIncoherentTask


def get_tok_count(dataset: OVIncoherentTask) -> dict[int, float]:
    """
    Returns a dictionary of how many tokens of each type are in the dataset (ignoring BOS)
    """
    ds_tok_count: dict[int, float] = defaultdict(float)
    for batch in dataset.data:
        for sample in batch:
            for tok in sample:
                tok = tok.item()
                if tok != dataset.cfg.BOS_tok:
                    ds_tok_count[tok] += 1 / (
                        dataset.data.shape[0]
                        * dataset.data.shape[1]
                        * dataset.data.shape[2]
                    )

    return ds_tok_count


def check_uniformity(
    dataset: OVIncoherentTask,
) -> bool:  # < ---- This isn't working as expected
    """Check the dataset is appoximately uniform"""
    ds_tok_count = get_tok_count(dataset)
    ds_tok_count_wo_zero = {i: ds_tok_count[i] for i in ds_tok_count if i != 0}
    uniform_occurances = 1 / len(ds_tok_count)
    tolerance = 1.5
    token_most_occurances = max(
        ds_tok_count_wo_zero, key=lambda k: ds_tok_count_wo_zero[k]
    )
    token_least_occurances = min(
        ds_tok_count_wo_zero, key=lambda k: ds_tok_count_wo_zero[k]
    )
    most_occurances = ds_tok_count_wo_zero[token_most_occurances]
    least_occurances = ds_tok_count_wo_zero[token_least_occurances]
    if (least_occurances < (1 / tolerance) * uniform_occurances) or (
        most_occurances > (tolerance) * uniform_occurances
    ):
        print(
            f"Not enough [{token_least_occurances}] tokens or too many [{token_most_occurances}] tokens."
        )
        return False
    return True


def get_tok_dict(dataset: OVIncoherentTask) -> dict[int, list[tuple[int, int]]]:
    """
    Returns a dictionary of B tokens and corresponding examples (batch idx, sample idx)
    """
    tok_dict: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for batch_idx in range(dataset.cfg.n_batches):
        for sample_idx in range(dataset.cfg.n_samples_per_batch):
            tok_data_list = dataset.tok_data[batch_idx, sample_idx]
            for token_set in tok_data_list:
                b_tok = token_set["b_tok"]
                tok_dict[b_tok].append((batch_idx, sample_idx))
    return tok_dict


def check_b_tokens_in_ds(dataset: OVIncoherentTask) -> bool:
    """
    Check that all B tokens exist in the dataset
    """
    tok_dict = get_tok_dict(dataset)
    n_toks_in_tok_dict = len(tok_dict)
    n_skip_trigrams = dataset.cfg.n_skiptrigram
    if n_toks_in_tok_dict != n_skip_trigrams:
        print(
            f"There is a mismatch between number of B toks: {n_skip_trigrams} and number of tok_dict keys: {n_toks_in_tok_dict}"
        )
        return False
    b_toks_in_tok_dict = set(tok_dict.keys())
    all_b_toks = set(dataset.cfg.B_toks.tolist())
    if b_toks_in_tok_dict != all_b_toks:
        print(
            f"There are B tokens missing from the dataset, b_toks in tok dict: {b_toks_in_tok_dict}"
        )
        return False

    return True


def check_min_examples(dataset: OVIncoherentTask, min_examples: int) -> bool:
    """
    Check that all B tokens have at least min_examples examples
    """
    tok_dict = get_tok_dict(dataset)
    if min_examples < 10:
        print(f"min_examples should be greater than 10, min_examples: {min_examples}")
        return False

    for tok in tok_dict:
        if len(tok_dict[tok]) < min_examples:
            print(
                "Not enough examples for B token "
                + str(tok)
                + " with "
                + str(len(tok_dict[tok]))
                + " examples."
            )
            return False

    return True


def check_correct_C_tokens(dataset: OVIncoherentTask) -> bool:
    """
    Check if the C token for each example is correct
    """
    c_pos_out_of_bounds_dict: defaultdict[int, int] = defaultdict(int)
    for batch_idx, batch in enumerate(dataset.data):
        for sample_idx, sample in enumerate(batch):
            tok_data_list = dataset.tok_data[batch_idx, sample_idx]
            for token_set in tok_data_list:
                b_pos = token_set["b_pos"]
                b_tok = token_set["b_tok"]
                c_pos = b_pos + 1
                c_tok = b_tok + dataset.cfg.n_skiptrigram
                if c_pos == dataset.cfg.n_positions:
                    c_pos_out_of_bounds_dict[c_tok] += 1
                    continue
                if sample[c_pos] != c_tok:
                    print("C token not correct.")
                    return False
    return True


def out_of_bounds_C_toks(dataset: OVIncoherentTask) -> bool:
    """
    Check if the number of out of bounds C toks is too high
    """
    tok_dict = get_tok_dict(dataset)
    min_examples_b_tok = min([len(tok_dict[tok]) for tok in tok_dict])
    c_pos_out_of_bounds_dict: defaultdict[int, int] = defaultdict(int)
    for batch_idx, batch in enumerate(dataset.data):
        for sample_idx, sample in enumerate(batch):
            tok_data_list = dataset.tok_data[batch_idx, sample_idx]
            for token_set in tok_data_list:
                b_pos = token_set["b_pos"]
                b_tok = token_set["b_tok"]
                c_pos = b_pos + 1
                c_tok = b_tok + dataset.cfg.n_skiptrigram
                if c_pos == dataset.cfg.n_positions:
                    c_pos_out_of_bounds_dict[c_tok] += 1

    c_pos_out_of_bounds_global = max(c_pos_out_of_bounds_dict.values())
    if c_pos_out_of_bounds_global > 0.5 * min_examples_b_tok:
        print(f"C pos out of bounds on {c_pos_out_of_bounds_global} occurances.")
        return False
    return True


def check_valid_ds(dataset: OVIncoherentTask, min_examples: int) -> bool:
    check_b_toks = check_b_tokens_in_ds(dataset=dataset)
    check_min_b_toks = check_min_examples(dataset=dataset, min_examples=min_examples)
    check_correct_c_toks = check_correct_C_tokens(dataset=dataset)
    check_out_of_bounds_C_toks = out_of_bounds_C_toks(dataset=dataset)
    check_uniform = check_uniformity(dataset=dataset)
    if t.tensor(
        [
            check_b_toks,
            check_min_b_toks,
            check_correct_c_toks,
            check_out_of_bounds_C_toks,
            check_uniform,
        ]
    ).all():
        return True
    return False
