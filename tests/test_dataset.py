from dataclasses import replace

import pytest
import torch as t

from gated_attention.dataset.dataset_utils import (
    check_b_tokens_in_ds,
    check_correct_C_tokens,
    check_min_examples,
    check_uniformity,
    check_valid_ds,
    out_of_bounds_C_toks,
)
from gated_attention.dataset.toy_datasets import (
    OVIncoherentDatasetConfig,
    OVIncoherentTask,
)


# Check default dataset initialization
def test_default_dataset_config_init() -> None:
    """Check that the dataset config initialises correctly"""
    config = OVIncoherentDatasetConfig()
    assert config.n_positions == 11
    assert len(config.B_toks) == 5
    assert config.n_batches == 100
    assert config.n_samples_per_batch == 1000
    assert config.seed == 0
    assert t.equal(config.C_toks, config.B_toks + 5)
    assert config.n_skiptrigram == 5
    assert t.equal(config.ABC_toks, t.tensor(range(12)))  # 11
    assert len(config.B_TO_C) == 5
    assert config.B_TO_C[config.B_toks[0].item()] == config.C_toks[0].item()
    assert config.B_TO_C[config.B_toks[-1].item()] == config.C_toks[-1].item()
    assert config.n_total_samples == 1000 * 100
    assert config.n_tokens == len(config.ABC_toks)  # + 1  # + 1 for BOS token


def create_rand_int() -> int:
    return int(t.randint(5, 10, (1,)).item())


def create_rand_int_large() -> int:
    return int(t.randint(100, 1000, (1,)).item())


def create_rand_parameters() -> tuple[int, ...]:
    params = [create_rand_int() for x in range(5)]
    params[3] = create_rand_int_large()
    return tuple(params)


# Check parameterized dataset initialization
@pytest.mark.parametrize(
    "n_positions, n_B_toks, n_batches, n_samples_per_batch, seed",
    [
        create_rand_parameters(),
        create_rand_parameters(),
        create_rand_parameters(),
        create_rand_parameters(),
        create_rand_parameters(),
    ],
)
def test_dataset_config_init(
    n_positions: int, n_B_toks: int, n_batches: int, n_samples_per_batch: int, seed: int
) -> None:
    """Check that the dataset config initialises correctly"""
    config = OVIncoherentDatasetConfig(
        n_positions=n_positions,
        A_toks=t.tensor([0]),
        B_toks=t.tensor(range(1, n_B_toks + 1)),
        device="cpu",
        n_batches=n_batches,
        n_samples_per_batch=n_samples_per_batch,
        seed=seed,
    )

    assert config.n_positions == n_positions
    assert len(config.B_toks) == n_B_toks
    assert config.n_batches == n_batches
    assert config.n_samples_per_batch == n_samples_per_batch
    assert config.seed == seed
    assert t.equal(config.C_toks, config.B_toks + n_B_toks)
    assert config.n_skiptrigram == n_B_toks
    assert t.equal(config.ABC_toks, t.tensor(range(2 * (n_B_toks + 1))))
    assert len(config.B_TO_C) == n_B_toks
    assert config.B_TO_C[config.B_toks[0].item()] == config.C_toks[0].item()
    assert config.B_TO_C[config.B_toks[-1].item()] == config.C_toks[-1].item()
    assert config.n_total_samples == n_batches * n_samples_per_batch
    assert config.n_tokens == len(config.ABC_toks)


# Check Default Dataset Checks
def create_small_dataset() -> OVIncoherentTask:
    default_cfg = OVIncoherentDatasetConfig()
    small_cfg = replace(default_cfg, n_batches=1, n_samples_per_batch=1000)
    small_dataset = OVIncoherentTask(cfg=small_cfg)
    return small_dataset


small_dataset: OVIncoherentTask = create_small_dataset()


def test_default_dataset_checks() -> None:
    min_examples = 50
    assert check_b_tokens_in_ds(small_dataset)
    assert check_min_examples(dataset=small_dataset, min_examples=min_examples)
    assert check_correct_C_tokens(dataset=small_dataset)
    assert out_of_bounds_C_toks(dataset=small_dataset)
    assert check_uniformity(dataset=small_dataset)
    assert check_valid_ds(dataset=small_dataset, min_examples=min_examples)


# Check Parametrized Dataset Checks
@pytest.mark.parametrize(
    "n_positions, n_B_toks, n_batches, n_samples_per_batch, seed",
    [
        create_rand_parameters(),
        create_rand_parameters(),
        create_rand_parameters(),
        create_rand_parameters(),
        create_rand_parameters(),
    ],
)
def test_parametrize_dataset_checks(
    n_positions: int, n_B_toks: int, n_batches: int, n_samples_per_batch: int, seed: int
) -> None:
    config = OVIncoherentDatasetConfig(
        n_positions=n_positions,
        A_toks=t.tensor([0]),
        B_toks=t.tensor(range(1, n_B_toks + 1)),
        device="cpu",
        n_batches=n_batches,
        n_samples_per_batch=n_samples_per_batch,
        seed=seed,
    )
    config.n_batches = 1
    config.n_samples_per_batch = 10_000
    min_examples = 50
    parametrize_dataset = OVIncoherentTask(cfg=config)
    assert check_b_tokens_in_ds(parametrize_dataset), "check_b_tokens_in_ds"
    assert check_min_examples(
        dataset=parametrize_dataset, min_examples=min_examples
    ), "check_min_examples"
    assert check_correct_C_tokens(dataset=parametrize_dataset), "check_correct_C_tokens"
    assert out_of_bounds_C_toks(dataset=parametrize_dataset), "out_of_bounds_C_toks"
    assert check_uniformity(dataset=parametrize_dataset), "check_uniformity"
    assert check_valid_ds(
        dataset=parametrize_dataset, min_examples=min_examples
    ), "check_valid_ds"
