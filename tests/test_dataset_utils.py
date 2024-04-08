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


@pytest.fixture(scope="function")
def create_small_dataset():
    default_config = OVIncoherentDatasetConfig()
    small_cfg = replace(default_config, n_batches=1, n_samples_per_batch=1000)
    return OVIncoherentTask(cfg=small_cfg)


@pytest.fixture(scope="function")
def create_non_uniform_ds(create_small_dataset):
    last_B_tok = create_small_dataset.cfg.B_toks[-1]
    double_specific_B_tok = t.ones_like((create_small_dataset.data)) * last_B_tok
    # Note this datasets cfg and tok_data will be inaccurate
    create_small_dataset.data = t.cat(
        (create_small_dataset.data, double_specific_B_tok), dim=0
    )
    return create_small_dataset


@pytest.fixture(scope="function")
def create_perfectly_uniform_ds(create_small_dataset):
    b_toks = create_small_dataset.cfg.B_toks
    # Note this datasets cfg and tok_data will be inaccurate
    create_small_dataset.data = b_toks.repeat(1, 100, 2)
    return create_small_dataset


@pytest.fixture(scope="function")
def create_invalid_dataset_A(create_small_dataset):
    create_small_dataset.cfg.n_skiptrigram -= 1
    return create_small_dataset


@pytest.fixture(scope="function")
def create_invalid_dataset_B(create_small_dataset):
    create_small_dataset.cfg.B_toks = t.tensor([])
    return create_small_dataset


@pytest.fixture(scope="function")
def create_out_of_bounds_dataset():
    default_config = OVIncoherentDatasetConfig()
    invalid_cfg = replace(
        default_config,
        n_positions=3,
        B_toks=t.tensor([1]),
        n_batches=1,
        n_samples_per_batch=1000,
    )
    invalid_ds = OVIncoherentTask(cfg=invalid_cfg)
    invalid_ds.data[:, :, 0] = 0
    invalid_ds.data[:, :, 1] = 1
    invalid_ds.data[:, :, 2] = 2
    return invalid_ds


def test_check_b_toks_A_Y(create_small_dataset):
    assert check_b_tokens_in_ds(create_small_dataset)


def test_check_non_uniform_dataset(create_non_uniform_ds):
    assert not check_uniformity(create_non_uniform_ds)


def test_check_uniform_dataset(create_perfectly_uniform_ds):
    assert check_uniformity(create_perfectly_uniform_ds)


def test_check_b_toks_A(create_invalid_dataset_A):
    assert not check_b_tokens_in_ds(create_invalid_dataset_A)


def test_check_b_toks_B(create_invalid_dataset_B):
    assert not check_b_tokens_in_ds(create_invalid_dataset_B)


def test_min_examples_too_small(create_small_dataset):
    assert not check_min_examples(create_small_dataset, 5)


def test_not_enough_examples(create_small_dataset):
    assert not check_min_examples(create_small_dataset, 1000)


def test_check_correct_c_toks(create_invalid_dataset_A):
    assert not check_correct_C_tokens(create_invalid_dataset_A)


def test_check_out_of_bounds(create_out_of_bounds_dataset):
    assert not out_of_bounds_C_toks(create_out_of_bounds_dataset)


def test_check_valid_ds(create_non_uniform_ds):
    assert not check_valid_ds(create_non_uniform_ds, min_examples=100)
