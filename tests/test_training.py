import pytest
import torch as t
from safetensors.torch import load_model

from gated_attention.dataset.toy_datasets import OVIncoherentDatasetConfig
from gated_attention.modelling.traditional_transformer import (
    SimplifiedAttnOnlyModelConfig,
    VerySimpleAttnOnlyTransformer,
)
from gated_attention.train.train_traditional import (
    BToken,
    Epoch,
    OV_Incoherent_Training_Config,
    create_model,
    get_log_probs,
    get_loss,
    train_model,
)

DEVICE = "cuda:0" if t.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def model():
    cfg = SimplifiedAttnOnlyModelConfig()
    model = VerySimpleAttnOnlyTransformer(cfg).to(DEVICE)
    return model


@pytest.fixture(scope="module")
def batch():
    n_samples = 100
    n_positions = 10
    return t.randint(0, 11, (n_samples, n_positions), device=DEVICE)


def test_get_log_probs_shape(model: VerySimpleAttnOnlyTransformer, batch):
    log_probs = get_log_probs(batch, model)
    log_probs = log_probs.to(DEVICE)
    assert log_probs.shape == t.Size([100, 9, model.cfg.d_vocab])


def test_get_log_probs_device(model: VerySimpleAttnOnlyTransformer, batch):
    batch = batch.to(DEVICE)
    model = model.to(DEVICE)

    log_probs = get_log_probs(batch, model)
    assert log_probs.device == t.device(DEVICE)


def test_get_log_probs_gradients(model: VerySimpleAttnOnlyTransformer, batch):
    input_data = batch[:, :-1]
    logits = model(input_data)  # [32, 9, n_logits]

    loss = logits.sum()
    loss.backward()
    assert model.named_parameters() != []
    for name, param in model.named_parameters():
        assert param.grad is not None
        assert t.isfinite(param.grad).all().item()


def test_get_loss_shape(model: VerySimpleAttnOnlyTransformer, batch):
    loss = get_loss(batch, model)
    assert loss.shape == t.Size([])
    assert loss.item() != 0


def test_model_param_change():
    """Test that all model parameters change after training run"""
    model_cfg = SimplifiedAttnOnlyModelConfig()
    dataset_cfg = OVIncoherentDatasetConfig()
    train_cfg = OV_Incoherent_Training_Config(model_cfg, dataset_cfg)
    orig_model = create_model(train_cfg)
    train_cfg.epochs = 1
    initial_params = [param.clone() for param in orig_model.parameters()]
    new_model, _ = train_model(orig_model, training_cfg=train_cfg)
    for param, initial in zip(new_model.parameters(), initial_params):
        assert not t.equal(param, initial), "Model Parameter did not change in training"
        assert t.allclose(param, initial, atol=10), "Model Parameter changed too much"


def test_load_model():
    dataset_cfg = OVIncoherentDatasetConfig(
        B_toks=t.tensor([1, 2, 3]),
        device=DEVICE,
        n_batches=10,
    )
    model_cfg = SimplifiedAttnOnlyModelConfig(
        dataset_cfg=dataset_cfg, n_heads=2, d_head=1, device=DEVICE, output_bias=False
    )
    orig_model = VerySimpleAttnOnlyTransformer(model_cfg)
    PATH = "tests/test_models/toy_model_2H_3ST_MANUAL.st"
    load_model(model=orig_model, filename=PATH)

    assert isinstance(orig_model, VerySimpleAttnOnlyTransformer)


def test_epoch_class():
    score_example = 0.5
    metric = "accuracy"
    b_token_example = 2
    metric_example = BToken(metrics={metric: score_example})
    epoch = Epoch(b_token={b_token_example: metric_example})
    print(b_token_example)
    print(epoch)
    assert isinstance(epoch, Epoch)
    assert isinstance(metric_example, BToken)
