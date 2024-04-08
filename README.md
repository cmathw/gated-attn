# Gated Attention

This is a research code drop for reproducing the main results from the [Gated Attention](...) post, creating models that have OV-Incoherent attention head superposition and applying the gated attention mechanism to remove it.

![Attention Head Superposition](/assets/3_repr.png)

# Project Structure
```
gated-attention/
├── gated_attention/
│   ├── dataset/
│   │   ├── dataset_utils.py
│   │   └── toy_datasets.py
│   ├── experiments/
│   │   ├── analysis_utils.py
│   │   ├── create_gated_attention_model.py
│   │   └── create_skip_trigram_model.py
│   ├── modelling/
│   │   ├── modified_attention.py
│   │   ├── traditional_transformer.py
│   │   └── transformer_components/
│   │       ├── attention.py
│   │       ├── config.py
│   │       └── embed_unembed.py
│   └── train/
│       ├── train_gated_attention.py
│       └── train_traditional.py
├── tests/
│   ├── conftest.py
│   ├── test_analysis.py
│   ├── test_dataset_utils.py
│   ├── test_dataset.py
│   ├── test_end_to_end.py
│   ├── test_training.py
│   └── test_transformer_components.py
├── README.md
├── Makefile
├── poetry.lock
└── pyproject.toml
```
* Files for creating OV-Incoherent Skip Trigrams can be found in the `gated_attention/dataset/` directory.
* Files for creating the Gated Attention model can be found in the `gated_attention/modelling/` directory.
* Files for training the Gated Attention model can be found in the `gated_attention/train/` directory.
* Files for running experiments can be found in the `gated_attention/experiments/` directory.
* Models mentioned in the [Gated Attention](...) post can be found in the `gated_attention/experiments/saved_models` directory as safetensor files.

Typical workflow involves:

1. Training a traditional, simplified attention-only model to complete the OV-Incoherent task in `gated_attention/experiments/create_skip_trigram_model.py`.

2. Using this model, train a model that uses the gated attention mechanism to match the input-output behaviour of the original model but without attention head superposition. This can be done in `gated_attention/experiments/create_gated_attention_model.py`.

# Development Setup
* Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer).
  
* Install with dev dependencies:
  
```bash
poetry config virtualenvs.in-project true
poetry install --with dev
```

* Run tests (exluding slow tests):

```bash
make test
```

* Run all tests (including end-to-end and other slow tests, this will likely take a 5-10 minutes):

```bash
make test-all
```

* To *run* formatting on the codebase:

```bash
make format
```

* To *check* formatting on the codebase:

```bash
make check-format
```

* To clean up all temporary files and directories:

```bash
make clean
```

* To see makefile help:

```
make help
```
