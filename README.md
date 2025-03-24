# CTRPlayground: Train CTR models for fun

1. Data
    - Data **preprocessing** is important: dense features -> sparse features, threshold for infrequent feature.

2. Model
    - Scaling (both model size and training step) is not significant.

3. Optimization
    - AdamW and LR scheduler are useful.

## Some Thoughts

- The projection head of LLMs is the retrieval embedding matrix for all tokens. Hence, next token prediction could be considered as vector retrieval to some extent

# Datasets

- **Criteo**: `HF_ENDPOINT=https://hf-mirror.com python criteo.py` to download from hf.

# Quick Start

`python run.py --model_config_path model/DeepFM/config.yaml`

# Reference

- [FuxiCTR](https://github.com/reczoo/FuxiCTR)
- [CTR Datasets](https://github.com/reczoo/Datasets)
