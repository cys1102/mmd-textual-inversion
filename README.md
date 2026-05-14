# Inversion and Distribution Matching for Few-Shot Learning

This repository contains the experiment code for **Bridging the Gap: Synthetic Data Augmentation through Inversion and Distribution Matching for Few-shot Learning**. The main experiment trains textual-inversion pseudo-token embeddings with an additional distribution-matching loss, controlled by `--dist_match`.

Paper: [https://dl.acm.org/doi/pdf/10.1145/3703412.3703440](https://dl.acm.org/doi/pdf/10.1145/3703412.3703440)

## Code Layout

- `ti_mmd.py`: trains per-class textual-inversion embeddings with the diffusion denoising loss plus the `L_MMD` / distribution-matching term.
- `aggregate_embeddings.py`: merges class-level `learned_embeds.bin` files into `{dataset}-tokens-mmd{dist_match}/...pt`.
- `train_classifier.py`: generates synthetic images with the learned tokens and trains/evaluates the few-shot classifier.
- `classifier.py`: ResNet-50 or DeiT classifier wrapper.
- `semantic_aug/`: dataset adapters and generative augmentation modules.
- `diffusers_/`: local Diffusers fork used by the augmentation pipeline.
- `scripts/`: runnable examples for the main workflow.

## Setup

Create a Python environment with CUDA-enabled PyTorch, then install the remaining dependencies:

```bash
pip install -r requirements.txt
accelerate config
```

Set dataset roots through environment variables before running experiments:

```bash
export PETS_DIR=/path/to/oxford-pets
export DTD_ROOT=/path/to/dtd
export FLOWERS102_DIR=/path/to/flowers102
```

Other supported variables include `COCO_DIR`, `PASCAL_DIR`, `AIRCRAFT_ROOT`, `CUB_ROOT`, `CALTECH101_DIR`, `IMAGENET_DIR`, and `SPURGE_DIR`.

## Main Experiment Workflow

1. Train MMD-regularized textual-inversion embeddings:

```bash
DATASET=pets DIST_MATCH=0.005 EXAMPLES_PER_CLASS="4 8 16" \
  bash scripts/run_mmd_textual_inversion.sh
```

2. Aggregate learned class embeddings:

```bash
DATASET=pets DIST_MATCH=0.005 EXAMPLES_PER_CLASS="4 8 16" \
  bash scripts/aggregate_mmd_embeddings.sh
```

3. Generate synthetic images and train the classifier:

```bash
DATASET=pets DIST_MATCH=0.005 EXAMPLES_PER_CLASS="8" \
  bash scripts/train_classifier_mmd_ti.sh
```

Outputs are written to `fine-tuned-mmd*/`, `*-tokens-mmd*/`, `aug/`, and `*-baselines/`.

## Core Loss

`ti_mmd.py` freezes the VAE and UNet, updates only the added CLIP token embedding, and restores all non-placeholder embeddings after each optimizer step. The objective is:

```text
loss = diffusion_mse + dist_match * distribution_matching_loss
```

The distribution term is implemented near the training loop where `model_pred_ws` and `target_ws` are compared.

## Citation

If you use this code, cite the paper:

```bibtex
@inproceedings{chung2024bridging,
  title={Bridging the Gap: Synthetic Data Augmentation through Inversion and Distribution Matching for Few-shot Learning},
  author={Chung, Yunsung and Wang, Janet and Hamm, Jihun},
  booktitle={Proceedings of the 4th International Conference on AI-ML Systems},
  pages={1--5},
  year={2024}
}
```
