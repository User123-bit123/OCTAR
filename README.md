# OCTAR

Official implementation of **OCTAR: Occlusion-aware CLIP-guided Token Anchor Refinement for Person Re-Identification**.

OCTAR uses frozen CLIP text semantics and detached CLIP-aligned visual tokens during training to construct realistic clean/occluded pairs and selectively refine identity representations. The inference network is unchanged from the CLIP-ReID image encoder.

## Method

The released configuration contains the final method used in the paper:

| Module | Purpose | Main implementation |
| --- | --- | --- |
| SOC | CLIP-guided source occluder and target body-region selection | `compose_soc_view`, `INPUT.SOC` |
| RTG | Reliable clean-teacher selection | `_clip_token_reliability`, `_rtg_image_scores`, `_rtg_teacher_weights` |
| R-MTD | Region-aware masked token distillation | `_region_balanced_candidate_weights`, `_rmtd_loss` |
| PTA | Purified token anchor construction | `_pta_anchor` |
| APR | Momentum identity-prototype update and regularization | `_apr_update_prototype_bank`, `_apr_loss` |

The method implementation is in `processor/processor_octar.py`. A paper-to-code description is available in [docs/METHOD.md](docs/METHOD.md).

## Installation

```bash
conda create -n octar python=3.8 -y
conda activate octar
pip install torch torchvision
pip install -r requirements.txt
```

The experiments in the paper were run on one NVIDIA GeForce RTX 3090 GPU with 24 GB memory.

## Datasets

Set `DATASETS.ROOT_DIR` to a directory containing the required datasets:

```text
data/
|-- Market-1501-v15.09.15/
|-- dukemtmcreid/
|-- Occluded_Duke/
`-- Occluded_REID/
```

Occluded-Duke can also be placed under `dukemtmcreid/Occluded_Duke/`. The Occluded-ReID loader accepts `Occluded_REID`, `OccludedREID`, or `Occluded-ReID`.

## Training

Train the final OCTAR configuration on Occluded-Duke:

```bash
CUDA_VISIBLE_DEVICES=0 python train_octar.py \
  --config_file configs/person/octar_occluded_duke.yml \
  DATASETS.ROOT_DIR /path/to/data
```

Market-1501 and DukeMTMC-reID use the corresponding configurations:

```bash
CUDA_VISIBLE_DEVICES=0 python train_octar.py \
  --config_file configs/person/octar_market1501.yml \
  DATASETS.ROOT_DIR /path/to/data

CUDA_VISIBLE_DEVICES=0 python train_octar.py \
  --config_file configs/person/octar_dukemtmc.yml \
  DATASETS.ROOT_DIR /path/to/data
```

The SIE+OLP variant uses stride `[12, 12]` and camera SIE. A ready-to-run Occluded-Duke configuration is included:

```bash
CUDA_VISIBLE_DEVICES=0 python train_octar.py \
  --config_file configs/person/octar_occluded_duke_sie_olp.yml \
  DATASETS.ROOT_DIR /path/to/data
```

For Market-1501 or DukeMTMC-reID, add the same options to the corresponding basic configuration:

```bash
MODEL.STRIDE_SIZE "[12, 12]" MODEL.SIE_CAMERA True MODEL.SIE_COE 1.0
```

To reuse a Stage 1 prompt checkpoint:

```bash
SOLVER.STAGE1.SKIP True SOLVER.STAGE1.WEIGHT /path/to/ViT-B-16_stage1_120.pth
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python test_octar.py \
  --config_file configs/person/octar_occluded_duke.yml \
  DATASETS.ROOT_DIR /path/to/data \
  TEST.WEIGHT /path/to/ViT-B-16_best.pth
```

For Occluded-ReID, evaluate a model trained on Market-1501:

```bash
CUDA_VISIBLE_DEVICES=0 python test_octar.py \
  --config_file configs/person/octar_occluded_reid_eval.yml \
  DATASETS.ROOT_DIR /path/to/data \
  TEST.WEIGHT /path/to/market1501_weight.pth
```

When evaluating the Market-1501 OLP model on Occluded-ReID, retain `MODEL.STRIDE_SIZE "[12, 12]"` and disable camera SIE with `MODEL.SIE_CAMERA False` because camera identities do not correspond across the two datasets.

## Reported Results

No re-ranking is used.

| Dataset | Configuration | mAP | Rank-1 |
| --- | --- | ---: | ---: |
| Occluded-Duke | OCTAR | 63.6 | 72.0 |
| Occluded-Duke | OCTAR + SIE + OLP | 64.0 | 71.2 |
| Market-1501 | OCTAR | 90.6 | 95.5 |
| Market-1501 | OCTAR + SIE + OLP | 91.5 | 96.0 |
| DukeMTMC-reID | OCTAR | 83.6 | 91.3 |
| DukeMTMC-reID | OCTAR + SIE + OLP | 84.3 | 91.7 |
| Occluded-ReID | OCTAR | 85.0 | 87.5 |
| Occluded-ReID | OCTAR + OLP | 85.7 | 88.4 |

## Release Scope

This repository contains source code and final experiment configurations only. Model weights, datasets, training logs, generated figures, and ablation-only launchers are intentionally excluded.

