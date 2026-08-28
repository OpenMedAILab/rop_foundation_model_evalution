# rop_foundation_model_evalution

**NeoROP-FM** — development-aware representation learning on infant retinal images for
longitudinal prediction of treatment-requiring retinopathy of prematurity (ROP).

This repository contains the experiment codebase for:

- benchmarking general vision and retinal foundation encoders on ROP classification
  (frozen features + linear probes under a leave-one-dataset-out protocol);
- continued self-supervised pretraining (iBOT / MAE) on public infant fundus images,
  with development-aware auxiliary heads (postmenstrual-age regression and
  visit-consistency contrastive learning);
- longitudinal modelling, label-efficiency and uncertainty-triage analyses, plus the
  aggregation / statistics and paper-figure tooling used throughout the project.

All quantitative results and statistical analyses are reported in the accompanying
manuscript.

## Background

Retinopathy of prematurity (ROP) is a proliferative retinal vascular disease of preterm
infants and a leading cause of preventable childhood blindness. Timely screening is
effective, but examination is resource-intensive and requires specialist expertise.

Most pretrained vision encoders are trained on natural images or adult retinal fundus
photos. Infant fundus images differ markedly — smaller anatomy, immature vasculature, and
strong developmental trends along postmenstrual age (PMA) — so both domain adaptation and
development-aware representations are open questions for ROP screening support.

This codebase implements the public-data experiment line ("Version A") of the NeoROP-FM
project:

- a unified leave-one-dataset-out (LODO) evaluation protocol over four public ROP
  datasets, with strict harmonized binary labels, patient-level bootstrap confidence
  intervals, and isolated (inner-CV) checkpoint selection;
- continued self-supervised pretraining of infant-domain encoders (iBOT on DINOv2-S as
  the main route, MAE on RETFound-Green as the fallback), optionally with
  development-aware auxiliary heads;
- downstream analyses: label efficiency, leave-one-dataset-out (LOTO) sensitivity,
  longitudinal prediction, uncertainty-based triage, and external-fold generalization.

## Repository layout

```
configs/            YAML configs; one per experiment stage (hyperparams + seeds)
experiments/        Shell orchestrators (E0–E5) + gpu_queue.sh GPU gate
scripts/            Ad-hoc analysis and reporting scripts
src/neoropfm/
  common.py         paths, seeding, GPU gate
  data/             manifest & split builders, QC, audits
  models/           backbone registry (8 encoders) + vendored RETFound model
  train/            feature extraction, LODO probe, SSL pretraining, PEFT,
                    label efficiency, longitudinal models
  eval/             metrics, aggregation (bootstrap CI / DeLong), uncertainty,
                    representation, subgroup analysis
  stats/            bootstrap / DeLong / Holm helpers
  ssl/              iBOT & MAE components, multi-crop augment, auxiliary heads
environment.yml     exact environment (anaconda3, torch 2.13.0+cu126)
pyproject.toml      setuptools package metadata
```

## Installation

```bash
conda env create -f environment.yml && conda activate neoropfm
pip install -e .
```

The code was developed with Python 3.14 / torch 2.13.0+cu126 on a single RTX 4090 (48 GB).
CPU-only execution works for probes, aggregation and statistics; feature extraction and
SSL pretraining need a GPU.

## Data preparation

No data is distributed with this repository. Download the public datasets yourself and
place them under `data/public_data/` (or point `NEOROPFM_PUBLIC_DATA_ROOT` anywhere):

| Directory | Dataset | License |
|---|---|---|
| `ridirp/` | Retinal Images of Infants (RIDIRP) | as per original release |
| `szeh_irops/` | SZEH IROPS | as per original release |
| `farfum_rop/` | FARFUM-RoP (figshare) | CC BY-NC-ND |
| `rop_vl/` | ROP-VL | as per original release |
| `hvdropdb/` | HVDROPDB (Mendeley Data v3, DOI 10.17632/xw5xc7xrmp.3) | CC BY 4.0 |
| `macretina/`, … | auxiliary SSL corpora (MACRetina, PLOS severity subset, vessel/disc segmentation raws) | as per original releases |

Environment variables:

- `NEOROPFM_PUBLIC_DATA_ROOT` — public dataset root (default `data/public_data/`).
- `RETFOUND_MAE_CKPT` — path to the RETFound-MAE Nature-2023 CFP checkpoint
  (default `outputs/weights/retfound_mae/`).
- `NEOROPFM_BENCHMARK_CACHE` — optional directory holding pre-extracted baseline feature
  caches for the `cache` feature source (default `outputs/benchmark_cache`).

## Quick start

Build the strict-label manifest and LODO splits (E0), then run the E1 benchmark:

```bash
python3 -m neoropfm.data.build_manifest_v2          # data/manifests/public_rop_manifest_v2.csv
python3 -m neoropfm.data.check_splits               # split-leak QC

MIN_FREE_MB=30000 experiments/gpu_queue.sh \
  python3 -m neoropfm.train.extract_features --config configs/extract.yaml --model dinov2_vitb14
python3 -m neoropfm.train.probe --config configs/probe_extract.yaml
python3 -m neoropfm.eval.aggregate --config configs/aggregate.yaml
```

Each experiment stage is a `configs/<name>.yaml` plus a `python3 -m neoropfm.*` module or
a `scripts/*.py`, orchestrated by the shell scripts in `experiments/`. Results are written
to `outputs/{experiment}/`.

## Experiments

| Stage | What | Entry point |
|---|---|---|
| E0 | manifest v2, split QC | `experiments/e0_manifest.sh` |
| E1 | frozen-probe baseline benchmark | `experiments/e1_benchmark.sh` |
| E2 | continued SSL pretraining (iBOT main, MAE fallback) | `experiments/e2_ssl.sh` |
| E2-LOTO | leave-one-dataset-out sensitivity retraining | `experiments/e2_loto.sh` |
| E3 | longitudinal POC (clinical vs image features, temporal transformer) | `experiments/e3_longitudinal.sh` |
| E4 | ablations (multi-view, label efficiency, PMA/DDS representation) | `scripts/e41c_multi_view.py`, `src/neoropfm/train/label_efficiency.py`, `src/neoropfm/eval/representation_analysis.py` |
| E5 | uncertainty triage (risk–coverage, DCA, workload simulation) | `scripts/e5_sensitivity_locked.py`, `src/neoropfm/eval/uncertainty_analysis.py` |

## Reproducibility protocol

- Labels follow a strict binary harmonization (negative = normal / Stage 1–2 without
  plus; positive = Stage ≥3 / plus / A-ROP; pre-plus, treated and non-ROP pathology
  excluded). Per-dataset mapping rules are documented in
  `src/neoropfm/data/` (`build_manifest_v2.py`, `audit_ridirp.py`) and the sensitivity
  analysis `src/neoropfm/data/label_rule_sensitivity.py`.
- Features are extracted once and cached under `outputs/features/{model}/`; probes and
  statistics never recompute them.
- Screening thresholds (sens95/sens98) are locked on the training fold only.
- Patient-level bootstrap CI (≥2000 iterations, seed 42, unit = `dataset|patient_id`;
  image-level fallback where patient grouping is unavailable).
- Checkpoint/model selection is isolated: inner-CV selection per run, then a one-shot
  evaluation of the held-out fold (`scripts/checkpoint_selection_isolated.py`).

## License

- Code: MIT (see `LICENSE`).
- Model weights are not redistributed: DINOv2 (LVD-142M) is CC BY-NC 4.0; RETFound-Green
  is released under a custom non-commercial research license (CNCRL) and its use must be
  declared in publications; RETFound-MAE and timm ImageNet weights follow their original
  licenses.
- Public dataset images/labels remain under the licenses of their original releases (see
  the table above). FARFUM-RoP is CC BY-NC-ND; do not redistribute derived label files
  beyond research use.

## Citation

Citation details will be added upon publication of the accompanying manuscript. For
questions, please open an issue in this repository.
