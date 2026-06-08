# eta

Extreme Event Aware (η-) Learning

This repository contains notebook-based experiments for the paper *Extreme Event Aware (η-) Learning*. The main workflow studies supervised learning under extreme-event data scarcity: an ERM/MSE model is trained on scarce paired data, then an η model continues from the ERM weights with an additional one-dimensional quantile-Wasserstein regularizer on the scalar observable of extremeness.

## Repository Structure

- `models.py`: FCNN, SRCNN, diffusion components, U-Net, and Flow Matching wrappers.
- `train_utils.py`: MSE pretraining and η-learning routines for toy scalar/state experiments.
- `metric.py`: quantile/Wasserstein-style losses.
- `test_utils.py`: evaluation wrappers for scalar and state-map outputs.
- `utils.py`: devices, seeds, activations, result containers, and distribution helpers.
- `kde.py`, `plot_utils.py`: density estimation and plotting helpers.
- `revision_utils.py`: revision-only helpers for spatial metrics, prior sensitivity, and runtime measurement.
- `docs/equation_map.md`: map from paper equations/diagnostics to implementation locations.

## Notebook-to-Experiment Map

- `toy--2D->1D.ipynb`: 2D-to-1D toy scalar-map experiment.
- `toy--2D->2D.ipynb`: 2D-to-2D toy state-map experiment.
- `ERA5Land.ipynb`: vanilla ERA5-Land precipitation super-resolution.
- `ERA5Land-DGM.ipynb`: Flow Matching model training and sampling.
- `ERA5Land-DGM-Plot.ipynb`: Flow Matching PDF plots and η-corrected generated samples.
- `ERA5Land-EVD.ipynb`: hypothesized GEVD/heavier-tail precipitation experiment.
- `ERA5Land-Computational-Overhead.ipynb`: revision runtime and peak-VRAM measurements for precipitation experiments.

Revision diagnostics are displayed directly inside the corresponding notebooks. The added sections do not save new figures or tables by default.

## Setup

Create an environment with Python 3.9-compatible packages, then install dependencies:

```sh
python -m pip install -r requirements.txt
```

Some ERA5/server-side dependencies were not installed in the local Codex environment when `requirements.txt` was created, so their exact versions are marked as unavailable there. Before final archival reruns, pin those package versions from the remote experiment server if possible.

## Data Requirements

Full paper experiments require the processed ERA5-Land daily-maximum precipitation file and trained/generated artifacts on the remote server-style checkout:

```text
data/ERA5/era5land_USA_SouthEast_1999-2023_dailymax.nc
data/toy/var10/N100.pth
models/precip-srcnn/
models/fm/
samples/
```

Large data, trained models, and generated samples are not expected to be present in every local checkout.

## Included Data

The repository is expected to include the data needed for the experiments. The included data is intended to support reproducible reruns of the notebook workflows; full paper-scale ERA5-Land experiments still require the complete 1999-2023 ERA5-Land processing pipeline and the server-style artifact layout listed above.

## Full Experiment Run Order

Run from the server checkout expected by the notebooks:

```sh
cd /home/research/jenzheng/documents/kai/research/eta/eta
python -m py_compile *.py
jupyter lab
```

Notebook order for paper experiments:

1. `toy--2D->1D.ipynb`
2. `toy--2D->2D.ipynb`
3. `ERA5Land.ipynb`
4. `ERA5Land-DGM.ipynb`
5. `ERA5Land-DGM-Plot.ipynb`
6. `ERA5Land-EVD.ipynb`
7. `ERA5Land-Computational-Overhead.ipynb`

The full ERA5-Land pipeline requires the complete 1999-2023 ERA5-Land processing workflow and the server-side artifact layout above.

## Revision Diagnostics

New notebook-visible revision sections include:

- Toy spatial uncertainty across η-estimator realizations in both toy notebooks.
- Vanilla ERA5-Land full-field RMSE, rRMSE, SSIM, and spatial-correlation metrics in `ERA5Land.ipynb`.
- Prior-misspecification sensitivity for perturbed GEVD/reference quantiles in `ERA5Land-EVD.ipynb`.
- Wall-clock runtime and peak-VRAM measurements in `ERA5Land-Computational-Overhead.ipynb`.
- Documentation notes appended to all touched notebooks.

Most revision diagnostics are displayed directly in notebooks and are not saved as standalone files. No new cross-notebook result tables are saved by default.

## Hardware and Runtime Notes

The precipitation notebooks are GPU-oriented. The original runbook references NVIDIA V100-style server execution for toy and ERA5 runs. Full ERA5-Land and Flow Matching reruns can take hours depending on GPU count, memory, and I/O.

Approximate runtimes should be filled from `ERA5Land-Computational-Overhead.ipynb` after it has been run on the target hardware. That notebook reports:

- vanilla ERM training time,
- vanilla η training time with IICT,
- Flow Matching HR/LR training time,
- Flow Matching HR/LR sampling time,
- η-map pass-through time for generated LR samples,
- hypothesized-GEVD η training time,
- peak allocated/reserved GPU VRAM.

## Development Checks

Syntax-check root Python modules:

```sh
python -m py_compile *.py
```

Smoke-test imports when the full environment is available:

```sh
python -c "import models, train_utils, test_utils, revision_utils"
```

## Non-Destructive Revision Policy

The revision work appends notebook sections and adds helper/documentation files while leaving existing experiment code, outputs, saved models, saved samples, and saved figures untouched. Existing duplicate or exploratory notebook cells are intentionally preserved for backward reproducibility.
