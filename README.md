# eta

Extreme Event Aware (η-) Learning

This repository contains notebook-based experiments for the paper *Extreme Event Aware (η-) Learning*. The main workflow studies supervised learning under extreme-event data scarcity: an ERM/MSE model is trained on scarce paired data, then an η model continues from the ERM weights with an additional Wasserstein regularizer on the scalar observable of extremeness.

## Repository Structure

- `src/models.py`: FCNN, SRCNN, diffusion components, U-Net, and Flow Matching wrappers.
- `src/train_utils.py`: MSE pretraining and η-learning routines for toy scalar/state experiments.
- `src/metric.py`: quantile/Wasserstein-style losses.
- `src/test_utils.py`: evaluation wrappers for scalar and state-map outputs.
- `src/utils.py`: devices, seeds, activations, result containers, and distribution helpers.
- `src/kde.py`, `src/plot_utils.py`: density estimation and plotting helpers.
- `src/revision_utils.py`: revision-only helpers for spatial metrics, prior sensitivity, and runtime measurement.
- `docs/equation_map.md`: map from manuscript objects/equation placeholders to implementation locations.
- `docs/data_and_models.md`: external data and pretrained-model layout, placeholders, and path setup.
- `scripts/check_external_assets.py`: checks the local data/checkpoint layout without downloading anything.
- `scripts/make_smoke_dataset.py`: builds the small ERA5-Land smoke-test subset from the full daily-max file.
- `examples/smoke_test_toy2d.py`: CPU-friendly synthetic smoke test for imports, MSE, and quantile-W1 wiring.

## Notebook-to-Experiment Map

- `notebooks/toy--2D->1D.ipynb`: 2D-to-1D toy scalar-map experiment.
- `notebooks/toy--2D->2D.ipynb`: 2D-to-2D toy state-map experiment.
- `notebooks/ERA5Land.ipynb`: vanilla ERA5-Land precipitation super-resolution.
- `notebooks/ERA5Land-DGM.ipynb`: Flow Matching model training and sampling.
- `notebooks/ERA5Land-DGM-Plot.ipynb`: Flow Matching PDF plots and η-corrected generated samples.
- `notebooks/ERA5Land-EVD.ipynb`: hypothesized GEVD/heavier-tail precipitation experiment.
- `notebooks/ERA5Land-Computational-Overhead.ipynb`: revision runtime and peak-VRAM measurements for precipitation experiments.

Revision diagnostics are displayed directly inside the corresponding notebooks. The added sections do not save new figures or tables by default.

## Environment Setup

The pinned setup targets the tested `eta` conda environment used for the notebook experiments: Python 3.11.8, PyTorch 2.2.0+cu118, CUDA 11.8, and the package versions listed in `requirements.txt`.

Use an existing `eta` environment, or create an equivalent one and install the pinned Python dependencies:

```sh
conda activate eta
python -m pip install -r requirements.txt
```

For notebook use, register a Jupyter kernel:

```sh
python -m ipykernel install --user --name eta --display-name "eta"
```

PyTorch/CUDA wheels depend on the user’s CUDA driver and target accelerator. `requirements.txt` includes the PyTorch CUDA 11.8 wheel index because the remote `eta` environment uses `torch==2.2.0+cu118`.

Minimal import check:

```sh
python - <<'PY'
import torch, numpy, scipy
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
PY
```

The pins document the tested runtime stack; they do not guarantee bitwise reproducibility across GPUs, CUDA versions, cuDNN versions, or BLAS backends.

### `requirements.txt` and `environment.yml`

`requirements.txt` is the curated pip-facing dependency list for the Python packages used by the repository. `environment.yml` is the fuller conda environment export and is the stronger record for reproducing the remote `eta` environment, especially for compiled packages, conda-managed libraries, and CUDA-related packages.

These files can disagree if they are generated from different views of the same environment or at different times. For example, `pip freeze` and Python package metadata report pip-installed distributions, while `conda env export` reports conda packages plus a `pip:` subsection. If a package exists in both conda and pip records, or if one file is regenerated after a package upgrade, versions can diverge.

When refreshing dependency documentation, regenerate both files back-to-back from the activated remote environment:

```sh
conda activate eta
conda env export --no-builds | sed '/^prefix:/d' > environment.yml
python -m pip freeze > requirements.full-freeze.txt
```

Then update the curated `requirements.txt` from the same environment. If the curated file and `environment.yml` disagree, treat `environment.yml` as the authoritative remote-environment record and make `requirements.txt` consistent with it or clearly document why it intentionally differs.

## Command-Line Experiments

Notebook workflows are also available as Python entry points under `experiments/`. They preserve the notebook defaults, read data and artifacts from this checkout by default, and do not delete notebooks or existing artifacts.

Examples:

```sh
cd /path/to/eta

python -m experiments.toy_2d_to_1d
python -m experiments.toy_2d_to_2d
python -m experiments.era5land
python -m experiments.era5land_evd --misspec-alpha -1 -0.5 0.5 1.5 2
python -m experiments.era5land_dgm --field lores --train --sample
python -m experiments.era5land_dgm_plot
python -m experiments.era5land_computational_overhead
```

Use `--help` on any entry point for runtime controls such as shortened probe epochs, checkpoint paths, sample output paths, and explicit `--train-*` flags. See `docs/command_line_experiments.md` for the full command-line runbook.

## Lightweight Checks

Syntax-check Python modules:

```sh
python -m py_compile src/*.py experiments/*.py scripts/check_external_assets.py examples/smoke_test_toy2d.py
```

Run the synthetic smoke test without ERA5-Land data or pretrained models:

```sh
python examples/smoke_test_toy2d.py
```

Check whether the local data/checkpoint layout is present:

```sh
python scripts/check_external_assets.py
```

The full precipitation and generative-model experiments require external datasets and pretrained checkpoints. The minimal smoke tests are intended only to verify installation and pipeline wiring.

## Code, Data, and Model Availability

Source code is maintained in this GitHub repository. Data, pretrained models, and generated samples are deposited on Zenodo:

- Datasets (ERA5-Land, toy, smoke-test subset): [`10.5281/zenodo.21635446`](https://doi.org/10.5281/zenodo.21635446)
- Pretrained models and generated samples: [`10.5281/zenodo.21635468`](https://doi.org/10.5281/zenodo.21635468)
- Code archive, all versions: [`10.5281/zenodo.21636106`](https://doi.org/10.5281/zenodo.21636106)
- Code archive, `v1.0.0`: [`10.5281/zenodo.21636107`](https://doi.org/10.5281/zenodo.21636107)

Checksums are published as `SHA256SUMS-data.txt` in the dataset record and `SHA256SUMS-models-samples.txt` inside `eta-models-samples-v1.0.0.zip` in the model record. Verify a download with `shasum -a 256 -c`.

Both Zenodo records are released under CC-BY-4.0. The ERA5-Land files are derived products of ERA5-Land (Copernicus Climate Change Service / ECMWF, obtained from the Copernicus Climate Data Store) and remain subject to the Licence to Use Copernicus Products.

Large data files, generated samples, and pretrained checkpoints are not redistributed in this GitHub repository. Download them from the records above and stage them under the repository checkout as shown below.

The default layout keeps data and artifacts under the repository checkout:

```text
/path/to/eta/
  data/
  models/
  samples/
  result_log/
```

See `docs/data_and_models.md`, `data/README.md`, `checkpoints/README.md`, and `.env.example` for the expected directory layout and placeholders.

## Data and Checkpoint Requirements

The notebooks and scripts default to a single repository-root layout:

```text
CODE_DIR=/path/to/eta
WORK_DIR=/path/to/eta
```

The repository root is expected to contain assets equivalent to:

```text
data/toy.pth
data/era5land_USA_SouthEast_1999-2023_dailymax.nc
models/precip-srcnn/
models/fm/
samples/
```

The external asset layout is documented in `docs/data_and_models.md`. Unpacking the data and model archives at the repository root reproduces this layout directly.

## Full Experiment Run Order

Run Jupyter from the code checkout expected by the notebooks:

```sh
cd /path/to/eta
python -m py_compile src/*.py experiments/*.py scripts/check_external_assets.py examples/smoke_test_toy2d.py
jupyter lab
```

Notebook order for paper experiments:

1. `notebooks/toy--2D->1D.ipynb`
2. `notebooks/toy--2D->2D.ipynb`
3. `notebooks/ERA5Land.ipynb`
4. `notebooks/ERA5Land-DGM.ipynb`
5. `notebooks/ERA5Land-DGM-Plot.ipynb`
6. `notebooks/ERA5Land-EVD.ipynb`
7. `notebooks/ERA5Land-Computational-Overhead.ipynb`

The full ERA5-Land pipeline requires the complete 1999-2023 ERA5-Land processing workflow and externally hosted artifacts. Do not expect a fresh GitHub checkout to contain those assets.

## Revision Diagnostics

Notebook-visible revision sections include:

- Toy spatial uncertainty across η-estimator realizations in both toy notebooks.
- Vanilla ERA5-Land full-field RMSE, eta-vs-MSE relative RMSE increase, and mean/std SSIM metrics in `notebooks/ERA5Land.ipynb`.
- Prior-misspecification sensitivity for perturbed GEVD/reference quantiles in `notebooks/ERA5Land-EVD.ipynb`.
- Wall-clock runtime and peak-VRAM measurements in `notebooks/ERA5Land-Computational-Overhead.ipynb`.

Most revision diagnostics are displayed directly in notebooks and are not saved as standalone files. No new cross-notebook result tables are saved by default.

## Hardware and Expected Runtimes

Measured values below come from `result_log/computational_overhead_revision_results_2026-06-22.md`, which summarizes `notebooks/ERA5Land-Computational-Overhead.ipynb`. The measurements used bounded probes and proportional full-run estimates, so they should be treated as approximate.

| Component | Tested hardware | Approx. GPU VRAM | Approx. runtime | Notes |
|---|---:|---:|---:|---|
| Minimal smoke test | CPU acceptable | not measured | seconds | Synthetic arrays only; verifies imports, MSE, and quantile-W1 wiring. |
| Toy 2D-to-1D MSE baseline | Tesla V100-SXM2-32GB-LS | 0.027 GB allocated | 25.9 min | 3000 MSE iterations from timing probe. |
| Toy 2D-to-1D eta continuation | Tesla V100-SXM2-32GB-LS | 0.523 GB allocated | 3.6 min | Uses large auxiliary grid for quantile matching. |
| Toy 2D-to-2D MSE baseline | Tesla V100-SXM2-32GB-LS | 0.027 GB allocated | 0.7 min | 3000 MSE iterations from timing probe. |
| Toy 2D-to-2D eta continuation | Tesla V100-SXM2-32GB-LS | 0.539 GB allocated | 3.0 min | State-map eta continuation. |
| ERA5-Land MSE training | Tesla V100-SXM2-32GB-LS | 0.341 GB allocated | 54.8 min | SRCNN baseline; 500 epochs. |
| ERA5-Land eta continuation | Tesla V100-SXM2-32GB-LS | 0.899 GB allocated | 10.4 min | SRCNN eta continuation with IICT and 102 W1 fields. |
| GEVD eta run | Tesla V100-SXM2-32GB-LS | 1.762 GB allocated | 9.6 min | Uses 350 GEVD reference quantiles with `omega=1`. |
| Flow Matching HR training | Tesla V100-SXM2-32GB-LS | 0.308 GB allocated | 14.7 min | Probe-based estimate for HR FM training component. |
| Flow Matching LR training | Tesla V100-SXM2-32GB-LS | 0.258 GB allocated | 32.4 min | Probe-based estimate for LR FM training component. |
| Flow Matching HR sampling | Tesla V100-SXM2-32GB-LS | 0.280 GB allocated | 10.2 h | 9044 generated samples and 200 ODE steps. |
| Flow Matching LR sampling | Tesla V100-SXM2-32GB-LS | 0.261 GB allocated | 19.3 min | 9044 generated samples and 100 ODE steps. |
| LR eta pass-through | Tesla V100-SXM2-32GB-LS | 0.278 GB allocated | 4.5 s | Pass generated LR samples through trained SRCNN eta map. |

The tested system had 31.73 GB GPU VRAM and 503.76 GB system RAM. Full precipitation and generative-model runs require external datasets and checkpoint archives; the smoke tests do not.

## Non-Destructive Revision Policy

The revision work appends notebook sections and adds helper/documentation files while leaving existing experiment code, outputs, saved models, saved samples, and saved figures untouched. Existing duplicate or exploratory notebook cells are intentionally preserved for backward reproducibility.
