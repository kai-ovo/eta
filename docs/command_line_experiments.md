# Command-Line Experiment Runbook

The original notebooks remain the paper-facing interactive records. The Python entry points in `experiments/` repack the same workflows for command-line execution, checkpoint loading, shorter smoke runs, and batch scheduling.

Run from the repository root:

```sh
cd /path/to/eta
conda activate eta
export ETA_CODE_DIR=/path/to/eta
```

All scripts accept:

```sh
--code-dir /path/to/eta        # helper-code checkout, defaults to this repo or ETA_CODE_DIR
--work-dir /path/to/eta        # data/checkpoint/sample root, defaults to ETA_WORK_DIR or this repo
--device cuda                  # optional, defaults to CUDA when available
```

The default layout is a single repository root:

```text
CODE_DIR=/path/to/eta
WORK_DIR=/path/to/eta
```

Ordinary runs should leave `ETA_WORK_DIR` unset. Use `ETA_WORK_DIR` or `--work-dir` only for an advanced local storage override.

## Toy Scalar: 2D To 1D

Notebook: `notebooks/toy--2D->1D.ipynb`

```sh
python -m experiments.toy_2d_to_1d
```

Useful controls:

```sh
python -m experiments.toy_2d_to_1d \
  --mse-epochs 3000 \
  --pretrain-epochs 1000 \
  --eta-epochs 3000 \
  --eta-seeds 25 \
  --lambda-sweep \
  --save-npz result_log/toy_2d_to_1d_cli_outputs.npz
```

This trains the FCNN MSE baseline, initializes eta from the Gaussian-field pretrain, continues with the scalar quantile-W1 objective, and optionally runs the notebook lambda sweep.

## Toy State: 2D To 2D

Notebook: `notebooks/toy--2D->2D.ipynb`

```sh
python -m experiments.toy_2d_to_2d
```

The script constructs `u=(u_1,u_2)` from the scalar toy map and the Fourier mode, uses `g(u)=2*abs(u_1)+0.5*abs(u_2)`, and trains the architecture-matched MSE and eta models.

## ERA5-Land Downscaling

Notebook: `notebooks/ERA5Land.ipynb`

Evaluate cached checkpoints:

```sh
python -m experiments.era5land
```

Train missing checkpoints explicitly:

```sh
python -m experiments.era5land --train-mse
python -m experiments.era5land --train-eta
```

The script uses the same defaults as the notebook: `num_years=0.5`, `ds_fact=10`, `SRCNN(hidden_dim=64, num_blocks=3)`, `lambda=1`, tail threshold `150`, `omega=30`, and `150` eta epochs.

## GEVD Prior And Misspecified Tails

Notebook: `notebooks/ERA5Land-EVD.ipynb`

```sh
python -m experiments.era5land_evd
```

Run the revision misspecification alphas:

```sh
python -m experiments.era5land_evd \
  --misspec-alpha -1 -0.5 0.5 1.5 2 \
  --train-missing-misspec \
  --metrics-csv result_log/era5land_evd_cli_metrics.csv
```

Misspecified-tail checkpoints are written under `WORK_DIR/models/precip-srcnn/misspec/` with the same alpha filename tags used by the notebook, such as `alpha_m0p5` and `alpha_0p5`.

## Flow Matching Train And Sample

Notebook: `notebooks/ERA5Land-DGM.ipynb`

Train and sample a low-resolution FM model:

```sh
python -m experiments.era5land_dgm \
  --field lores \
  --num-years 25 \
  --epochs 80 \
  --nsteps 100 \
  --train \
  --sample
```

Train or sample high-resolution FM models by switching `--field hires` and setting the notebook schedule, for example `--num-years 0.5 --epochs 200 --nsteps 200`.

## Flow Matching Eta Pass-Through

Notebook: `notebooks/ERA5Land-DGM-Plot.ipynb`

```sh
python -m experiments.era5land_dgm_plot \
  --summary-csv result_log/era5land_dgm_cli_summary.csv \
  --save-maxima result_log/era5land_dgm_cli_maxima.npz
```

The script loads generated LR/HR sample arrays, loads the eta downscaler, passes LR generated samples through eta, and reports max-observable summaries for truth, LR samples, HR-FM samples, and eta-corrected samples.

## Computational Overhead

Notebook: `notebooks/ERA5Land-Computational-Overhead.ipynb`

```sh
python -m experiments.era5land_computational_overhead \
  --output-csv result_log/computational_overhead_cli.csv
```

This script is precipitation-only. It measures bounded probes for SRCNN MSE training, vanilla eta continuation, Flow Matching training, Flow Matching sampling, eta pass-through, and GEVD eta continuation, then scales by the paper settings documented in the output table.
