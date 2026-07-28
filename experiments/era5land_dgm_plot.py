"""Command-line version of ERA5Land-DGM-Plot.ipynb."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.common import (
    add_common_path_args,
    build_srcnn,
    load_era5_data,
    load_state_flexible,
    precip_eta_path,
    save_npz,
    save_table_csv,
    setup_cli,
    summarize_maxima,
)
from revision_utils import predict_field_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_path_args(parser)
    parser.add_argument("--era5-data", type=Path, default=None, help="ERA5-Land file. Defaults to WORK_DIR/data/era5land_USA_SouthEast_1999-2023_dailymax.nc.")
    parser.add_argument("--ds-fact", type=int, default=10)
    parser.add_argument("--num-years", type=float, default=0.5)
    parser.add_argument("--w1-tail-thresh", type=float, default=150.0)
    parser.add_argument("--omega", type=int, default=30)
    parser.add_argument("--lores-samples", type=Path, default=None)
    parser.add_argument("--hires-samples-0p5", type=Path, default=None)
    parser.add_argument("--hires-samples-2p5", type=Path, default=None)
    parser.add_argument("--hires-samples-10", type=Path, default=None)
    parser.add_argument("--hires-samples-25", type=Path, default=None)
    parser.add_argument("--eta-checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-parallel", choices=["always", "auto", "never"], default="always")
    parser.add_argument("--save-eta-samples", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--save-maxima", type=Path, default=None)
    return parser.parse_args()


def default_sample_paths(work_dir: Path) -> dict[str, Path]:
    return {
        "lores": work_dir / "samples/9044samples-lores-fm-100step-unet16-25yr-80epoch.npy",
        "hires_0p5": work_dir / "samples/9044samples-hires-fm-200step-unet16-0.5yr-200epoch.npy",
        "hires_2p5": work_dir / "samples/9044samples-hires-fm-200step-unet16-2.5yr-100epoch.npy",
        "hires_10": work_dir / "samples/9044samples-hires-fm-200step-unet16-10yr-100epoch.npy",
        "hires_25": work_dir / "samples/9044samples-hires-fm-200step-unet16-25yr-50epoch.npy",
    }


def load_required(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} samples: {path}")
    return np.load(path)


def main() -> int:
    args = parse_args()
    ctx = setup_cli(args)
    paths = default_sample_paths(ctx.work_dir)
    lores_path = args.lores_samples or paths["lores"]
    hires_0p5_path = args.hires_samples_0p5 or paths["hires_0p5"]
    hires_2p5_path = args.hires_samples_2p5 or paths["hires_2p5"]
    hires_10_path = args.hires_samples_10 or paths["hires_10"]
    hires_25_path = args.hires_samples_25 or paths["hires_25"]

    era5 = load_era5_data(ctx.work_dir, ds_fact=args.ds_fact, era5_data=args.era5_data)
    lores_samples = load_required(lores_path, "lores")
    hires_samples_0p5 = load_required(hires_0p5_path, "hires 0.5-year")
    optional_samples = {
        "HR generated 2.5yr": np.load(hires_2p5_path) if hires_2p5_path.exists() else None,
        "HR generated 10yr": np.load(hires_10_path) if hires_10_path.exists() else None,
        "HR generated 25yr": np.load(hires_25_path) if hires_25_path.exists() else None,
    }

    eta_checkpoint = args.eta_checkpoint or precip_eta_path(
        ctx.work_dir, args.num_years, args.ds_fact, int(args.w1_tail_thresh), args.omega
    )
    if not eta_checkpoint.exists():
        raise FileNotFoundError(f"Missing eta downscaler checkpoint: {eta_checkpoint}")
    model_eta = build_srcnn(args.ds_fact, args.data_parallel)
    model_eta = load_state_flexible(model_eta, eta_checkpoint, ctx.device).to(ctx.device)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(lores_samples, dtype=torch.float32).unsqueeze(1),
            torch.tensor(hires_samples_0p5, dtype=torch.float32).unsqueeze(1),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    eta_from_lores = predict_field_array(model_eta, loader, device=ctx.device)
    if args.save_eta_samples is not None:
        args.save_eta_samples.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_eta_samples, eta_from_lores)
        print(f"Saved eta-downscaled samples: {args.save_eta_samples}")

    summaries = [
        summarize_maxima("HR truth", era5.tp_trim_numpy),
        summarize_maxima("LR truth", era5.tp_trim_ds_numpy),
        summarize_maxima("LR generated 25yr", lores_samples),
        summarize_maxima("HR generated 0.5yr", hires_samples_0p5),
        summarize_maxima("eta(LR generated)", eta_from_lores),
    ]
    for name, samples in optional_samples.items():
        if samples is not None:
            summaries.append(summarize_maxima(name, samples))
    summary_df = pd.DataFrame(summaries)
    print(summary_df.to_string(index=False))
    save_table_csv(summary_df, args.summary_csv)
    save_npz(
        args.save_maxima,
        truth_max=np.max(era5.tp_trim_numpy, axis=(1, 2)),
        lores_truth_max=np.max(era5.tp_trim_ds_numpy, axis=(1, 2)),
        lores_sample_max=np.max(lores_samples, axis=(1, 2)),
        hires_sample_0p5_max=np.max(hires_samples_0p5, axis=(1, 2)),
        eta_lores_sample_max=np.max(eta_from_lores, axis=(1, 2)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
