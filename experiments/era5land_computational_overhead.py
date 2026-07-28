"""Command-line version of ERA5Land-Computational-Overhead.ipynb.

This script keeps the overhead measurement precipitation-only: vanilla
downscaling, Flow Matching training/sampling/eta pass-through, and GEVD eta.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.common import (
    add_common_path_args,
    build_srcnn,
    build_unet,
    gevd_reference,
    get_dgm_train,
    load_era5_data,
    load_state_flexible,
    make_precip_tensors,
    precip_mse_path,
    precip_tail_objects,
    save_table_csv,
    setup_cli,
    stopwatch,
)
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from revision_utils import format_seconds, predict_field_array, train_precip_eta_no_save, train_srcnn_mse_no_save
from utils import global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_path_args(parser)
    parser.add_argument("--era5-data", type=Path, default=None, help="ERA5-Land file. Defaults to WORK_DIR/data/era5land_USA_SouthEast_1999-2023_dailymax.nc.")
    parser.add_argument("--ds-fact", type=int, default=10)
    parser.add_argument("--num-years", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-aux-size", type=int, default=768)
    parser.add_argument("--mse-probe-epochs", type=int, default=1)
    parser.add_argument("--eta-probe-epochs", type=int, default=1)
    parser.add_argument("--fm-probe-batches", type=int, default=2)
    parser.add_argument("--fm-sample-steps", type=int, default=4)
    parser.add_argument("--fm-sample-count", type=int, default=4)
    parser.add_argument("--fm-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--data-parallel", choices=["always", "auto", "never"], default="auto")
    return parser.parse_args()


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def add_row(rows, experiment, component, seconds, peak, scale_factor, paper_setting, timed_probe, notes=""):
    estimated = seconds * scale_factor if np.isfinite(seconds) else np.nan
    rows.append(
        {
            "experiment": experiment,
            "component": component,
            "paper_setting": paper_setting,
            "timed_probe": timed_probe,
            "probe_seconds": seconds,
            "probe_time": format_seconds(seconds),
            "scale_factor": scale_factor,
            "estimated_full_seconds": estimated,
            "estimated_full_time": format_seconds(estimated) if np.isfinite(estimated) else np.nan,
            **peak,
            "notes": notes,
        }
    )


def field_pass_work(num_epochs: int, train_size: int, aux_size: int, n_quantiles: int = 0) -> int:
    return num_epochs * (train_size + aux_size + n_quantiles)


def run_fm_train_probe(train_data: np.ndarray, n_batches: int, batch_size: int, device: torch.device, data_parallel: str):
    model = build_unet(n_channels=16, data_parallel=data_parallel).to(device)
    path = AffineProbPath(scheduler=CondOTScheduler())
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()
    tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True, num_workers=0)
    model.train()
    batches_done = 0
    for (x_1,) in loader:
        optimizer.zero_grad()
        x_1 = x_1.to(device)
        x_0 = torch.randn_like(x_1, device=device)
        t = torch.rand(x_1.shape[0], device=device)
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        loss = loss_fn(model(path_sample.x_t, path_sample.t), path_sample.dx_t)
        loss.backward()
        optimizer.step()
        batches_done += 1
        if batches_done >= n_batches:
            break
    return batches_done


def run_fm_sample_probe(image_shape: tuple[int, int], nsteps: int, nsamples: int, device: torch.device, data_parallel: str):
    model = build_unet(n_channels=16, data_parallel=data_parallel).to(device)
    h, w = image_shape
    x = torch.randn(nsamples, 1, h, w, device=device)
    t = torch.zeros(nsamples, device=device)
    step_size = 1.0 / nsteps
    model.eval()
    with torch.no_grad():
        for _ in range(nsteps):
            velocity = model(x, t)
            x = x + step_size * velocity
            t += step_size
    return x.detach().cpu()


def main() -> int:
    args = parse_args()
    ctx = setup_cli(args)
    global_seed(args.seed)

    era5 = load_era5_data(ctx.work_dir, ds_fact=args.ds_fact, era5_data=args.era5_data)
    tensors = make_precip_tensors(era5, args.num_years, args.batch_size, ctx.device)
    probe_aux_size = min(args.probe_aux_size, len(era5.tp_trim_ds_numpy))
    probe_idx = np.linspace(0, len(era5.tp_trim_ds_numpy) - 1, probe_aux_size, dtype=int)
    probe_lr = era5.tp_trim_ds_numpy[probe_idx]
    probe_hr = era5.tp_trim_numpy[probe_idx]
    probe_loader = DataLoader(
        TensorDataset(torch.tensor(probe_lr, dtype=torch.float32).unsqueeze(1), torch.tensor(probe_hr, dtype=torch.float32).unsqueeze(1)),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    rows = []

    cleanup()
    model = build_srcnn(args.ds_fact, args.data_parallel).to(ctx.device)
    _, seconds, peak = stopwatch(
        train_srcnn_mse_no_save,
        model,
        tensors.train_loader,
        probe_loader,
        args.mse_probe_epochs,
        3e-4,
        200,
        0.2,
        ctx.device,
        device=ctx.device,
    )
    scale = (500 * (tensors.train_size + len(era5.tp_trim_ds_numpy))) / max(args.mse_probe_epochs * (tensors.train_size + probe_aux_size), 1)
    add_row(
        rows,
        "ERA5-Land downscaling",
        "MSE baseline training",
        seconds,
        peak,
        scale,
        "SRCNN, 500 epochs, 0.5-year supervised split, full auxiliary evaluation",
        f"{args.mse_probe_epochs} epoch(s), {probe_aux_size} auxiliary fields",
    )

    model_mse = build_srcnn(args.ds_fact, args.data_parallel).to(ctx.device)
    mse_path = precip_mse_path(ctx.work_dir, args.num_years, args.ds_fact)
    if mse_path.exists():
        model_mse = load_state_flexible(model_mse, mse_path, ctx.device).to(ctx.device)
        source = f"loaded {mse_path.name}"
    else:
        source = "fresh initialization; MSE checkpoint not found"
    num_w1_days, _, w1_truemax_full = precip_tail_objects(era5, 150)
    probe_sorted_indices = np.argsort(np.max(probe_hr, axis=(1, 2)))
    w1_truedays_probe = probe_sorted_indices[-min(num_w1_days, probe_aux_size):]
    w1_truemax_probe = w1_truemax_full[-len(w1_truedays_probe):]

    cleanup()
    _, seconds, peak = stopwatch(
        train_precip_eta_no_save,
        model_mse,
        probe_loader,
        tensors.train_input,
        tensors.train_target,
        probe_lr,
        w1_truemax_probe,
        w1_truedays_probe,
        args.eta_probe_epochs,
        3e-4,
        1.0,
        30,
        True,
        args.seed,
        ctx.device,
        True,
        device=ctx.device,
    )
    scale = field_pass_work(150, tensors.train_size, len(era5.tp_trim_ds_numpy), num_w1_days) / max(
        field_pass_work(args.eta_probe_epochs, tensors.train_size, probe_aux_size, len(w1_truedays_probe)), 1
    )
    add_row(
        rows,
        "ERA5-Land downscaling",
        "eta continuation",
        seconds,
        peak,
        scale,
        "SRCNN eta, 150 epochs, lambda=1, omega=30, full LR inference set",
        f"{args.eta_probe_epochs} epoch(s), {probe_aux_size} auxiliary fields",
        notes=f"Initialized from {source}; no checkpoint saved.",
    )

    for field, paper_epochs, paper_years, paper_steps in [
        ("lores", 80, 25.0, 100),
        ("hires", 200, 0.5, 200),
    ]:
        train_data, _ = get_dgm_train(era5, field, paper_years)
        cleanup()
        _, seconds, peak = stopwatch(
            run_fm_train_probe,
            train_data,
            args.fm_probe_batches,
            args.fm_batch_size,
            ctx.device,
            args.data_parallel,
            device=ctx.device,
        )
        total_batches = int(np.ceil(len(train_data) / args.fm_batch_size)) * paper_epochs
        scale = total_batches / max(args.fm_probe_batches, 1)
        add_row(
            rows,
            "Flow Matching",
            f"{field} FM training",
            seconds,
            peak,
            scale,
            f"UNet16, {paper_years:g} years, {paper_epochs} epochs",
            f"{args.fm_probe_batches} optimizer batch(es)",
        )

        cleanup()
        _, seconds, peak = stopwatch(
            run_fm_sample_probe,
            tuple(train_data.shape[1:]),
            args.fm_sample_steps,
            args.fm_sample_count,
            ctx.device,
            args.data_parallel,
            device=ctx.device,
        )
        scale = (9044 * paper_steps) / max(args.fm_sample_count * args.fm_sample_steps, 1)
        add_row(
            rows,
            "Flow Matching",
            f"{field} FM sampling",
            seconds,
            peak,
            scale,
            f"9044 samples, {paper_steps} ODE steps",
            f"{args.fm_sample_count} samples, {args.fm_sample_steps} Euler proxy steps",
            notes="Probe uses Euler proxy steps for bounded timing.",
        )

    cleanup()
    model_eta = build_srcnn(args.ds_fact, args.data_parallel).to(ctx.device)
    if mse_path.exists():
        model_eta = load_state_flexible(model_eta, mse_path, ctx.device).to(ctx.device)
    lores_probe_loader = DataLoader(TensorDataset(torch.tensor(probe_lr, dtype=torch.float32).unsqueeze(1)), batch_size=256)
    _, seconds, peak = stopwatch(predict_field_array, model_eta, lores_probe_loader, ctx.device, False, device=ctx.device)
    scale = 9044 / max(probe_aux_size, 1)
    add_row(
        rows,
        "Flow Matching",
        "eta pass-through",
        seconds,
        peak,
        scale,
        "Pass 9044 generated LR samples through trained eta downscaler",
        f"{probe_aux_size} LR fields",
    )

    gevd = gevd_reference(era5, tau=0.95, num_w1_days=350)
    cleanup()
    model_gevd = build_srcnn(args.ds_fact, args.data_parallel).to(ctx.device)
    if mse_path.exists():
        model_gevd = load_state_flexible(model_gevd, mse_path, ctx.device).to(ctx.device)
    _, seconds, peak = stopwatch(
        train_precip_eta_no_save,
        model_gevd,
        probe_loader,
        tensors.train_input,
        tensors.train_target,
        probe_lr,
        torch.tensor(gevd["Q_gevd"][-len(w1_truedays_probe):], dtype=torch.float32),
        w1_truedays_probe,
        args.eta_probe_epochs,
        3e-4,
        1.0,
        1,
        True,
        args.seed,
        ctx.device,
        True,
        device=ctx.device,
    )
    scale = field_pass_work(150, tensors.train_size, len(era5.tp_trim_ds_numpy), 350) / max(
        field_pass_work(args.eta_probe_epochs, tensors.train_size, probe_aux_size, len(w1_truedays_probe)), 1
    )
    add_row(
        rows,
        "ERA5-Land GEVD",
        "GEVD eta continuation",
        seconds,
        peak,
        scale,
        "SRCNN eta, GEVD reference, 150 epochs, lambda=1, omega=1",
        f"{args.eta_probe_epochs} epoch(s), {probe_aux_size} auxiliary fields",
        notes=f"GEVD shape={gevd['fit']['shape']:.3f}, loc={gevd['fit']['location']:.3f}, scale={gevd['fit']['scale']:.3f}.",
    )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    save_table_csv(df, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
