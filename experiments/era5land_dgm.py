"""Command-line version of ERA5Land-DGM.ipynb."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiments.common import (
    add_common_path_args,
    build_unet,
    fm_checkpoint_path,
    fm_sample_path,
    get_dgm_train,
    load_era5_data,
    load_state_flexible,
    setup_cli,
)
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from utils import global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_path_args(parser)
    parser.add_argument("--era5-data", type=Path, default=None, help="ERA5-Land file. Defaults to WORK_DIR/data/era5land_USA_SouthEast_1999-2023_dailymax.nc.")
    parser.add_argument("--field", choices=["hires", "lores"], default="hires")
    parser.add_argument("--ds-fact", type=int, default=10)
    parser.add_argument("--num-years", type=float, default=25.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--nsteps", type=int, default=200)
    parser.add_argument("--nsamples", type=int, default=9044)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-samples", type=Path, default=None)
    parser.add_argument("--checkpoint-includes-batch-size", action="store_true")
    parser.add_argument("--data-parallel", choices=["always", "auto", "never"], default="always")
    return parser.parse_args()


def train_flow_matching(model, train_data: np.ndarray, epochs: int, batch_size: int, lr: float, device: torch.device, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    path = AffineProbPath(scheduler=CondOTScheduler())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cfm_loss = nn.MSELoss()
    fm_train = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(TensorDataset(fm_train), batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    model.to(device)
    best_loss = float("inf")
    for epoch in tqdm(range(epochs), desc="FM epochs"):
        model.train()
        cumulative_loss = 0.0
        for batch_idx, (x_1,) in enumerate(train_loader):
            optimizer.zero_grad()
            x_1 = x_1.to(device)
            x_0 = torch.randn_like(x_1, device=device)
            t = torch.rand(x_1.shape[0], device=device)
            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            loss = cfm_loss(model(path_sample.x_t, path_sample.t), path_sample.dx_t)
            cumulative_loss += float(loss.detach().cpu())
            loss.backward()
            optimizer.step()
        epoch_loss = cumulative_loss / max(batch_idx + 1, 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
        if ((epoch + 1) % 10 == 0) or ((epoch + 1) == epochs):
            print(f"Epoch {epoch + 1}/{epochs}, best_loss={best_loss:.6f}, epoch_loss={epoch_loss:.6f}")
    return best_loss


def dopri5_step(model, x, t, h):
    k1 = model(x, t)
    k2 = model(x + h * (1 / 5) * k1, t + h / 5)
    k3 = model(x + h * (3 / 40 * k1 + 9 / 40 * k2), t + 3 * h / 10)
    k4 = model(x + h * (44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3), t + 4 * h / 5)
    k5 = model(x + h * (19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212 / 729 * k4), t + 8 * h / 9)
    k6 = model(x + h * (9017 / 3168 * k1 - 355 / 33 * k2 + 46732 / 5247 * k3 + 49 / 176 * k4 - 5103 / 18656 * k5), t + h)
    return x + h * (35 / 384 * k1 + 500 / 1113 * k3 + 125 / 192 * k4 - 2187 / 6784 * k5 + 11 / 84 * k6)


def sample_flow_matching(model, image_shape: tuple[int, int], nsteps: int, nsamples: int, batch_size: int, device: torch.device) -> torch.Tensor:
    h, w = image_shape
    step_size = 1.0 / nsteps
    samples = []
    n_batches = nsamples // batch_size + int(nsamples % batch_size != 0)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx in range(n_batches):
            current_batch = batch_size if batch_idx < n_batches - 1 else nsamples - batch_size * batch_idx
            x = torch.randn(current_batch, 1, h, w, device=device)
            t = torch.zeros(current_batch, device=device)
            for _ in tqdm(range(nsteps), desc=f"sample batch {batch_idx + 1}/{n_batches}"):
                x = dopri5_step(model, x, t, step_size)
                t += step_size
            samples.append(x.detach().cpu())
    return torch.cat(samples, dim=0)[:nsamples]


def main() -> int:
    args = parse_args()
    ctx = setup_cli(args)
    global_seed(args.seed)

    checkpoint = args.checkpoint or fm_checkpoint_path(
        ctx.work_dir,
        args.field,
        args.n_channels,
        args.num_years,
        args.epochs,
        args.batch_size if args.checkpoint_includes_batch_size else None,
    )
    output_samples = args.output_samples or fm_sample_path(
        ctx.work_dir,
        args.field,
        args.nsamples,
        args.nsteps,
        args.n_channels,
        args.num_years,
        args.epochs,
    )

    if not args.train and not args.sample:
        print(f"No action requested. Checkpoint path: {checkpoint}")
        print(f"Sample path: {output_samples}")
        return 0

    era5 = load_era5_data(ctx.work_dir, ds_fact=args.ds_fact, era5_data=args.era5_data)
    train_data, inverse_transform = get_dgm_train(era5, args.field, args.num_years)
    image_shape = tuple(train_data.shape[1:])
    model = build_unet(n_channels=args.n_channels, data_parallel=args.data_parallel).to(ctx.device)
    if args.train:
        print(f"Training {args.field} Flow Matching model on shape={image_shape}; saving {checkpoint}")
        train_flow_matching(model, train_data, args.epochs, args.batch_size, args.lr, ctx.device, checkpoint)

    if args.sample:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing FM checkpoint: {checkpoint}. Pass --train or choose --checkpoint.")
        model = load_state_flexible(model, checkpoint, ctx.device).to(ctx.device)
        sample_batch_size = args.sample_batch_size or args.nsamples
        print(f"Sampling {args.nsamples} {args.field} fields with {args.nsteps} ODE steps from {checkpoint}")
        sampled = sample_flow_matching(model, image_shape, args.nsteps, args.nsamples, sample_batch_size, ctx.device)
        sampled_np = inverse_transform(sampled.squeeze(1).numpy())
        output_samples.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_samples, sampled_np)
        print(f"Saved samples: {output_samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
