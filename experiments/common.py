"""Shared command-line utilities mirroring the notebook experiment setup."""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import quad
from scipy.stats import genextreme
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for import_path in (REPO_ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from models import SRCNN, UNet
from revision_utils import (
    get_fixed_days_max_pos,
    get_max_time_pos,
    precip_spatial_metrics,
    predict_field_array,
)
from test_utils import test_model_mse
from utils import global_seed


DEFAULT_CODE_DIR = REPO_ROOT
DEFAULT_WORK_DIR = REPO_ROOT


@dataclass
class ToyScalarData:
    X: torch.Tensor
    Y: torch.Tensor
    GRIDS: torch.Tensor
    Y_GRID: torch.Tensor
    x_train: torch.Tensor
    y_train: torch.Tensor
    N: int


@dataclass
class ToyStateData:
    X: torch.Tensor
    U: torch.Tensor
    Y: torch.Tensor
    GRIDS: torch.Tensor
    U1_GRID: torch.Tensor
    U2_GRID: torch.Tensor
    x_train: torch.Tensor
    u_train: torch.Tensor
    N: int


@dataclass
class Era5Data:
    tp_numpy: np.ndarray
    tp_ds_numpy: np.ndarray
    tp_trim_numpy: np.ndarray
    tp_trim_ds_numpy: np.ndarray
    kept_days: np.ndarray
    trim_days: np.ndarray
    max_values: np.ndarray
    sorted_indices: np.ndarray
    sorted_max_values: np.ndarray


@dataclass
class PrecipTensors:
    train_input: torch.Tensor
    train_target: torch.Tensor
    test_input: torch.Tensor
    test_target: torch.Tensor
    train_loader: DataLoader
    test_loader: DataLoader
    train_size: int


@dataclass
class CliContext:
    work_dir: Path
    device: torch.device


@dataclass
class PrecipContext:
    work_dir: Path
    device: torch.device
    era5: Era5Data
    tensors: PrecipTensors


def add_common_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--code-dir", type=Path, default=None, help="Python helper checkout. Defaults to ETA_CODE_DIR or this repo.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Data/checkpoint root. Defaults to ETA_WORK_DIR or this repo.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:0, or cpu. Defaults to CUDA when available.")


def add_toy_args(parser: argparse.ArgumentParser, default_eta_seeds: list[int], include_lr: bool = False) -> None:
    parser.add_argument("--toy-data", type=Path, default=None, help="Toy dataset file. Defaults to WORK_DIR/data/toy.pth.")
    parser.add_argument("--mse-epochs", type=int, default=3000)
    parser.add_argument("--pretrain-epochs", type=int, default=1000)
    parser.add_argument("--eta-epochs", type=int, default=3000)
    parser.add_argument("--eta-seeds", type=int, nargs="+", default=default_eta_seeds)
    parser.add_argument("--lambda", dest="lambd", type=float, default=1.0)
    parser.add_argument("--omega", type=int, default=100)
    if include_lr:
        parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--skip-mse", action="store_true")
    parser.add_argument("--skip-eta", action="store_true")
    parser.add_argument("--save-npz", type=Path, default=None)


def add_precip_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--era5-data", type=Path, default=None, help="ERA5-Land file. Defaults to WORK_DIR/data/era5land_USA_SouthEast_1999-2023_dailymax.nc.")
    parser.add_argument("--ds-fact", type=int, default=10)
    parser.add_argument("--num-years", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--data-parallel", choices=["always", "auto", "never"], default="always")


def setup_cli(args: argparse.Namespace) -> CliContext:
    resolve_code_dir(args.code_dir)
    return CliContext(work_dir=resolve_work_dir(args.work_dir), device=resolve_device(args.device))


def resolve_code_dir(code_dir: Path | None = None) -> Path:
    path = code_dir or Path(os.environ.get("ETA_CODE_DIR", DEFAULT_CODE_DIR))
    if not path.exists():
        path = REPO_ROOT
    for import_path in (path, path / "src"):
        if import_path.exists() and str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))
    return path


def resolve_work_dir(work_dir: Path | None = None) -> Path:
    if work_dir is not None:
        return work_dir
    env_path = os.environ.get("ETA_WORK_DIR")
    if env_path:
        return Path(env_path)
    return DEFAULT_WORK_DIR


def resolve_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_load(path: Path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def maybe_data_parallel(model: nn.Module, mode: str = "auto") -> nn.Module:
    if mode == "always":
        return nn.DataParallel(model)
    if mode == "auto" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def build_fcnn(outdim: int, init: str, positive_output: bool = False, data_parallel: str = "always") -> nn.Module:
    from models import FCNN

    model = FCNN([256, 256, 256], activation="psilu", indim=2, outdim=outdim, init=init, positive_output=positive_output)
    return maybe_data_parallel(model, data_parallel)


def load_state_flexible(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    state = torch_load(path, map_location=device)
    try:
        model.load_state_dict(state)
        return model
    except RuntimeError:
        state_has_module = any(key.startswith("module.") for key in state)
        model_is_parallel = isinstance(model, nn.DataParallel)
        if state_has_module and not model_is_parallel:
            stripped = {key.replace("module.", "", 1): value for key, value in state.items()}
            model.load_state_dict(stripped)
        elif (not state_has_module) and model_is_parallel:
            model.module.load_state_dict(state)
        elif state_has_module and model_is_parallel:
            model.load_state_dict(state)
        else:
            wrapped = nn.DataParallel(model)
            wrapped.load_state_dict(state)
            return wrapped
    return model


def load_toy_scalar_data(work_dir: Path, toy_data: Path | None = None) -> ToyScalarData:
    data_path = toy_data or work_dir / "data/toy.pth"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing toy data: {data_path}")
    data = torch_load(data_path, map_location="cpu")
    X = data["X_all"].float()
    Y = data["Y_all"].float()
    GRIDS = data["in_grid"].float()
    Y_GRID = data["out_grid"].float()
    x_train = data["x_train"].float()
    y_train = data["y_train"].float()
    N = int(np.sqrt(GRIDS.shape[0]))
    return ToyScalarData(X=X, Y=Y, GRIDS=GRIDS, Y_GRID=Y_GRID.reshape(N, N), x_train=x_train, y_train=y_train, N=N)


def fourier_mode_2d(x: torch.Tensor, freq=(1 / 6, 1 / 8), phase=(0, 0), amplitude=0.1) -> torch.Tensor:
    return -amplitude * torch.sin(2 * np.pi * freq[0] * x[..., 0] + phase[0]) * torch.sin(
        2 * np.pi * freq[1] * x[..., 1] + phase[1]
    )


def state_observable(u: torch.Tensor) -> torch.Tensor:
    return 2 * torch.abs(u[:, 0]) + 0.5 * torch.abs(u[:, 1])


def load_toy_state_data(work_dir: Path, toy_data: Path | None = None) -> ToyStateData:
    scalar = load_toy_scalar_data(work_dir, toy_data=toy_data)
    U1 = scalar.Y.reshape(-1, 1)
    U2 = fourier_mode_2d(scalar.X).reshape(-1, 1)
    U = torch.cat((U1, U2), dim=1)
    u_train = torch.cat((scalar.y_train.reshape(-1, 1), fourier_mode_2d(scalar.x_train).reshape(-1, 1)), dim=1)
    U1_GRID = scalar.Y_GRID
    U2_GRID = fourier_mode_2d(scalar.GRIDS).reshape(scalar.N, scalar.N)
    Y = state_observable(U)
    return ToyStateData(
        X=scalar.X,
        U=U,
        Y=Y,
        GRIDS=scalar.GRIDS,
        U1_GRID=U1_GRID,
        U2_GRID=U2_GRID,
        x_train=scalar.x_train,
        u_train=u_train,
        N=scalar.N,
    )


def pretrain_toy_scalar(seed: int, epochs: int, device: torch.device):
    from train_utils import generate_2d_gaussian, grf_pretrain

    global_seed(seed)
    model = build_fcnn(outdim=1, init="xavier normal", positive_output=False).to(device)
    gaussian_field, _ = generate_2d_gaussian(seed=seed, sigma=0.3)
    return grf_pretrain(gaussian_field, model, epochs, n_grid=50, grid_step=2, device=device)


def pretrain_toy_state(data: ToyStateData, seed: int, epochs: int, device: torch.device):
    from train_utils import generate_2d_gaussian, grf_pretrain

    global_seed(seed)
    model = build_fcnn(outdim=2, init="kaiming normal", positive_output=False).to(device)
    gaussian_field, _ = generate_2d_gaussian(seed=seed, sigma=0.3)
    pretrain_field = torch.cat((gaussian_field.unsqueeze(-1), data.U2_GRID.unsqueeze(-1)), dim=-1)
    return grf_pretrain(pretrain_field, model, epochs, n_grid=50, grid_step=2, device=device)


def print_metric_rows(rows: Iterable[dict], include_lambda: bool = False) -> None:
    for row in rows:
        lambd = f" lambda={row['lambda']}" if include_lambda and "lambda" in row else ""
        print(
            f"{row['run']}: seed={row['seed']}{lambd} "
            f"final_train_mse={row['final_train_mse']:.6g} tail_w1={row['tail_w1']:.6g}"
        )


def toy_scalar_quantiles() -> torch.Tensor:
    return torch.cat(
        (
            torch.linspace(0.0, 0.0001, 10),
            torch.linspace(0.0001, 0.001, 10),
            torch.linspace(0.001, 0.01, 10),
            torch.linspace(0.01, 0.1, 9),
            torch.linspace(0.1, 0.9, 20),
            torch.linspace(0.9, 0.99, 21),
            torch.linspace(0.99, 0.999, 21),
            torch.linspace(0.999, 0.9999, 21),
            torch.linspace(0.9999, 0.99999, 21),
            torch.linspace(0.99999, 0.999999, 21),
            torch.linspace(0.999999, 0.9999999, 21),
        )
    )


def toy_state_quantiles() -> torch.Tensor:
    return torch.cat(
        (
            torch.linspace(0, 0.9, 41),
            torch.linspace(0.9, 0.99, 21),
            torch.linspace(0.99, 0.999, 21),
            torch.linspace(0.999, 0.9999, 21),
            torch.linspace(0.9999, 0.99999, 21),
            torch.linspace(0.99999, 0.999999, 21),
            torch.linspace(0.999999, 0.9999999, 21),
        )
    )


def load_era5_data(work_dir: Path, ds_fact: int = 10, trim_tail_thresh: float = 240.0, era5_data: Path | None = None) -> Era5Data:
    import xarray as xr

    era5_path = era5_data or work_dir / "data/era5land_USA_SouthEast_1999-2023_dailymax.nc"
    if not era5_path.exists():
        raise FileNotFoundError(f"Missing ERA5-Land data: {era5_path}")
    dataset = xr.open_dataset(era5_path, engine="netcdf4")
    tp_numpy = (dataset["tp"].values * 1000).astype(np.float32)
    tp_ds_numpy = tp_numpy[:, ::ds_fact, ::ds_fact]
    max_values_original = np.max(tp_numpy, axis=(1, 2))
    sorted_indices_original = np.argsort(max_values_original)
    sorted_max_values_original = max_values_original[sorted_indices_original]
    num_trim_days = int(np.sum(sorted_max_values_original > trim_tail_thresh))
    trim_days = sorted_indices_original[-num_trim_days:] if num_trim_days > 0 else np.array([], dtype=int)
    kept_days = np.delete(np.arange(len(tp_numpy)), trim_days)
    tp_trim_numpy = tp_numpy[kept_days]
    tp_trim_ds_numpy = tp_ds_numpy[kept_days]
    max_values = np.max(tp_trim_numpy, axis=(1, 2))
    sorted_indices = np.argsort(max_values)
    sorted_max_values = max_values[sorted_indices]
    return Era5Data(
        tp_numpy=tp_numpy,
        tp_ds_numpy=tp_ds_numpy,
        tp_trim_numpy=tp_trim_numpy,
        tp_trim_ds_numpy=tp_trim_ds_numpy,
        kept_days=kept_days,
        trim_days=trim_days,
        max_values=max_values,
        sorted_indices=sorted_indices,
        sorted_max_values=sorted_max_values,
    )


def make_precip_tensors(era5: Era5Data, num_years: float, batch_size: int, device: torch.device) -> PrecipTensors:
    tp_trim_ds_tensor = torch.tensor(era5.tp_trim_ds_numpy, dtype=torch.float32).unsqueeze(1)
    tp_trim_tensor = torch.tensor(era5.tp_trim_numpy, dtype=torch.float32).unsqueeze(1)
    train_size = int((num_years / 25) * len(tp_trim_ds_tensor))
    train_input = tp_trim_ds_tensor[:train_size].to(device)
    train_target = tp_trim_tensor[:train_size].to(device)
    test_input = deepcopy(tp_trim_ds_tensor).to(device)
    test_target = deepcopy(tp_trim_tensor).to(device)
    train_loader = DataLoader(TensorDataset(train_input, train_target), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_input, test_target), batch_size=25 * batch_size, shuffle=False)
    return PrecipTensors(
        train_input=train_input,
        train_target=train_target,
        test_input=test_input,
        test_target=test_target,
        train_loader=train_loader,
        test_loader=test_loader,
        train_size=train_size,
    )


def prepare_precip_context(args: argparse.Namespace) -> PrecipContext:
    ctx = setup_cli(args)
    global_seed(args.seed)
    era5 = load_era5_data(ctx.work_dir, ds_fact=args.ds_fact, era5_data=args.era5_data)
    tensors = make_precip_tensors(era5, args.num_years, args.batch_size, ctx.device)
    return PrecipContext(work_dir=ctx.work_dir, device=ctx.device, era5=era5, tensors=tensors)


def build_srcnn(ds_fact: int = 10, data_parallel: str = "always") -> nn.Module:
    return maybe_data_parallel(SRCNN(hidden_dim=64, num_blocks=3, scale_factor=ds_fact), data_parallel)


def load_required_mse_model(args: argparse.Namespace, work_dir: Path, device: torch.device) -> nn.Module:
    model = build_srcnn(args.ds_fact, args.data_parallel).to(device)
    path = precip_mse_path(work_dir, args.num_years, args.ds_fact)
    if not path.exists():
        raise FileNotFoundError(f"Missing MSE checkpoint: {path}. Run experiments/era5land.py --train-mse first.")
    return load_state_flexible(model, path, device).to(device)


def precip_mse_path(work_dir: Path, num_years: float, ds_fact: int) -> Path:
    return work_dir / "models/precip-srcnn" / f"srcnn-mse-{num_years}yr-{ds_fact}ds.pth"


def precip_eta_path(work_dir: Path, num_years: float, ds_fact: int, w1_tail_thresh, omega: int) -> Path:
    return work_dir / "models/precip-srcnn" / f"srcnn-eta-{num_years}yr-{ds_fact}ds-{w1_tail_thresh}tail-{omega}omega.pth"


def precip_gevd_eta_path(work_dir: Path, num_years: float, ds_fact: int, w1_tail_thresh, omega: int) -> Path:
    return work_dir / "models/precip-srcnn" / f"srcnn-eta-gevd-{num_years}yr-{ds_fact}ds-{int(w1_tail_thresh)}tail-{omega}omega.pth"


def alpha_tag(alpha: float) -> str:
    return f"{float(alpha):g}".replace("-", "m").replace(".", "p")


def precip_misspec_eta_path(work_dir: Path, alpha: float, num_years: float, ds_fact: int, w1_tail_thresh, omega: int) -> Path:
    misspec_dir = work_dir / "models/precip-srcnn/misspec"
    year_tag = alpha_tag(num_years)
    filename = (
        f"srcnn-eta-misspec-train_model_eta-alpha_{alpha_tag(alpha)}-"
        f"{year_tag}yr-{ds_fact}ds-{int(w1_tail_thresh)}tail-{omega}omega.pth"
    )
    return misspec_dir / filename


def train_srcnn_mse_cli(model: nn.Module, tensors: PrecipTensors, num_epochs: int, lr: float, device: torch.device) -> dict:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.2)
    mse_loss = nn.MSELoss()
    history = {"train_mse": [], "test_mse": []}
    for epoch in range(num_epochs):
        model.train()
        running_train = 0.0
        for input_data, target in tensors.train_loader:
            input_data = input_data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = mse_loss(model(input_data), target)
            loss.backward()
            optimizer.step()
            running_train += float(loss.detach().cpu())
        scheduler.step()
        _, test_mse = test_model_mse(model, tensors.test_loader, device=device)
        history["train_mse"].append(running_train / max(len(tensors.train_loader), 1))
        history["test_mse"].append(test_mse)
        if ((epoch + 1) % 25 == 0) or ((epoch + 1) == num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}, train MSE: {history['train_mse'][-1]:.6f}, test MSE: {test_mse:.6f}")
    return history


def precip_tail_objects(era5: Era5Data, w1_tail_thresh: float) -> tuple[int, np.ndarray, torch.Tensor]:
    num_w1_days = int(np.sum(era5.sorted_max_values > w1_tail_thresh))
    if num_w1_days == 0:
        raise ValueError(f"No ERA5 fields exceed w1_tail_thresh={w1_tail_thresh}")
    w1_truedays = era5.sorted_indices[-num_w1_days:]
    w1_truemax = torch.tensor(era5.sorted_max_values[-num_w1_days:], dtype=torch.float32)
    return num_w1_days, w1_truedays, w1_truemax


def evaluate_precip_model(model: nn.Module, test_loader: DataLoader, w1_truemax: torch.Tensor, device: torch.device) -> tuple[np.ndarray, float, float]:
    w1_loss = lambda x, y: torch.abs(x - y).mean()
    full_output, test_mse = test_model_mse(model, test_loader, device=device)
    max_values_output = np.max(full_output, axis=(1, 2))
    sorted_max_values_output = np.sort(max_values_output)
    w1_max_test = torch.tensor(sorted_max_values_output[-len(w1_truemax):], dtype=torch.float32, device=device)
    w1_loss_test = float(w1_loss(w1_max_test, w1_truemax.to(device)).detach().cpu())
    return full_output, test_mse, w1_loss_test


def train_precip_eta_cli(
    model_eta: nn.Module,
    tensors: PrecipTensors,
    era5: Era5Data,
    seed: int,
    num_epochs: int,
    w1_truemax: torch.Tensor,
    w1_truedays: np.ndarray,
    lambd_: float,
    omega: int,
    varying_days: bool,
    lr: float,
    save_path: Path | None,
    device: torch.device,
) -> tuple[nn.Module, dict[str, list[float]]]:
    global_seed(seed)
    model_eta.to(device)
    optimizer = torch.optim.Adam(model_eta.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    w1_loss = lambda x, y: torch.abs(x - y).mean()
    w1_truemax = w1_truemax.to(device)
    full_output_eta, _ = test_model_mse(model_eta, tensors.test_loader, device=device)
    num_w1_days = len(w1_truemax)
    if varying_days:
        w1_days, w1_max_pos = get_max_time_pos(full_output_eta, num_w1_days)
    else:
        w1_days = w1_truedays
        w1_max_pos = get_fixed_days_max_pos(full_output_eta, w1_truedays)

    history = {"train_mse": [], "train_w1": [], "test_mse": [], "test_w1": []}
    best_w1 = math.inf
    if save_path is not None:
        ensure_parent(save_path)

    for epoch in range(num_epochs):
        model_eta.train()
        optimizer.zero_grad()
        w1_input = torch.tensor(era5.tp_trim_ds_numpy[w1_days], dtype=torch.float32).unsqueeze(1).to(device)
        mse_output = model_eta(tensors.train_input)
        w1_output = model_eta(w1_input).squeeze()
        w1_max = w1_output[tuple(w1_max_pos.to(device))]
        mse_loss_val = mse_loss(mse_output, tensors.train_target)
        w1_loss_val = w1_loss(w1_max, w1_truemax)
        loss = mse_loss_val + lambd_ * w1_loss_val
        loss.backward()
        optimizer.step()

        full_output_eta, test_mse, w1_loss_test = evaluate_precip_model(model_eta, tensors.test_loader, w1_truemax, device)
        if (epoch + 1) % omega == 0:
            if varying_days:
                w1_days, w1_max_pos = get_max_time_pos(full_output_eta, num_w1_days)
            else:
                w1_max_pos = get_fixed_days_max_pos(full_output_eta, w1_days)

        if w1_loss_test < best_w1:
            best_w1 = w1_loss_test
            if save_path is not None:
                torch.save(model_eta.state_dict(), save_path)

        history["train_mse"].append(float(mse_loss_val.detach().cpu()))
        history["train_w1"].append(float(w1_loss_val.detach().cpu()))
        history["test_mse"].append(test_mse)
        history["test_w1"].append(w1_loss_test)
        if ((epoch + 1) % 25 == 0) or ((epoch + 1) == num_epochs):
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Best W1: {best_w1:.6f}, Curr W1: {w1_loss_test:.6f}, Test MSE: {test_mse:.6f}"
            )

    if save_path is not None and save_path.exists():
        model_eta = load_state_flexible(model_eta, save_path, device)
        model_eta.to(device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model_eta, history


def fit_gev_with_cutoff(data: np.ndarray, cutoff: float, scale_lift: float = 4.0) -> dict:
    filtered_data = data[data <= cutoff]
    if len(filtered_data) == 0:
        raise ValueError("No data points are below the GEVD cutoff.")
    shape, loc, scale = genextreme.fit(filtered_data)
    scale = scale + scale_lift
    c = quad(lambda t: genextreme.pdf(t, shape, loc=loc, scale=scale), -np.inf, cutoff)[0]
    x = np.linspace(min(filtered_data), cutoff, len(data))
    pdf = np.array([genextreme.pdf(val, shape, loc=loc, scale=scale) / c for val in x])
    return {"shape": shape, "location": loc, "scale": scale, "cutoff": cutoff, "c": c, "x": x, "pdf": pdf}


def gevd_reference(era5: Era5Data, tau: float = 0.95, num_w1_days: int = 350, scale_lift: float = 4.0) -> dict:
    cutoff = float(np.ceil(max(era5.max_values)) + 20)
    fit = fit_gev_with_cutoff(era5.max_values, cutoff=cutoff, scale_lift=scale_lift)
    q = torch.linspace(tau, 1, num_w1_days)
    q_np = q.detach().cpu().numpy()
    Q_gevd = np.array([genextreme.ppf(float(prob) * fit["c"], fit["shape"], fit["location"], fit["scale"]) for prob in q_np])
    y_tau = float(genextreme.ppf(tau * fit["c"], fit["shape"], fit["location"], fit["scale"]))
    w1_truedays = era5.sorted_indices[-num_w1_days:]
    return {"fit": fit, "quantiles": q, "Q_gevd": Q_gevd, "y_tau": y_tau, "w1_truedays": w1_truedays}


def empirical_quantiles(values: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.quantile(np.ravel(values).astype(float), np.ravel(q).astype(float))


def save_table_csv(df: pd.DataFrame, path: Path | None) -> None:
    if path is None:
        return
    ensure_parent(path)
    df.to_csv(path, index=False)
    print(f"Saved table: {path}")


def save_npz(path: Path | None, **arrays) -> None:
    if path is None:
        return
    ensure_parent(path)
    np.savez_compressed(path, **arrays)
    print(f"Saved arrays: {path}")


def dgm_transform(data: np.ndarray) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    clipped = np.clip(data, 1e0, None)
    logged = np.log(clipped)
    train_mean = logged.mean()
    train_std = logged.std()
    transformed = (logged - train_mean) / train_std
    inverse_transform = lambda x: np.exp(x * train_std + train_mean)
    return transformed.astype(np.float32), inverse_transform


def get_dgm_train(era5: Era5Data, field: str, num_years: float) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    train_size = int((num_years / 25) * len(era5.tp_trim_ds_numpy))
    if field == "hires":
        data = era5.tp_trim_numpy[:train_size]
    elif field == "lores":
        data = era5.tp_trim_ds_numpy[:train_size]
    else:
        raise ValueError(f"Unknown field={field}")
    return dgm_transform(data)


def build_unet(n_channels: int = 16, data_parallel: str = "always") -> nn.Module:
    model = UNet(
        image_channels=1,
        n_channels=n_channels,
        ch_mults=(1, 2, 2, 4),
        is_attn=(False, False, True, True),
        n_blocks=1,
    )
    return maybe_data_parallel(model, data_parallel)


def year_tag(num_years: float) -> str:
    return f"{float(num_years):g}"


def fm_checkpoint_path(work_dir: Path, field: str, n_channels: int, num_years: float, epochs: int, batch_size: int | None = None) -> Path:
    if batch_size is None:
        filename = f"{field}-fm-unet{n_channels}-{year_tag(num_years)}yr-{epochs}epoch.pth"
    else:
        filename = f"{field}-fm-unet{n_channels}-{year_tag(num_years)}yr-{epochs}epoch-{batch_size}bsize.pth"
    return work_dir / "models/fm" / filename


def fm_sample_path(work_dir: Path, field: str, nsamples: int, nsteps: int, n_channels: int, num_years: float, epochs: int) -> Path:
    return work_dir / "samples" / f"{nsamples}samples-{field}-fm-{nsteps}step-unet{n_channels}-{year_tag(num_years)}yr-{epochs}epoch.npy"


def summarize_maxima(name: str, fields: np.ndarray) -> dict[str, float | str]:
    maxima = np.max(fields, axis=(1, 2))
    return {
        "name": name,
        "n": int(len(maxima)),
        "mean_max": float(np.mean(maxima)),
        "q95_max": float(np.quantile(maxima, 0.95)),
        "q99_max": float(np.quantile(maxima, 0.99)),
        "max": float(np.max(maxima)),
    }


def stopwatch(func: Callable, *args, device: torch.device | None = None, **kwargs):
    if torch.cuda.is_available() and device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = func(*args, **kwargs)
    if torch.cuda.is_available() and device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = {
            "peak_vram_allocated_GB": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_vram_reserved_GB": torch.cuda.max_memory_reserved(device) / 1024**3,
        }
    else:
        peak = {"peak_vram_allocated_GB": np.nan, "peak_vram_reserved_GB": np.nan}
    return result, time.perf_counter() - start, peak
