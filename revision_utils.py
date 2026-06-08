"""Revision-only helpers for reviewer-requested diagnostics.

These functions are intentionally separate from the original experiment helpers so
that existing notebook logic and saved artifacts remain untouched.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import math
import platform
import sys
import time
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def to_numpy(array):
    """Return a detached CPU numpy array without modifying the input."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def format_seconds(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:d}:{minutes:02d}:{secs:05.2f}"


def reset_peak_memory(device=None) -> None:
    """Reset CUDA peak-memory counters when CUDA is available."""
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.reset_peak_memory_stats()
        else:
            torch.cuda.reset_peak_memory_stats(device)


def get_peak_vram(device=None) -> dict[str, float]:
    """Return peak allocated and reserved CUDA memory in GB."""
    if not torch.cuda.is_available():
        return {
            "peak_vram_allocated_GB": np.nan,
            "peak_vram_reserved_GB": np.nan,
        }
    if device is None:
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
    else:
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
    gb = 1024**3
    return {
        "peak_vram_allocated_GB": allocated / gb,
        "peak_vram_reserved_GB": reserved / gb,
    }


def measure_wall_time(func: Callable, *args, device=None, **kwargs):
    """Run a callable and return ``(result, elapsed_seconds, peak_vram_dict)``."""
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    reset_peak_memory(device)
    start = time.perf_counter()
    result = func(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return result, elapsed, get_peak_vram(device)


@contextlib.contextmanager
def disable_checkpoint_saving():
    """Temporarily replace torch/NumPy save calls with no-op logging.

    The yielded list records attempted save targets. This is useful in timing
    notebooks where the same control flow as the paper is desired but new files
    must not be written.
    """
    blocked_paths = []
    torch_save = torch.save
    np_save = np.save

    def _record_only(path, *args, **kwargs):
        blocked_paths.append(str(path))
        return None

    torch.save = _record_only
    np.save = _record_only
    try:
        yield blocked_paths
    finally:
        torch.save = torch_save
        np.save = np_save


def run_without_saving(func: Callable, *args, **kwargs):
    """Run a callable while blocking accidental torch.save and np.save calls."""
    with disable_checkpoint_saving() as blocked_paths:
        result = func(*args, **kwargs)
    return result, blocked_paths


def hardware_summary(device=None) -> dict[str, object]:
    """Collect lightweight hardware and runtime metadata for notebook display."""
    cuda_available = torch.cuda.is_available()
    if device is None:
        device = torch.device("cuda" if cuda_available else "cpu")
    gpu_name = None
    gpu_total_vram_GB = np.nan
    if cuda_available:
        idx = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(idx)
        gpu_total_vram_GB = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    return {
        "device_used": str(device),
        "gpu_name": gpu_name,
        "gpu_total_vram_GB": gpu_total_vram_GB,
        "cuda_available": cuda_available,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": sys.version.split()[0],
        "os": platform.platform(),
        "cpu_model": platform.processor() or platform.machine(),
    }


def tail_w1_quantile(samples, reference, quantiles) -> float:
    """Approximate one-dimensional W1 over supplied quantile levels."""
    sample_values = np.ravel(to_numpy(samples)).astype(float)
    reference_values = np.ravel(to_numpy(reference)).astype(float)
    q = np.ravel(to_numpy(quantiles)).astype(float)
    q = np.clip(q, 0.0, 1.0)
    sample_q = np.quantile(sample_values, q)
    reference_q = np.quantile(reference_values, q)
    return float(np.mean(np.abs(sample_q - reference_q)))


def empirical_quantiles(values, quantiles) -> np.ndarray:
    """Return empirical quantiles as a numpy array."""
    return np.quantile(np.ravel(to_numpy(values)).astype(float), np.ravel(to_numpy(quantiles)).astype(float))


def precip_spatial_metrics(pred, true, data_range: float | None = None, eps: float = 1e-12) -> dict[str, float]:
    """Compute full-field RMSE, rRMSE, mean SSIM, and mean spatial correlation."""
    pred = to_numpy(pred).astype(float)
    true = to_numpy(true).astype(float)
    if pred.shape != true.shape:
        raise ValueError(f"pred and true must have the same shape, got {pred.shape} and {true.shape}")

    diff = pred - true
    rmse_grid = float(np.sqrt(np.mean(diff**2)))
    denom = float(np.sqrt(np.sum(true**2)))
    rrmse_grid = float(np.sqrt(np.sum(diff**2)) / (denom + eps))
    mean_corr = mean_spatial_corr(pred, true, eps=eps)
    mean_ssim = mean_structural_similarity(pred, true, data_range=data_range)
    return {
        "RMSE_grid": rmse_grid,
        "rRMSE_grid": rrmse_grid,
        "mean_SSIM": mean_ssim,
        "mean_spatial_corr": mean_corr,
    }


def mean_spatial_corr(pred, true, eps: float = 1e-12) -> float:
    """Mean per-sample Pearson correlation over flattened fields."""
    pred = to_numpy(pred).astype(float).reshape(pred.shape[0], -1)
    true = to_numpy(true).astype(float).reshape(true.shape[0], -1)
    pred_centered = pred - pred.mean(axis=1, keepdims=True)
    true_centered = true - true.mean(axis=1, keepdims=True)
    numerator = np.sum(pred_centered * true_centered, axis=1)
    denominator = np.sqrt(np.sum(pred_centered**2, axis=1)) * np.sqrt(np.sum(true_centered**2, axis=1))
    corr = numerator / (denominator + eps)
    corr[denominator <= eps] = np.nan
    return float(np.nanmean(corr))


def mean_structural_similarity(pred, true, data_range: float | None = None) -> float:
    """Mean SSIM using skimage when available; returns NaN if unavailable."""
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return float("nan")

    pred = to_numpy(pred).astype(float)
    true = to_numpy(true).astype(float)
    if data_range is None:
        data_range = float(np.nanmax(true) - np.nanmin(true))
    if not np.isfinite(data_range) or data_range <= 0:
        return float("nan")

    scores = []
    for pred_i, true_i in zip(pred, true):
        try:
            scores.append(structural_similarity(pred_i, true_i, data_range=data_range))
        except ValueError:
            scores.append(np.nan)
    return float(np.nanmean(scores))


def nearest_upsample_lr(lr_fields, target_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor upsample LR fields to the target HR shape."""
    lr_fields = to_numpy(lr_fields)
    if lr_fields.ndim != 3:
        raise ValueError("lr_fields must have shape (N, H_lr, W_lr)")
    scale_h = int(target_shape[0] // lr_fields.shape[1])
    scale_w = int(target_shape[1] // lr_fields.shape[2])
    upsampled = np.repeat(np.repeat(lr_fields, scale_h, axis=1), scale_w, axis=2)
    return upsampled[:, : target_shape[0], : target_shape[1]]


def precipitation_metrics_table(
    true_fields,
    predictions: Mapping[str, np.ndarray],
    threshold_quantiles: Iterable[float] = (0.95, 0.975, 0.99),
    data_range: float | None = None,
) -> pd.DataFrame:
    """Build the stratified precipitation spatial-fidelity table."""
    true_fields = to_numpy(true_fields)
    max_values = np.max(true_fields, axis=(1, 2))
    subsets = [("full", np.nan, np.nan, np.ones(len(true_fields), dtype=bool))]
    for q in threshold_quantiles:
        threshold = float(np.quantile(max_values, q))
        subsets.append((f"max >= Q_true({q:g})", q, threshold, max_values >= threshold))

    rows = []
    for subset_name, q, threshold, mask in subsets:
        for method, pred in predictions.items():
            metrics = precip_spatial_metrics(to_numpy(pred)[mask], true_fields[mask], data_range=data_range)
            rows.append({
                "subset": subset_name,
                "threshold_quantile": q,
                "threshold_value": threshold,
                "method": method,
                "N_samples": int(mask.sum()),
                **metrics,
            })
    return pd.DataFrame(rows)


def conditional_mean_values(fields, threshold: float) -> np.ndarray:
    """Per-field mean over cells exceeding a threshold."""
    fields = to_numpy(fields).astype(float)
    mask = fields >= threshold
    counts = mask.sum(axis=(1, 2))
    sums = (fields * mask).sum(axis=(1, 2))
    out = sums / np.where(counts > 0, counts, np.nan)
    return out


def weighted_coverage_values(fields, threshold: float) -> np.ndarray:
    """Per-field fraction of precipitation mass above a threshold."""
    fields = to_numpy(fields).astype(float)
    total = fields.sum(axis=(1, 2))
    tail = (fields * (fields >= threshold)).sum(axis=(1, 2))
    return tail / np.where(np.abs(total) > 0, total, np.nan)


def mean_wasserstein_stat_error(pred, true, thresholds, stat_fn: Callable[[np.ndarray, float], np.ndarray]) -> float:
    """Average empirical one-dimensional W1 error for thresholded field statistics."""
    from scipy.stats import wasserstein_distance

    errors = []
    for threshold in thresholds:
        pred_vals = stat_fn(pred, threshold)
        true_vals = stat_fn(true, threshold)
        pred_vals = pred_vals[np.isfinite(pred_vals)]
        true_vals = true_vals[np.isfinite(true_vals)]
        if len(pred_vals) == 0 or len(true_vals) == 0:
            errors.append(np.nan)
        else:
            errors.append(wasserstein_distance(pred_vals, true_vals))
    return float(np.nanmean(errors))


def predict_field_array(model, data_loader, device=None, target_is_required: bool = True) -> np.ndarray:
    """Run a field model over a loader and return squeezed numpy predictions."""
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)
            outputs.append(model(x).detach().squeeze().cpu())
    return torch.cat(outputs, dim=0).numpy()


def get_max_time_pos(full_output: np.ndarray, num_w1_days: int):
    """Return IICT day indices and component positions for field maxima."""
    max_values_output = np.max(full_output, axis=(1, 2))
    sorted_indices_output = np.argsort(max_values_output)
    w1_days = sorted_indices_output[-num_w1_days:]
    max_indices = np.argmax(full_output.reshape(full_output.shape[0], -1), axis=1)
    max_pos_row, max_pos_col = np.unravel_index(max_indices, full_output.shape[1:])
    max_pos = np.stack([np.arange(full_output.shape[0]), max_pos_row, max_pos_col], axis=1).T
    w1_max_pos = torch.tensor(max_pos[:, w1_days])
    w1_max_pos = torch.cat((torch.arange(num_w1_days).reshape(1, -1), w1_max_pos[1:, :]), dim=0)
    w1_max_pos.requires_grad = False
    return w1_days, w1_max_pos


def get_fixed_days_max_pos(full_output: np.ndarray, w1_days):
    """Return component positions for a fixed set of IICT days."""
    w1_days = np.asarray(w1_days)
    max_indices = np.argmax(full_output.reshape(full_output.shape[0], -1), axis=1)
    max_pos_row, max_pos_col = np.unravel_index(max_indices, full_output.shape[1:])
    max_pos = np.stack([np.arange(full_output.shape[0]), max_pos_row, max_pos_col], axis=1).T
    fixed_days_max_pos = torch.tensor(max_pos[:, w1_days])
    fixed_days_max_pos = torch.cat((torch.arange(len(w1_days)).reshape(1, -1), fixed_days_max_pos[1:, :]), dim=0)
    fixed_days_max_pos.requires_grad = False
    return fixed_days_max_pos


def make_w1_input(lr_fields: np.ndarray, w1_days, device=None) -> torch.Tensor:
    """Create the LR tensor used in the IICT W1 update."""
    tensor = torch.tensor(lr_fields[w1_days], dtype=torch.float32).unsqueeze(1)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def train_srcnn_mse_no_save(
    model,
    train_loader,
    test_loader,
    num_epochs: int,
    lr: float,
    scheduler_step_size: int | None = None,
    scheduler_gamma: float = 1.0,
    device=None,
) -> dict[str, list[float]]:
    """Train an SRCNN ERM baseline without writing checkpoints."""
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if scheduler_step_size is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    mse_loss = nn.MSELoss()
    history = {"train_mse": [], "test_mse": []}
    for _ in range(num_epochs):
        model.train()
        running_train = 0.0
        for input_data, target in train_loader:
            input_data = input_data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = mse_loss(model(input_data), target)
            loss.backward()
            optimizer.step()
            running_train += loss.item()
        if scheduler is not None:
            scheduler.step()

        model.eval()
        running_test = 0.0
        with torch.no_grad():
            for test_in, test_tar in test_loader:
                test_in = test_in.to(device)
                test_tar = test_tar.to(device)
                running_test += mse_loss(model(test_in), test_tar).item()
        history["train_mse"].append(running_train / max(len(train_loader), 1))
        history["test_mse"].append(running_test / max(len(test_loader), 1))
    return history


def train_precip_eta_no_save(
    model_eta,
    test_loader,
    mse_input,
    mse_target,
    lr_fields_for_w1: np.ndarray,
    w1_truemax,
    w1_truedays,
    num_epochs: int,
    lr: float,
    lambd_: float,
    omega: int,
    varying_days: bool,
    seed: int | None = None,
    device=None,
    keep_best_by_w1: bool = True,
) -> tuple[object, dict[str, list[float]], np.ndarray]:
    """Train the precipitation eta map without writing checkpoints."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    if device is None:
        device = next(model_eta.parameters()).device

    model_eta.to(device)
    mse_input = mse_input.to(device)
    mse_target = mse_target.to(device)
    w1_truemax = torch.as_tensor(w1_truemax, dtype=torch.float32, device=device)
    w1_truedays = np.asarray(w1_truedays)

    optimizer = torch.optim.Adam(model_eta.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    w1_loss = lambda x, y: torch.abs(x - y).mean()
    history = {"train_mse": [], "train_w1": [], "test_w1": []}
    best_w1 = math.inf
    best_state = None

    full_output_eta = predict_field_array(model_eta, test_loader, device=device)
    if varying_days:
        w1_days, w1_max_pos = get_max_time_pos(full_output_eta, len(w1_truemax))
    else:
        w1_days = w1_truedays
        w1_max_pos = get_fixed_days_max_pos(full_output_eta, w1_days)

    for epoch in range(num_epochs):
        model_eta.train()
        optimizer.zero_grad()

        if varying_days:
            w1_input = make_w1_input(lr_fields_for_w1, w1_days, device=device)
        else:
            w1_input = make_w1_input(lr_fields_for_w1, w1_truedays, device=device)

        mse_output = model_eta(mse_input)
        w1_output = model_eta(w1_input).squeeze()
        w1_max_pos_device = w1_max_pos.to(device)
        w1_max = w1_output[tuple(w1_max_pos_device)]

        mse_loss_val = mse_loss(mse_output, mse_target)
        w1_loss_val = w1_loss(w1_max, w1_truemax)
        loss = mse_loss_val + lambd_ * w1_loss_val
        loss.backward()
        optimizer.step()

        full_output_eta = predict_field_array(model_eta, test_loader, device=device)
        sorted_max_values_output = np.sort(np.max(full_output_eta, axis=(1, 2)))
        w1_max_test = torch.tensor(sorted_max_values_output[-len(w1_truemax):], dtype=torch.float32, device=device)
        w1_loss_test = float(w1_loss(w1_max_test, w1_truemax).detach().cpu())
        if keep_best_by_w1 and w1_loss_test < best_w1:
            best_w1 = w1_loss_test
            best_state = copy.deepcopy(model_eta.state_dict())

        history["train_mse"].append(float(mse_loss_val.detach().cpu()))
        history["train_w1"].append(float(w1_loss_val.detach().cpu()))
        history["test_w1"].append(w1_loss_test)

        if (epoch + 1) % omega == 0:
            if varying_days:
                w1_days, w1_max_pos = get_max_time_pos(full_output_eta, len(w1_truemax))
            else:
                w1_max_pos = get_fixed_days_max_pos(full_output_eta, w1_days)

    if keep_best_by_w1 and best_state is not None:
        model_eta.load_state_dict(best_state)
        full_output_eta = predict_field_array(model_eta, test_loader, device=device)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model_eta, history, full_output_eta
