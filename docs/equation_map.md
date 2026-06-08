# Equation and Diagnostic Implementation Map

This map links the paper notation and revision diagnostics to the implementation. Existing notebook cells are referenced by notebook section names because the repository is notebook-oriented and cell numbers may shift when revision cells are appended.

| paper equation / algorithm / diagnostic | mathematical object | implementation location | function/cell/notebook | notes |
|---|---|---|---|---|
| η-learning scalar objective | MSE on `(x_i, y_i)` plus `lambda * W1(phi_# mu, nu_0)` | `toy--2D->1D.ipynb`, `train_utils.py` | `train_eta_xy` | Uses supervised MSE and quantile W1 on scalar outputs. |
| η-learning state objective | MSE on `(x_i, u_i)` plus `lambda * W1((g o phi)_# mu, nu_0)` | `toy--2D->2D.ipynb`, `ERA5Land.ipynb`, `ERA5Land-EVD.ipynb`, `train_utils.py` | `train_eta_xuy`; notebook `train_model_eta` cells | State-map versions apply the regularizer to `g(u)`, not to the full state law. |
| Quantile approximation of one-dimensional `W1` | `mean_q |Q_model(q) - Q_ref(q)|` | `metric.py`, `revision_utils.py`, notebooks | `wloss`, `tail_w1_quantile` | Training and revision diagnostics keep the regularizer one-dimensional. |
| Tail-biased quantile grid | `Q subset [tau, 1]`, often dense near the upper tail | `toy--2D->1D.ipynb`, `toy--2D->2D.ipynb`, `ERA5Land-EVD.ipynb` | `quantiles` cells | Toy notebooks use multi-scale grids near one; GEVD uses `torch.linspace(tau, 1, 350)`. |
| IICT / quantile-index refresh | inference indices `I` and max-component indices `J`, refreshed every `omega` | `ERA5Land.ipynb`, `ERA5Land-EVD.ipynb`, `revision_utils.py` | `get_max_time_pos`, `get_fixed_days_max_pos`, `train_precip_eta_no_save` | Preserves the memory/stability pattern for max-like observables. |
| Observable `g(u)=max(u)` | HR spatial maximum precipitation | `ERA5Land.ipynb`, `ERA5Land-DGM-Plot.ipynb`, `ERA5Land-EVD.ipynb` | `np.max(..., axis=(1,2))` cells | Used for precipitation tail PDFs, quantiles, and W1 losses. |
| Toy scalar observable | `y(x)` | `toy--2D->1D.ipynb` | loaded `Y`, `Y_GRID`; `eval_result` | Direct scalar-map learning. |
| Toy state observable | `g(u)=2|u_1|+|u_2|/2` | `toy--2D->2D.ipynb` | `def y(u)` | Used for state-map eta regularization and revision spatial uncertainty. |
| Conditional mean | `m(u_0;t,n_g)` | `revision_utils.py`, `ERA5Land-EVD.ipynb` revision section | `conditional_mean_values`, `mean_wasserstein_stat_error` | Returns NaN for empty exceedance sets and uses `nanmean` behavior through finite filtering. |
| Weighted coverage | `c(u_0;t,n_g)` | `ERA5Land.ipynb`, `revision_utils.py`, `ERA5Land-EVD.ipynb` revision section | existing `get_coverage_stats`; `weighted_coverage_values` | Existing notebook plots coverage PDFs; revision GEVD sensitivity reports distributional W1 errors. |
| Full-grid RMSE | `RMSE_grid` | `revision_utils.py`, `ERA5Land.ipynb` revision section | `precip_spatial_metrics`, `precipitation_metrics_table` | Reported for vanilla precipitation MSE, eta, and trivial LR-upsample baseline. |
| Full-grid rRMSE | `rRMSE_grid` | `revision_utils.py`, `ERA5Land.ipynb`, `ERA5Land-EVD.ipynb` revision sections | `precip_spatial_metrics` | Used for vanilla spatial fidelity and prior-misspecification sensitivity. |
| Mean SSIM | average `SSIM(u_hat_j, u_j)` | `revision_utils.py`, `ERA5Land.ipynb`, `ERA5Land-EVD.ipynb` revision sections | `mean_structural_similarity` | Uses `skimage.metrics.structural_similarity` when available; otherwise reports NaN. |
| Mean spatial correlation | average flattened-field Pearson correlation | `revision_utils.py`, `ERA5Land.ipynb` revision section | `mean_spatial_corr` | Handles zero-variance fields with NaN-safe averaging. |
| Toy spatial-uncertainty diagnostics | `z_hat_k`, `a_hat_k`, `V_loc`, `E_loc`, `E_amp`, exceedance probability | `toy--2D->1D.ipynb`, `toy--2D->2D.ipynb` revision sections | appended revision cells | Uses existing contour grid and available eta realizations; parameterized for `K=20`. |
| Toy component-wise allocation diagnostics | `rRMSE_{u_1}`, `rRMSE_{u_2}` | `toy--2D->2D.ipynb` revision section | appended revision cells | Reports full-grid and true-tail-neighborhood component rRMSE. |
| Prior-misspecification diagnostics | `Q_alpha`, `d_nu(alpha)`, `D_tail(alpha)`, physical-fidelity metrics | `ERA5Land-EVD.ipynb` revision section | appended revision cells | Trains/evaluates separate in-memory eta maps by alpha; no checkpoints saved. |
| Computational-overhead measurements | wall time, peak VRAM, overhead ratios | `ERA5Land-Computational-Overhead.ipynb`, `revision_utils.py` | `measure_wall_time`, `reset_peak_memory`, `get_peak_vram`, notebook timing cells | Covers precipitation experiments only and separates FM training, sampling, and eta pass-through. |
| No-save timing guard | blocked checkpoint/sample writes during timing | `revision_utils.py` | `disable_checkpoint_saving`, `run_without_saving` | Available for timing cells that reuse save-oriented control flow. |
