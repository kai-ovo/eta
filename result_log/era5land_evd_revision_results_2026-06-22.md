# ERA5-Land and EVD Revision Results

This log summarizes the executed ERA5-Land revision outputs from
`notebooks/ERA5Land.ipynb` and `notebooks/ERA5Land-EVD.ipynb`. The notebooks were run in the tested
`eta` conda environment and loaded existing MSE, eta, GEVD-eta, and
misspecified-tail eta checkpoints from `WORK_DIR`.

## Data Setup

- Project root: `/path/to/eta`.
- ERA5-Land file: `data/ERA5/era5land_USA_SouthEast_1999-2023_dailymax.nc`.
- Observable: $g(u)=\max u$ over the HR precipitation grid.
- Original daily fields: 9050.
- Trimmed days above 240 mm: 6 days.
- Retained evaluation fields: $N=9044$.
- HR state shape: $80 \times 160$.
- LR input shape after factor-10 subsampling: $8 \times 16$.
- Supervised training split: 0.5 years, $n=180$ paired LR/HR fields.
- Architecture for all precipitation checkpoints referenced here: `SRCNN(hidden_dim=64, num_blocks=3, scale_factor=10)`, wrapped in `nn.DataParallel`.

The spatial-fidelity metrics are computed over full HR fields. For prediction
$\widehat u_j$ and truth $u_j$, with $n_g=80\cdot160$ grid cells:

$$
{\rm RMSE}_{\rm grid}
= \left[\frac{1}{N n_g}\sum_{j=1}^N \|\widehat u_j-u_j\|_2^2\right]^{1/2}.
$$

For the vanilla ERA5-Land table, the eta row also reports the relative RMSE
increase against the architecture-matched MSE downscaler:

$$
\frac{{\rm RMSE}_{\eta}-{\rm RMSE}_{\rm MSE}}{{\rm RMSE}_{\rm MSE}}.
$$

For each field pair, SSIM is computed on the full 2D HR field using
`skimage.metrics.structural_similarity` with `win_size=7` and
`data_range=max(true_fields)-min(true_fields)`. The table reports

$$
\overline{\rm SSIM}
=
\frac{1}{N}\sum_{j=1}^N {\rm SSIM}(\widehat u_j,u_j),
$$

and the standard deviation of the per-field SSIM values across the same
evaluation set.

## notebooks/ERA5Land.ipynb: Vanilla Precipitation Revision

### Eta Setup

The vanilla precipitation eta run uses the empirical HR max-observable law as
the reference law $\nu_0$. The eta objective is

$$
\mathcal L(\phi)
=
\frac{1}{n}\sum_{i=1}^n \|\phi(x_i)-u_i\|_2^2
+ \lambda W_1((g\circ \phi)_\#\mu,\nu_0),
$$

approximated with tail order statistics for the scalar observable $g(u)=\max u$.

Key settings from the executed notebook:

- MSE checkpoint loaded: `srcnn-mse-0.5yr-10ds.pth`.
- MSE test output shape: `(9044, 80, 160)`.
- MSE test loss printed by notebook: `7.888037443161011`.
- Tail threshold: $t_*=150$ mm.
- Number of tail days/order statistics: 102.
- Eta checkpoint loaded: `srcnn-eta-0.5yr-10ds-150tail-30omega.pth`.
- Eta settings recorded in notebook: seed 43, 150 eta epochs, Adam learning rate $3\times 10^{-4}$, $\lambda=1$, $\omega=30$, varying tail indices enabled.
- Eta test output shape: `(9044, 80, 160)`.
- Eta MSE test loss printed by notebook: `9.756021817525228`.
- Eta W1 test loss printed by notebook: `0.661181628704071`.

### Spatial-Fidelity Table

The displayed strata are based on the true HR maximum $g(u)=\max u$. The
lower-maximum strata use $g(u)\le Q_{\rm true}(q)$, and the upper-tail strata
use $g(u)\ge Q_{\rm true}(q)$.

| subset | threshold_quantile | threshold_value | method | N_samples | RMSE_grid | mean_SSIM | std_SSIM | eta_RMSE_rel_worse_than_MSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full |  |  | MSE downscaler | 9044 | 2.877611 | 0.947003 | 0.043672 |  |
| full |  |  | $\eta$ downscaler | 9044 | 3.111615 | 0.944207 | 0.046011 | 0.081319 |
| q <= 0.7 | 0.700 | 53.260022 | MSE downscaler | 6331 | 1.739108 | 0.965508 | 0.029943 |  |
| q <= 0.7 | 0.700 | 53.260022 | $\eta$ downscaler | 6331 | 1.846915 | 0.963688 | 0.031964 | 0.061990 |
| q <= 0.8 | 0.800 | 65.589246 | MSE downscaler | 7235 | 2.008767 | 0.960048 | 0.033681 |  |
| q <= 0.8 | 0.800 | 65.589246 | $\eta$ downscaler | 7235 | 2.140332 | 0.957853 | 0.035877 | 0.065496 |
| q <= 0.9 | 0.900 | 86.963075 | MSE downscaler | 8139 | 2.344438 | 0.954014 | 0.037959 |  |
| q <= 0.9 | 0.900 | 86.963075 | $\eta$ downscaler | 8139 | 2.506664 | 0.951470 | 0.040278 | 0.069196 |
| q >= 0.95 | 0.950 | 106.220704 | MSE downscaler | 453 | 6.469183 | 0.877379 | 0.040998 |  |
| q >= 0.95 | 0.950 | 106.220704 | $\eta$ downscaler | 453 | 7.130227 | 0.873164 | 0.042823 | 0.102184 |
| q >= 0.975 | 0.975 | 125.503281 | MSE downscaler | 227 | 7.191779 | 0.875916 | 0.040929 |  |
| q >= 0.975 | 0.975 | 125.503281 | $\eta$ downscaler | 227 | 7.932229 | 0.872659 | 0.042427 | 0.102958 |
| q >= 0.99 | 0.990 | 155.473397 | MSE downscaler | 91 | 7.904075 | 0.870943 | 0.038264 |  |
| q >= 0.99 | 0.990 | 155.473397 | $\eta$ downscaler | 91 | 8.631702 | 0.869387 | 0.039693 | 0.092057 |

### ERA5-Land Reading

Across the full evaluation set, the eta downscaler has
${\rm RMSE}_{\rm grid}=3.111615$ compared with `2.877611` for the MSE
downscaler, an 8.13% relative increase. Mean SSIM remains close:
`0.944207` for eta versus `0.947003` for MSE. The eta SSIM standard deviation
is `0.046011`, slightly above the MSE value `0.043672`.

In the lower-maximum strata, the eta RMSE increase is smaller than on the full
set. The relative RMSE increases are 6.20% for `q <= 0.7`, 6.55% for
`q <= 0.8`, and 6.92% for `q <= 0.9`. Mean SSIM remains high in these bulk
regimes, from `0.963688` at `q <= 0.7` to `0.951470` at `q <= 0.9` for eta.
The corresponding MSE mean SSIM values are `0.965508`, `0.960048`, and
`0.954014`.

In the upper-tail strata, both downscalers have larger RMSE because the
evaluation fields are more extreme. The eta relative RMSE increase is 10.22%
for `q >= 0.95`, 10.30% for `q >= 0.975`, and 9.21% for `q >= 0.99`. Mean SSIM
stays near the MSE value in all three tail strata: eta has `0.873164`,
`0.872659`, and `0.869387`, compared with MSE values `0.877379`, `0.875916`,
and `0.870943`.

Overall, the vanilla ERA5-Land revision table shows that eta learning preserves
spatial structure close to the architecture-matched MSE baseline, with a modest
RMSE cost. The RMSE cost is smaller in the lower/bulk subsets and larger in the
upper-tail subsets, while the mean SSIM gap stays small across all strata.

## GEVD Reference Setup

The EVD notebook uses a hypothesized heavier-tailed law for the scalar
observable $Y=g(u)=\max u$. The fitted/truncated GEVD quantile map is

$$
Q_{\rm GEVD}(q)
=
{\rm GEVD}^{-1}\left(q c;\, k,\mu,\sigma\right),
$$

where $c$ is the retained mass below the cutoff. The notebook fits the GEVD to
the retained HR max-observable values, manually increases the fitted scale by 4,
and truncates near the observed upper support.

Key settings from the executed notebook:

- MSE checkpoint loaded: `srcnn-mse-0.5yr-10ds.pth`.
- MSE test output shape: `(9044, 80, 160)`.
- MSE test loss printed by notebook: `8.361367225646973`.
- Truncated GEVD cutoff: 258.0 mm.
- GEVD shape parameter: $k=-0.17868831075604058$.
- GEVD scale used after manual inflation: $\sigma=25.928067249254624$.
- Tail probability outside the retained support: $\gamma=0.004696358286656821$.
- Tail quantile threshold: $\tau=0.95$.
- Corresponding GEVD threshold: $Q_{\rm GEVD}(\tau)=122.83570750509358$ mm.
- Number of eta reference quantiles: 350.
- GEVD eta checkpoint loaded: `srcnn-eta-gevd-0.5yr-10ds-122tail-1omega.pth`.
- GEVD eta settings recorded in notebook: seed 43, 150 eta epochs, Adam learning rate $3\times 10^{-4}$, $\lambda=1$, $\omega=1$, varying tail indices enabled.
- GEVD eta test output shape: `(9044, 80, 160)`.
- GEVD eta MSE test loss printed by notebook: `10.80391534169515`.
- GEVD eta W1 test loss printed by notebook: `0.5844018855814646`.

## Prior-Misspecification Family

The revision block defines perturbed reference quantiles

$$
Q_\alpha(q)
=
Q_{\rm true}(q)
+ \alpha\left(Q_{\rm GEVD}(q)-Q_{\rm true}(q)\right),
$$

for

$$
\alpha \in \{-1,\,-0.5,\,0.5,\,1.5,\,2\}.
$$

The $\alpha=0$ and $\alpha=1$ cases are reference endpoints. Negative
$\alpha$ moves the reference tail below the empirical truth tail, while
$\alpha>1$ extrapolates past the GEVD tail.

The displayed quantile table confirms that the perturbations are modest near
$q=0.95$ and largest near $q=1$.

### Prior Quantiles: First Five Displayed Rows

| q | Q_true | Q_GEVD | Q_alpha=-1 | Q_alpha=-0.5 | Q_alpha=0.5 | Q_alpha=1.5 | Q_alpha=2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.950000 | 106.220702 | 122.835698 | 89.605706 | 97.913204 | 114.528200 | 131.143196 | 139.450694 |
| 0.950143 | 106.497304 | 122.952758 | 90.041849 | 98.269576 | 114.725031 | 131.180485 | 139.408212 |
| 0.950287 | 106.676961 | 123.070124 | 90.283799 | 98.480380 | 114.873543 | 131.266705 | 139.463287 |
| 0.950430 | 106.714097 | 123.187897 | 90.240297 | 98.477197 | 114.950997 | 131.424797 | 139.661696 |
| 0.950573 | 106.728451 | 123.305980 | 90.150922 | 98.439686 | 115.017215 | 131.594744 | 139.883509 |

### Prior Quantiles: Last Five Displayed Rows

| q | Q_true | Q_GEVD | Q_alpha=-1 | Q_alpha=-0.5 | Q_alpha=0.5 | Q_alpha=1.5 | Q_alpha=2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.999427 | 227.794521 | 250.317520 | 205.271522 | 216.533022 | 239.056021 | 261.579020 | 272.840519 |
| 0.999570 | 233.729424 | 252.142654 | 215.316194 | 224.522809 | 242.936039 | 261.349268 | 270.555883 |
| 0.999713 | 234.098937 | 254.029328 | 214.168545 | 224.133741 | 244.064132 | 263.994524 | 273.959719 |
| 0.999857 | 235.321087 | 255.979828 | 214.662345 | 224.991716 | 245.650457 | 266.309199 | 276.638569 |
| 1.000000 | 237.287674 | 258.000000 | 216.575348 | 226.931511 | 247.643837 | 268.356163 | 278.712326 |

## Misspecified-Tail Checkpoints

Each misspecified-tail eta checkpoint was loaded from
`WORK_DIR/models/precip-srcnn/misspec/`. The displayed run used cached
checkpoints for all five alpha values.

| alpha | alpha_tag | model_filename | status |
| --- | --- | --- | --- |
| -1.0 | m1 | `srcnn-eta-misspec-train_model_eta-alpha_m1-0p5yr-10ds-122tail-1omega.pth` | loaded_existing |
| -0.5 | m0p5 | `srcnn-eta-misspec-train_model_eta-alpha_m0p5-0p5yr-10ds-122tail-1omega.pth` | loaded_existing |
| 0.5 | 0p5 | `srcnn-eta-misspec-train_model_eta-alpha_0p5-0p5yr-10ds-122tail-1omega.pth` | loaded_existing |
| 1.5 | 1p5 | `srcnn-eta-misspec-train_model_eta-alpha_1p5-0p5yr-10ds-122tail-1omega.pth` | loaded_existing |
| 2.0 | 2 | `srcnn-eta-misspec-train_model_eta-alpha_2-0p5yr-10ds-122tail-1omega.pth` | loaded_existing |

## Misspecified-Tail Spatial Fidelity Results

| alpha | alpha_tag | N_samples | RMSE_grid | mean_SSIM | std_SSIM | status |
| --- | --- | --- | --- | --- | --- | --- |
| -1.0 | m1 | 9044 | 2.926038 | 0.947280 | 0.043792 | loaded_existing |
| -0.5 | m0p5 | 9044 | 2.967890 | 0.946510 | 0.044193 | loaded_existing |
| 0.5 | 0p5 | 9044 | 3.193944 | 0.944542 | 0.045382 | loaded_existing |
| 1.5 | 1p5 | 9044 | 3.423614 | 0.940689 | 0.048021 | loaded_existing |
| 2.0 | 2 | 9044 | 3.527282 | 0.941217 | 0.047701 | loaded_existing |

## EVD Reading

The misspecified-tail sweep shows a clear RMSE increase as the reference tail is
moved farther into the heavy-tail direction. The lowest RMSE occurs at
$\alpha=-1$, with ${\rm RMSE}_{\rm grid}=2.926038$. The highest RMSE occurs at
$\alpha=2$, with ${\rm RMSE}_{\rm grid}=3.527282$, a 20.55% increase relative
to the $\alpha=-1$ case. The intermediate cases are ordered as expected by tail
strength: $\alpha=-0.5$ is close to $\alpha=-1$, $\alpha=0.5$ is moderately
higher, and $\alpha=1.5$ is close to the largest-error end of the sweep.

Mean SSIM remains high for all five misspecified-tail models. It is highest at
$\alpha=-1$ (`0.947280`) and lowest at $\alpha=1.5$ (`0.940689`). The
$\alpha=2$ model has the largest RMSE but a slightly higher mean SSIM than
$\alpha=1.5$ (`0.941217` versus `0.940689`), indicating that the amplitude error
captured by RMSE and the structural similarity score are not perfectly
monotone in this sweep.

The per-field SSIM spread increases mildly with heavier-tail perturbations.
The standard deviation is `0.043792` at $\alpha=-1$, rises to `0.048021` at
$\alpha=1.5$, and is `0.047701` at $\alpha=2$. This indicates that the
heavier-tail settings introduce somewhat more sample-to-sample variation in
spatial structural fidelity, while the average SSIM remains near 0.94.

Overall, the results support the intended diagnostic: eta learning follows the
prescribed scalar tail law, so changing the reference tail changes the learned
extreme amplitudes. The full-field RMSE is sensitive to this prior
misspecification, while SSIM shows that the generated spatial structures remain
close to the truth in aggregate.

## Execution Notes

The notebooks were executed with:

```sh
conda run -n eta jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/ERA5Land.ipynb
conda run -n eta jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/ERA5Land-EVD.ipynb
```

`skimage.metrics.structural_similarity` was available in the `eta` environment,
so the SSIM columns were evaluated numerically.
