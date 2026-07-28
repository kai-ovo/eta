# Computational Overhead Revision Results

This log summarizes the executed results in `notebooks/ERA5Land-Computational-Overhead.ipynb`.
The notebook was run from the repository root with

```sh
conda run -n eta jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/ERA5Land-Computational-Overhead.ipynb
```

No models, figures, samples, or tables were saved by the notebook. The final save-guard table reports: `No save calls attempted.`

## Estimation Setup

The notebook uses bounded timing probes and estimates paper-scale runtime by proportional work scaling:

$$
T_{\mathrm{full}} \approx T_{\mathrm{probe}}
\frac{W_{\mathrm{full}}}{W_{\mathrm{probe}}}.
$$

For toy examples, $W$ is the number of training iterations. For SRCNN downscaling,

$$
W_{\mathrm{MSE}} = E(n_{\mathrm{train}} + n_{\mathrm{aux}}),
$$

and for eta continuation with IICT,

$$
W_{\eta} = n_{\mathrm{aux}} + E(n_{\mathrm{train}} + n_q + n_{\mathrm{aux}}).
$$

For Flow Matching training, $W = E N$ in optimizer sample-passes. For Flow Matching sampling,

$$
W_{\mathrm{sample}} = N_{\mathrm{samples}} N_{\mathrm{ODE}},
$$

using generated-sample ODE-step passes.

## Tested Hardware

| item | value |
|---|---:|
| Device used | `cuda` |
| GPU model | Tesla V100-SXM2-32GB-LS |
| GPU total VRAM | 31.73 GB |
| System RAM | 503.76 GB |
| PyTorch | 2.2.0+cu118 |
| CUDA runtime | 11.8 |
| Python | 3.11.8 |
| OS | Linux 4.18.0-477.10.1.el8_8.x86_64 |

## Data And Hyperparameter Setup

| quantity | value |
|---|---:|
| Toy supervised training size | 100 |
| Toy auxiliary grid points | 1,000,000 |
| ERA5-Land trimmed fields | 9,044 |
| ERA5-Land HR shape | $80 \times 160$ |
| ERA5-Land LR shape | $8 \times 16$ |
| ERA5-Land supervised training fields | 180 |
| ERA5-Land auxiliary probe fields | 768 |

Toy settings follow the toy notebooks: FCNN with three width-256 hidden layers, `psilu`, 3000 MSE iterations, 1000 eta-pretraining iterations, 3000 eta-continuation iterations, $\lambda=1$, and $\omega=100$.

ERA5-Land downscaling uses `SRCNN(hidden_dim=64, num_blocks=3, scale_factor=10)`, batch size 64, Adam with learning rate $3\times10^{-4}$, 500 MSE epochs, and 150 eta epochs. The vanilla eta reference uses 102 empirical tail days above 150 mm with $\lambda=1$ and $\omega=30$.

Flow Matching uses `UNet(image_channels=1, n_channels=16, ch_mults=(1,2,2,4), is_attn=(False,False,True,True), n_blocks=1)`, Adam with learning rate $3\times10^{-4}$. The probes cover HR FM training on 0.5 years for 200 epochs, LR FM training on 25 years for 80 epochs, 9044 generated samples, 200 ODE steps for HR sampling, and 100 ODE steps for LR sampling.

The GEVD eta run uses $\tau=0.95$, $n_q=350$, 150 epochs, $\lambda=1$, and $\omega=1$. The fitted/truncated GEVD parameters were

$$
\kappa=-0.178688,\qquad \zeta=25.077246,\qquad
\sigma=25.928067,\qquad c=0.995304,
$$

with cutoff 258.0.

## Numerical Results

| experiment | component | probe | estimated full time | peak allocated VRAM (GB) | peak reserved VRAM (GB) | process RSS after probe (GB) |
|---|---|---|---:|---:|---:|---:|
| Toy 2D-to-1D | MSE baseline training | 10 iterations with full toy supervised set | 0:25:52.13 | 0.0273 | 0.0430 | 3.9831 |
| Toy 2D-to-1D | eta pretraining | 10 pretrain iterations | 0:00:18.03 | 0.0291 | 0.0469 | 3.9912 |
| Toy 2D-to-1D | eta continuation | 10 eta iterations after a one-step warm-start pretrain | 0:03:35.49 | 0.5232 | 0.5625 | 4.1084 |
| Toy 2D-to-2D | MSE baseline training | 10 iterations with full toy supervised set | 0:00:43.51 | 0.0273 | 0.0430 | 4.1084 |
| Toy 2D-to-2D | eta pretraining | 10 pretrain iterations | 0:00:16.07 | 0.0295 | 0.0469 | 4.1085 |
| Toy 2D-to-2D | eta continuation | 10 eta iterations after a one-step warm-start pretrain | 0:03:02.64 | 0.5390 | 0.6836 | 4.1094 |
| ERA5-Land downscaling | MSE baseline training | 1 epoch over 180 supervised fields plus 768 auxiliary eval fields | 0:54:47.54 | 0.3413 | 0.5664 | 4.5006 |
| ERA5-Land downscaling | eta continuation with IICT | 1 eta epoch with 768 auxiliary fields and 102 W1 fields | 0:10:24.33 | 0.8985 | 1.4902 | 4.6464 |
| Flow Matching | HR FM training | 1 optimizer step, probe batch 32, HR shape $80\times160$ | 0:14:41.94 | 0.3078 | 0.3906 | 5.1148 |
| Flow Matching | LR FM training | 1 optimizer step, probe batch 128, LR shape $8\times16$ | 0:32:24.95 | 0.2575 | 0.3613 | 5.1418 |
| Flow Matching | HR FM sampling | 32 samples, 1 ODE step, batch 32 | 10:14:36.18 | 0.2804 | 0.3477 | 5.1443 |
| Flow Matching | LR FM sampling | 512 samples, 5 ODE steps, batch 512 | 0:19:15.72 | 0.2606 | 0.3457 | 5.1443 |
| Flow Matching | eta pass-through of LR samples | 128 synthetic LR samples through SRCNN eta map | 0:00:04.54 | 0.2779 | 0.3164 | 5.1444 |
| GEVD prior downscaling | GEVD eta continuation | 1 eta epoch with 768 auxiliary fields and 350 GEVD W1 fields | 0:09:33.09 | 1.7621 | 2.3867 | 5.2354 |

All rows completed with status `ok`.

## Main Takeaways

For the vanilla ERA5-Land SRCNN experiment, the estimated MSE baseline training time is about 55 minutes, while the eta continuation is about 10 minutes under the proportional field-pass model. The eta continuation uses more peak GPU memory because it evaluates IICT tail fields and auxiliary outputs for $g(u)=\max(u)$.

For Flow Matching, sampling dominates the measured overhead: HR Dormand-Prince sampling is estimated at about 10.2 hours for 9044 samples and 200 ODE steps, while LR sampling is about 19 minutes for 9044 samples and 100 ODE steps. The eta pass-through of generated LR samples is negligible by comparison, about 4.5 seconds.

For GEVD prior downscaling, the eta continuation estimate is about 9.6 minutes. Its peak allocated VRAM is the largest measured SRCNN eta row, about 1.76 GB allocated and 2.39 GB reserved, because it uses 350 GEVD reference quantiles with $\omega=1$.

These are approximate timings from small probes. They exclude checkpoint and sample-write overhead by design, and peak VRAM reflects the probe batch sizes shown in the table.
