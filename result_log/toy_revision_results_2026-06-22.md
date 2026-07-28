# Toy Revision Results

This log summarizes the already-executed revision result blocks embedded in
`notebooks/toy--2D->1D.ipynb` and `notebooks/toy--2D->2D.ipynb`. No notebook cells were executed for
this summary; the values below were read from the stored notebook outputs.

## Common Toy Setup

Both toy notebooks use the same input distribution and scarce supervised split:

- Project root: `/path/to/eta`.
- Toy data path pattern: `data/toy/var10/N100.pth`.
- Input law: $X\sim N(0,10I_2)$.
- Supervised training size: $n=100$.
- Main eta architecture in the revision reruns: `FCNN` with three hidden layers of width 256 when defaults are used.
- Activation: `psilu`, i.e. $x\mapsto x\sigma(4x)$, when notebook defaults are used.
- Optimizer: Adam.
- MSE pretraining iterations per revision realization: 1000.
- Eta iterations per revision realization: 3000.
- Target number of eta realizations: $K=20$.

For each eta realization $k$, the diagnostics evaluate predictions on the
existing contour grid $\mathcal G_E=\mathcal G$. The predicted extreme location
and amplitude are

$$
\widehat z_k=\arg\max_{z\in\mathcal G_E} y_\eta^{(k)}(z),
\qquad
\widehat a_k=\max_{z\in\mathcal G_E} y_\eta^{(k)}(z).
$$

The true grid-level extreme location and amplitude are

$$
z_\star=\arg\max_{z\in\mathcal G_E} y(z),
\qquad
a_\star=\max_{z\in\mathcal G_E} y(z).
$$

The reported spatial uncertainty summaries are

$$
\bar z=\frac1K\sum_{k=1}^K \widehat z_k,
\qquad
V_{\rm loc}=\frac1K\sum_{k=1}^K \|\widehat z_k-\bar z\|_2^2,
$$

$$
E_{\rm loc}=\frac1K\sum_{k=1}^K \|\widehat z_k-z_\star\|_2,
\qquad
E_{\rm amp}=\frac1K\sum_{k=1}^K
\frac{|\widehat a_k-a_\star|}{|a_\star|}.
$$

The tail quantile discrepancy is reported as

$$
W_{1,Q}
\approx
\frac1{|Q|}\sum_{q\in Q}
\left|
Q_{(y_\eta^{(k)})_\#\mu}(q)-Q_{y_\#\mu}(q)
\right|.
$$

The exceedance-probability map shown in both notebooks is

$$
\widehat p_\eta(z;t)
=
\frac1K\sum_{k=1}^K
{\bf 1}\{y_\eta^{(k)}(z)>t\}.
$$

## 2D-to-1D Toy Scalar Map

### Revision Setup

Notebook: `notebooks/toy--2D->1D.ipynb`.

The scalar target is the toy observable map $y:X\to\mathbb R$. The revision
section quantifies how much the extreme location varies across eta estimators
that all receive the same scalar tail-distribution calibration.

Stored execution details:

- Revision markdown block is present.
- Summary result table is embedded in code cell 24, execution count 14.
- Exceedance-probability and predicted-location figure is embedded in code cell 26, execution count 15.
- `RUN_MISSING_ETA_REALIZATIONS = True` in the executed revision cell.
- The stored stdout shows missing revision seeds were trained during that earlier execution.
- Seeds used: `[22, 25, 27, 28, 29, 32, 33, 38, 39, 40, 41, 43, 51, 54, 57, 58, 63, 64, 67, 83]`.

### Numerical Results

| quantity | value |
| --- | ---: |
| $K$ | 20 |
| target $K$ | 20 |
| threshold $t$ | 0.433047 |
| $z_{\star,1}$ | 2.020202 |
| $z_{\star,2}$ | -2.020202 |
| $\bar z_1$ | -0.113131 |
| $\bar z_2$ | -0.460606 |
| $V_{\rm loc}$ | 15.935353 |
| $E_{\rm loc}$ | 4.245257 |
| $a_\star$ | 0.537137 |
| mean $\widehat a_k$ | 0.535940 |
| std. $\widehat a_k$ | 0.002714 |
| $E_{\rm amp}$ | 0.003821 |
| mean $W_{1,Q}$ | 0.001698 |
| std. $W_{1,Q}$ | 0.000580 |

### Reading

The eta realizations match the extreme amplitude and tail quantiles very closely:
the mean predicted maximum is 0.535940 versus $a_\star=0.537137$, the relative
amplitude error is 0.003821, and the mean tail quantile discrepancy is 0.001698.
At the same time, the location diagnostics show substantial residual spatial
uncertainty: the mean predicted extreme location $\bar z=(-0.113131,-0.460606)$
is far from the true grid extreme $z_\star=(2.020202,-2.020202)$, with
$E_{\rm loc}=4.245257$ and $V_{\rm loc}=15.935353$.

This supports the intended point of the toy revision: scalar tail calibration
can identify plausible extreme amplitudes and distributions without uniquely
identifying the pointwise location of an unobserved extreme event.

## 2D-to-2D Toy State Map

### Revision Setup

Notebook: `notebooks/toy--2D->2D.ipynb`.

The state target is $u=(u_1,u_2)$, where $u_1$ is the 2D-to-1D toy map and

$$
u_2(x)=-0.1\sin\left(\frac{\pi x_1}{3}\right)
\,\sin\left(\frac{\pi x_2}{4}\right).
$$

The scalar observable used for eta calibration and revision diagnostics is

$$
y(z)=g(u(z))=2|u_1(z)|+\frac12|u_2(z)|.
$$

The component-wise rRMSE diagnostic is

$$
{\rm rRMSE}_{u_j}
=
\frac{
\left[\sum_{z\in\mathcal G}(u_{j,\eta}(z)-u_j(z))^2\right]^{1/2}
}{
\left[\sum_{z\in\mathcal G}u_j(z)^2\right]^{1/2}
},
\qquad j=1,2.
$$

The tail-neighborhood component rRMSE uses the subset

$$
\{z:y(z)>t\},
$$

where $t$ is the empirical 0.99 observable quantile.

Stored execution details:

- Revision markdown block is present.
- Summary result table is embedded in code cell 22, execution count 11.
- Exceedance-probability and predicted-location figure is embedded in code cell 24, execution count 12.
- `RUN_MISSING_ETA_REALIZATIONS = True` in the executed revision cell.
- The stored stdout shows missing revision seeds were trained during that earlier execution.
- Seeds used: `[27, 28, 50, 63, 64, 83, 22, 25, 29, 32, 33, 38, 39, 40, 41, 43, 51, 54, 57, 58]`.

### Summary Results

The embedded pandas display for this wide summary table elides one middle column
with `...`; the notebook code computes $E_{\rm loc}$, but its exact stored value
is not visible in the rendered output. The visible stored values are:

| quantity | value |
| --- | ---: |
| $K$ | 20 |
| target $K$ | 20 |
| threshold $t$ | 0.886123 |
| tail-neighborhood grid cells | 36 |
| $z_{\star,1}$ | 2.020202 |
| $z_{\star,2}$ | -2.020202 |
| $\bar z_1$ | 0.759596 |
| $\bar z_2$ | -0.517172 |
| $V_{\rm loc}$ | 38.768810 |
| $a_\star$ | 1.117031 |
| mean $\widehat a_k$ | 1.100402 |
| std. $\widehat a_k$ | 0.030055 |
| $E_{\rm amp}$ | 0.016095 |
| mean $W_{1,Q}$ | 0.002406 |
| std. $W_{1,Q}$ | 0.000707 |
| mean ${\rm rRMSE}_{u_1}$ on full grid | 1.294439 |
| mean ${\rm rRMSE}_{u_2}$ on full grid | 1.885438 |
| mean ${\rm rRMSE}_{u_1}$ on tail neighborhood | 0.728659 |
| mean ${\rm rRMSE}_{u_2}$ on tail neighborhood | 0.865856 |

### Reading

As in the scalar toy, the observable tail is closely calibrated while pointwise
location and component allocation remain non-unique. The mean predicted
observable maximum is 1.100402 compared with $a_\star=1.117031$, giving
$E_{\rm amp}=0.016095$, and the mean tail quantile discrepancy is 0.002406.
However, the location spread is larger than in the 2D-to-1D toy, with
$V_{\rm loc}=38.768810$.

The component rRMSE values show that matching the scalar observable does not
force component-wise recovery of the state map. Full-grid component errors are
large on average, especially for $u_2$, while tail-neighborhood errors are lower
for both components. This is consistent with the eta-learning interpretation:
the regularizer constrains the distribution of the scalar observable
$g(u_\eta)$, not the full pointwise state decomposition.
