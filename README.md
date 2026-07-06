# Uncertainty quantification using L-moments

A MATLAB toolbox for distribution-independent uncertainty quantification
from **scarce samples, including data with extremes/outliers**. Instead of
assuming a distribution up front and estimating its parameters with
conventional moments (which are highly sensitive to extreme values), this
toolbox uses **L-moments** — robust, linear-combination-of-order-statistics
analogues of conventional moments — to:

1. Identify the most plausible parametric distribution for a sample via an
   **L-moment ratio diagram** (L-skewness vs. L-kurtosis).
2. Estimate that distribution's parameters directly from the sample
   L-moments.
3. Evaluate the fitted PDF/CDF, generate random variates from it, and
   quantify divergence between distributions.

A Python port with an interactive Streamlit UI is available in
[`python/`](python/) — same method and 9 supported families, plus a
ranked-identification view and side-by-side comparison against a
conventional-moment (MLE) fit. See [`python/README.md`](python/README.md)
for details, including how its outputs were validated in the absence of a
MATLAB installation.

Background and validation on statistical distributions and engineering
case studies (sheet-metal forming, speed reducer design, probabilistic
fatigue life) are described in:

> Jayaraman D, Ramu P. L-moments-based uncertainty quantification for
> scarce samples including extremes. *Structural and Multidisciplinary
> Optimization*. 2021 Aug;64(2):505-39.

## Requirements

- MATLAB (developed/tested on R2018b or later)
- Statistics and Machine Learning Toolbox (used for `pdf`, `cdf`,
  `random`, `fitdist`)

No installation is required beyond adding this folder to your MATLAB path:

```matlab
addpath('path/to/UQ')
```

## Quick start

```matlab
X = random('lognormal', 0, 0.5, 12, 1);   % a scarce sample
X(end+1) = 8*max(X);                       % ... with one extreme value

[Distribution_type, L_sample] = Identify_dist(X);
L1 = L_sample(1); L2 = L_sample(2); T3 = L_sample(3); T4 = L_sample(4);

Parameter = Parameter_estimation(X, Distribution_type{1}, L1, L2, T3, T4);

pdf_vals = PDF_l(linspace(0, max(X), 200), Distribution_type{1}, Parameter);
cdf_vals = CDF_l(linspace(0, max(X), 200), Distribution_type{1}, Parameter);
```

See [`demo_example.m`](demo_example.m) for a complete, runnable example
that also compares the L-moment fit against a conventional-moment (MLE)
fit on the same scarce, extreme-containing sample, and plots both.

## Function reference

| File | Purpose |
|---|---|
| `lmom.m` | Compute the first `nL` sample L-moments of a data vector (uses `LegendreShiftPoly.m` internally). |
| `LegendreShiftPoly.m` | Shifted Legendre polynomial coefficients, used by `lmom.m`. |
| `Identify_dist.m` | Identify the best-fit distribution family for a sample from its L-skewness/L-kurtosis position on the L-moment ratio diagram. |
| `Parameter_estimation.m` | Estimate a named distribution's parameters from sample L-moments (`L1`, `L2`, `T3`, `T4`). |
| `parameter_identify.m` | Convenience wrapper combining `Identify_dist.m` and `Parameter_estimation.m` for the top `K` candidate distributions. |
| `PDF_l.m` / `CDF_l.m` | Evaluate the PDF/CDF of a named distribution at given points and parameters. |
| `Random_l.m` | Generate random variates from a named distribution and parameter set. |
| `KLDiv.m` / `JSDiv.m` | Kullback-Leibler / Jensen-Shannon divergence between two (binned) probability distributions, used to compare fit quality. |
| `demo_example.m` | End-to-end illustrative example (see Quick start). |

### Supported distribution families

`uniform`, `normal`, `exponential`, `gumbel`, `logistic`,
`generalized extreme value`, `generalized pareto`, `lognormal`, `gamma`.

### Weibull: supported explicitly, excluded from auto-identification

The three-parameter Weibull (`weibul`) is fully supported when requested
by name — `Parameter_estimation.m`, `PDF_l.m`, `CDF_l.m`, and
`Random_l.m` all handle it. It is deliberately **not** among the families
`Identify_dist.m` selects automatically: its L-moment ratio curve passes
through or near other families' loci (shape k=1 *is* the exponential
point; near k≈3.6 it sits essentially on the normal point), so including
it in the automatic search makes identification ambiguous rather than
better. This mirrors the Python port's behavior.

Issues and pull requests are welcome.

## Correlated Latin Hypercube sampling (external dependency)

Earlier versions of this repository bundled `lhsgeneral.m` (correlated
Latin Hypercube sampling) by **Iman Moazzen** (2060 Project, IESVic,
University of Victoria, BC, Canada). It has been removed because its
redistribution license could not be confirmed, and it is not part of
this toolbox's core identify/estimate/evaluate pipeline. If your
workflow needs the correlated-sampling step described in the companion
papers, obtain it directly from the original author's MATLAB File
Exchange entry:
<https://www.mathworks.com/matlabcentral/fileexchange/56384-lhsgeneral-pd-correlation-n>.
Everything remaining in this repository is under the MIT license.

## Citing this software

If you use this toolbox in your research, please cite:

> Jayaraman D, Ramu P. L-moments-based uncertainty quantification for
> scarce samples including extremes. Structural and Multidisciplinary
> Optimization. 2021 Aug;64(2):505-39. https://doi.org/10.1007/s00158-021-02930-2

## License

MIT — see [LICENSE](LICENSE) (with a carve-out for the third-party file
noted above).

## Support

Questions and issues: deepanjayram@gmail.com or via the
[GitHub issue tracker](https://github.com/DeepanJayaraman/UQ/issues).
