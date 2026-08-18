# LMomFit: Distribution selection/uncertainty quantification using L-moments

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21268119.svg)](https://doi.org/10.5281/zenodo.21268119)
[![PyPI](https://img.shields.io/pypi/v/lmoments-uq)](https://pypi.org/project/lmoments-uq/)
[![View on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/184220-l-uq)

**LMomFit** is a MATLAB toolbox for distribution-independent uncertainty quantification
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

- MATLAB (developed/tested on R2018b or later) **or** GNU Octave 6.1 or later
- No toolboxes or packages. Since version 2.0.0 the toolbox depends only
  on functions core to both MATLAB and Octave; the Statistics and
  Machine Learning Toolbox is no longer required. (The optional
  `demo_example.m` is the one exception — it compares against `fitdist`,
  so it needs that toolbox. `demo_octave.m` is the portable equivalent.)

### MATLAB

No installation is required beyond adding this folder to your path:

```matlab
addpath('path/to/L-UQ')
```

### GNU Octave

Install the package from the tarball in [`octave/`](octave/):

```
pkg install octave/l-uq-2.0.0.tar.gz
pkg load l-uq
```

Rebuild the tarball after changing any `.m` source with
`python octave/build_octave_package.py`; it assembles the package from
the sources in this directory, so the MATLAB and Octave versions cannot
drift apart.

Verified on GNU Octave 11.3.0: `tests/octave_verify.m` runs 55 checks
with no packages loaded, and all pass. On Windows, `pkg install` needs
Octave's own `usr\bin` on the PATH ahead of `C:\Windows\System32`,
otherwise `pkg` picks up the system `tar.exe`, which cannot read the
MSYS-style paths Octave passes it.

## Quick start

```matlab
% A scarce sample with one extreme value
X = Random_l('lognormal', [0, 0.5, 0], 12, 1);
X(end+1) = 8*max(X);

% Identify the family and estimate its parameters, skipping any family
% whose closed-form estimator is undefined for this sample
[Distribution, Parameter, skipped] = fit_best(X);

% Evaluate the fit
grid     = linspace(0, max(X), 200);
pdf_vals = PDF_l(grid, Distribution, Parameter);
cdf_vals = CDF_l(grid, Distribution, Parameter);

% Score it against a larger reference sample -- no manual binning needed
X_reference = Random_l('lognormal', [0, 0.5, 0], 20000, 1);
JSDiv(Distribution, Parameter, X_reference)
KSStat(Distribution, Parameter, X_reference)
```

See [`demo_octave.m`](demo_octave.m) for a complete, runnable walkthrough
of the whole API that works in both MATLAB and Octave, or
[`demo_example.m`](demo_example.m) for the MATLAB-only version that also
compares against a conventional-moment (MLE) `fitdist` fit and plots both.

## Function reference

| File | Purpose |
|---|---|
| `lmom.m` | Compute the first `nL` sample L-moments of a data vector (uses `LegendreShiftPoly.m` internally). |
| `LegendreShiftPoly.m` | Shifted Legendre polynomial coefficients, used by `lmom.m`. |
| `Identify_dist.m` | Identify the best-fit distribution family for a sample from its L-skewness/L-kurtosis position on the L-moment ratio diagram. |
| `Parameter_estimation.m` | Estimate a named distribution's parameters from sample L-moments (`L1`, `L2`, `T3`, `T4`). |
| `parameter_identify.m` | Fit the top `K` feasible candidate distributions, so families that sit close together on the ratio diagram can be compared rather than collapsed to the nearest one. |
| `fit_best.m` | Identify and fit in one call, walking the ranking past any family whose closed-form estimator is undefined for the sample and reporting what it skipped. Preferred entry point. |
| `Identify_dist_bootstrap.m` | Uncertainty-aware identification: how often each family would be chosen across bootstrap resamples, with percentile intervals for `(t3, t4)` and an ambiguity flag. |
| `PDF_l.m` / `CDF_l.m` | Evaluate the PDF/CDF of a named distribution at given points and parameters. |
| `Random_l.m` | Generate random variates from a named distribution and parameter set. |
| `KLDiv.m` / `JSDiv.m` | Kullback-Leibler / Jensen-Shannon divergence. Takes either two binned mass vectors, or a fit and a raw sample — `JSDiv(Distribution, Parameter, X)` — binning internally. |
| `KSStat.m` | Kolmogorov-Smirnov statistic between a fit and a raw sample. Needs no binning, so it can confirm a divergence comparison is not a bin-width artifact. |
| `luq_dist.m` | Closed-form PDF/CDF/inverse-CDF for all ten families. The reason no Statistics toolbox is needed. |
| `luq_bin_fit.m` / `luq_percentile.m` | Internal helpers: shared binning rule for the divergences, and a `prctile` replacement. |
| `demo_octave.m` | End-to-end walkthrough that runs in both MATLAB and Octave (see Quick start). |
| `demo_example.m` | The MATLAB-only demo, which additionally compares against `fitdist`. |

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
[GitHub issue tracker](https://github.com/DeepanJayaraman/L-UQ/issues).
