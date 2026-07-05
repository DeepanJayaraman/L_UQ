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

Background and validation on statistical distributions and engineering
case studies (sheet-metal forming, speed reducer design, probabilistic
fatigue life) are described in:

> Jayaraman D, Ramu P. L-moments-based uncertainty quantification for
> scarce samples including extremes. *Structural and Multidisciplinary
> Optimization*. 2021 Aug;64(2):505-39.

## Requirements

- MATLAB (developed/tested on R2018b or later)
- Statistics and Machine Learning Toolbox (used for `pdf`, `cdf`,
  `random`, `makedist`, `fitdist`, `icdf`, `lhsdesign`, `corr`)

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
| `lhsgeneral.m` | Latin Hypercube Sampling of correlated random variables (third-party, see below). |
| `demo_example.m` | End-to-end illustrative example (see Quick start). |

### Supported distribution families

`uniform`, `normal`, `exponential`, `gumbel`, `logistic`,
`generalized extreme value`, `generalized pareto`, `lognormal`, `gamma`.

### Known limitations

- **Weibull (`weibul`) is not fully supported.** `Parameter_estimation.m`
  contains a Weibull branch, but `Identify_dist.m` never selects it (its
  candidate-distribution loop only spans the other 9 families), and
  `CDF_l.m` / `Random_l.m` have no Weibull branch at all. `PDF_l.m` calls
  an external `wblpdf3` function that is not included in this repository.
  Treat Weibull support as incomplete until these are added.
- `CDF_l.m`'s `gamma` branch does not apply the location-shift correction
  that `PDF_l.m`'s `gamma` branch applies (`X - P(3)`); this is a known
  inconsistency for shifted Gamma fits.

Issues and pull requests to close these gaps are welcome.

## Third-party code

`lhsgeneral.m` is authored by **Iman Moazzen** (2060 Project, IESVic,
University of Victoria, BC, Canada) and is included here, with
attribution preserved in its header, for reproducibility of the
correlated-sampling step used elsewhere in this line of research. It is
distributed under its original author's terms, not under this
repository's MIT license — see the note at the end of [`LICENSE`](LICENSE).

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
