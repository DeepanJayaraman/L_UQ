# Changelog

All notable changes to L-UQ are documented here. The project follows
[semantic versioning](https://semver.org/).

## [2.0.0] — 2026-08-07

Addresses the software and replication issues raised in the *Journal of
Statistical Software* editorial assessment: the dict-based Python API,
the missing replication dependencies and directory-creation crash, the
multi-script replication layout, the absence of an Octave package, and
the awkwardness of scoring a fit against data.

### Added

- **Result objects.** `fit_best` returns an `LMomentFit` with documented
  attributes (`.distribution`, `.parameters`, `.parameters_dict`,
  `.l_moments`, `.rank`, `.used_fallback`, `.skipped`, `.ranking`) and
  methods: `.pdf/.cdf/.sf/.ppf/.rvs/.interval/.frozen()` to evaluate the
  fitted distribution, `.js_div/.kl_div/.ks_stat/.ks_test()` to score it,
  `.summary()` to print it and `.plot()` to draw it. `identify_dist`,
  `identify_dist_bootstrap` and `parameter_identify` likewise return
  `IdentificationResult`, `BootstrapIdentification` and `CandidateFits`.
  All of them keep the 1.x mapping access (`fit["distribution"]`), so
  existing code and replication scripts run unchanged.
- **Kolmogorov-Smirnov statistic**: `ks_stat`/`ks_test` in Python,
  `KSStat.m` in MATLAB/Octave. Binning-free, so it can confirm that a
  Jensen-Shannon comparison is not an artifact of the chosen bins.
- **Octave package** under `octave/`, built from the shared `.m` sources
  by `octave/build_octave_package.py`. Installs with
  `pkg install l-uq-2.0.0.tar.gz`.
- `luq_dist.m`: closed-form PDF/CDF/inverse-CDF for all ten families,
  verified against `scipy.stats` to machine precision (worst discrepancy
  3.4e-16) before transcription.
- `demo_octave.m`: a walkthrough that runs in both MATLAB and Octave with
  no extra packages, exercising twelve toolbox functions rather than two.
- 28 new Python tests in `tests/test_results_api.py`, and a new Part H in
  `tests/octave_verify.m`.

### Verified

- 63 Python tests pass.
- All 55 checks in `tests/octave_verify.m` pass under **GNU Octave
  11.3.0 with no packages loaded**, including the machine-precision
  equivalence against the stored Python reference cases.
- The Octave package installs (`pkg install l-uq-2.0.0.tar.gz`), loads,
  and runs the full pipeline in a session where `norminv`, `normcdf`,
  `gevcdf`, `gppdf`, `wblpdf` and `fitdist` are all undefined.
- Octave and Python agree exactly on the Challenger worked example:
  same family (`gamma`, after skipping `lognormal`), same parameters
  `[5.24692, 3.15713, 53]`, same `P(T <= 31 F)`, same KS statistic
  (0.1929) and JS divergence (0.4220). `luq_percentile` reproduces
  `numpy.percentile` exactly (`[1, 25.75, 50.5, 100]` on `1:100`).
- Bootstrap intervals differ slightly between the two, as expected: the
  RNG streams are not shared across languages.

### Changed

- **No toolbox or package dependency.** `PDF_l`/`CDF_l`/`Random_l` no
  longer call MATLAB's name-dispatching `pdf`/`cdf`/`random`;
  `Parameter_estimation` uses `erfinv` instead of `norminv`;
  `Identify_dist_bootstrap` uses the new `luq_percentile` instead of
  `prctile`; `Identify_dist` no longer uses `vecnorm`. The toolbox now
  runs on a bare MATLAB or Octave installation.
- **`js_div`/`kl_div` accept a fit and a raw sample** in either order,
  binning internally — `js_div(fit, x_full)`, `fit.js_div(x_full)`, and
  `JSDiv(Distribution, Parameter, X)` in MATLAB. The two-vector form is
  unchanged, and the internal binning reproduces the manual binning
  callers previously wrote by hand, bit for bit, so no published number
  moves.
- `parameter_identify` now walks the ranking and returns the top `k`
  *feasible* candidates, recording any it skipped, instead of raising
  when a candidate's estimator domain excludes the sample. Pass
  `strict=True` for the old raising behaviour.
- `luq_percentile` follows numpy's linear-interpolation convention rather
  than `prctile`'s, so MATLAB, Octave and Python report the same
  bootstrap intervals.

### Fixed

- **MATLAB `parameter_identify` was broken.** `Identify_dist` only ever
  returns the single closest family, so any call with `K > 1` raised an
  index-out-of-bounds error, and `K = 1` passed a 1x1 cell rather than a
  char to `Parameter_estimation`. Its output arguments are now
  `[Distribution_type, Parameter, D_sorted, L_sample, skipped]`.
- Replication material: added `replication/requirements.txt` (`pandas`
  and `matplotlib` were undeclared), and fixed the `FIG_DIR.mkdir()`
  call that raised `FileNotFoundError` on a fresh unpack because the
  parent `paper/` directory did not exist.

### Replication material

- The three scripts are replaced by a single `replication/replication.py`,
  as the JSS instructions to authors require, written as a flat sequence
  of numbered sections with no analysis code inside functions so it can
  be stepped through interactively. It reproduces the previous results
  exactly — bit-identical win-rate percentages and L-moment error
  columns, worked examples agreeing to ~15 significant figures. The one
  number that moved is the fatigue B10 life for the L-moment fit
  (105.729 → 105.697 kcycles), now taken from the fitted quantile
  function instead of a 4000-point grid search.
- Section 1 is a guided tour on real data that prints each command and
  its output, for reproduction in the article.
- The superseded scripts are kept under `replication/legacy/`.

### Migration from 1.x

Python code that used the dict API keeps working. To adopt the new API,
replace `fit["distribution"]` with `fit.distribution` and manual binning
with `fit.js_div(x)`. MATLAB code calling `parameter_identify` must be
updated to the new output arguments; MATLAB code relying on `Random_l`
reproducing a particular stream for a given seed will get a different
(equally valid) stream, since variates now come from inverse transform
sampling.

## [1.2.0] — 2026-07-11

### Changed (breaking)
- **Python import namespace renamed `lmoments` → `lmoments_uq`** to
  avoid collision with the unrelated `lmoments` and `lmoments3`
  packages already on PyPI, which also install a top-level `lmoments`
  module. The PyPI distribution name is unchanged (`lmoments-uq`); only
  the import changes: `from lmoments_uq import ...`. Update any code
  that did `from lmoments import ...`. Repository (L-UQ), distribution
  (lmoments-uq), import (lmoments_uq), and article all now align.
- Added Palaniappan Ramu as a second author in the package metadata
  (`pyproject.toml`, `CITATION.cff`), matching the article.

(Contains all 1.1.0 changes below; 1.1.0 was released on GitHub but
not published to PyPI, so 1.2.0 is the first PyPI release carrying the
bootstrap identification and population-truth benchmark.)

## [1.1.0] — 2026-07-11

Adds uncertainty-aware identification and a population-truth benchmark,
in response to peer-review-style feedback on the manuscript.

### Added
- `identify_dist_bootstrap` (Python) / `Identify_dist_bootstrap.m`
  (MATLAB): bootstrap-based, uncertainty-aware distribution
  identification. Resamples the data with replacement, re-identifies on
  each resample, and returns per-family selection frequencies, 95%
  percentile confidence intervals for (t3, t4), and a clear/ambiguous
  status flag. Addresses the fact that at small n the single "closest"
  family is often not statistically distinguishable from the runner-up.
- Bootstrap identification unit tests in all three suites (Python +5,
  MATLAB +3, Octave section G).

### Changed
- The replication benchmark (`replication/run_all.py`) now scores every
  fit against the KNOWN parent distribution (population truth) rather
  than the small sample's own histogram: integrated absolute CDF error,
  extreme-quantile error, and the risk-relevant tail-probability error
  P(X > x_c) at the true 99th percentile, with Jensen-Shannon retained
  only as a secondary diagnostic. This removes histogram-binning
  sensitivity and the circularity of scoring a fit against the noisy
  sample it was estimated from.
- The benchmark additionally records identification accuracy (true
  family ranked first / in top three, split by 2-parameter point vs
  3-parameter curve families, and fallback rate), written to
  `replication/output/identification_accuracy.csv`.

## [1.0.1] — 2026-07-09

Documentation-only release; no code changes.

### Changed
- `python/README.md` (the PyPI project description): replaced the
  outdated "not validated by diffing against MATLAB output" caveat —
  written before a MATLAB installation was available — with the
  current verification status: MATLAB/Python equivalence to 1e-8 on
  fixed reference samples (via `tests/octave_verify.m` under GNU
  Octave 11.3 and `tests/test_uq_matlab.m` under MATLAB R2026a,
  19/19), plus machine-precision agreement with R's `lmom` where the
  closed forms coincide. Test count corrected (30), toolbox name
  updated to L-UQ.

## [1.0.0] — 2026-07-08

First stable release, prepared alongside the Journal of Statistical
Software submission.

### Added
- Python port of the full MATLAB toolbox (`python/lmoments/`):
  `lmom`, `pwm`, `l_moment_ratios`, `identify_dist`,
  `parameter_estimation`, `parameter_identify`, `fit_best`,
  `pdf_l`, `cdf_l`, `random_l`, `kl_div`, `js_div`.
- `fit_best` guarded fit with ranked fallback, in **both** languages
  (`python/lmoments/parameters.py`, `fit_best.m`): walks the
  ratio-diagram ranking and returns the first family whose closed-form
  estimator domain is satisfied, recording skipped families.
- Domain guards in both languages: estimator domain violations raise
  informative errors (`ParameterEstimationError` in Python,
  `LUQ:...` identifiers in MATLAB) instead of returning NaN.
- Explicit three-parameter Weibull support end-to-end in MATLAB
  (`Parameter_estimation`, `PDF_l`, `CDF_l`, `Random_l`); excluded
  from automatic identification by design (ratio-diagram curve overlap).
- Interactive Streamlit application (`python/app.py`).
- Test suites: 30 Python unit tests (`python/tests/`), a mirrored
  MATLAB suite (`tests/test_uq_matlab.m`), and an Octave-runnable
  verification script (`tests/octave_verify.m`, 38 checks) including
  machine-precision equivalence between the MATLAB and Python
  implementations on fixed reference samples.
- GitHub Actions CI (pytest on ubuntu/windows × Python 3.9/3.12;
  MATLAB suite via matlab-actions).

### Fixed
- `CDF_l.m`: gamma branch now applies the same location shift as
  `PDF_l.m` (shifted-gamma PDF/CDF consistency).
- `Identify_dist.m`: `round(x, 4)` rewritten in portable form
  (`round(x*1e4)/1e4`) so the toolbox runs unmodified under GNU Octave.
- Identification tests: normal/gamma acknowledged as a degenerate pair
  (the zero-skew limit of the shifted gamma is the normal), same policy
  as Gumbel/GEV and uniform/GP.

### Changed
- Toolbox renamed **UQ → L-UQ** (repository, paper, error identifiers)
  to be descriptive and avoid collision with the existing UQLab
  framework.
- MATLAB error identifier prefix `UQ:` → `LUQ:`.

### Removed
- `lhsgeneral.m` (third-party utility with unconfirmed licensing);
  the repository is now 100% MIT. The README points to the original
  File Exchange entry.
