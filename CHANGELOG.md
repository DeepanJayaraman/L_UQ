# Changelog

All notable changes to L-UQ are documented here. The project follows
[semantic versioning](https://semver.org/).

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
