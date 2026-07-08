# Changelog

All notable changes to L-UQ are documented here. The project follows
[semantic versioning](https://semver.org/).

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
