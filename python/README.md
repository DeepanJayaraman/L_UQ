# L-moments UQ — Python port

A Python port of the [L-UQ MATLAB toolbox](../README.md) one directory
up: distribution-independent uncertainty quantification from scarce
samples, including data with extremes, using L-moments. Same method,
same 9 supported distribution families, plus an interactive Streamlit
UI.

**Validation.** Every parameter-estimation formula and
distribution-parameterization/sign convention is validated against
known `scipy.stats` ground truth — see
[`tests/test_lmoments.py`](tests/test_lmoments.py) (35 tests) and
[`tests/test_results_api.py`](tests/test_results_api.py) (28 tests
covering the result objects and the fit-aware divergence API). In
addition, this port and the MATLAB/Octave implementation have been
verified equivalent: on fixed reference samples, `lmom`,
`Parameter_estimation`, and `CDF_l` reproduce this package's
L-moments, parameter vectors, and fitted CDF values to within 1e-8.
[`tests/octave_verify.m`](../tests/octave_verify.m) covers this in 55
checks, all passing under GNU Octave 11.3.0 with **no packages loaded**;
the MATLAB suite `tests/test_uq_matlab.m` also passes, under R2026a. On
the same samples the package agrees with R's `lmom` to machine
precision wherever the closed forms coincide.

As an end-to-end cross-check, both implementations produce identical
results on the Challenger worked example — same fitted family (`gamma`,
after skipping `lognormal`), same parameters `[5.24692, 3.15713, 53]`,
same `P(T <= 31 F)`, same KS statistic (0.1929) and JS divergence
(0.4220).

## Install

```bash
cd python
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[ui,dev]"
```

## Run the tests

```bash
pytest tests/ -v
```

## Run the illustrative example

```bash
python demo_example.py --no-show   # saves demo_example_output.png
```

## Run the interactive UI

```bash
streamlit run app.py
```

Lets you paste in a sample (or generate a synthetic scarce+extreme one),
see its L-moments and the ranked distribution-family match, pick a family
to fit, and view the PDF/CDF fit against the sample compared with a
conventional-moment (MLE) fit of the same family, plus Jensen-Shannon
divergence numbers quantifying the difference.

## Install

```bash
pip install lmoments-uq          # distribution name (hyphen)
```

The import namespace is `lmoments_uq` (underscore), distinct from the
unrelated `lmoments` / `lmoments3` packages on PyPI.

## API

`fit_best` is the entry point. It identifies the family and estimates its
parameters in one call, and returns an object that knows what to do with
itself:

```python
from lmoments_uq import fit_best

fit = fit_best(x_scarce)

fit.distribution        # 'gamma'
fit.parameters          # array([5.2469, 3.1571, 53.0])
fit.parameters_dict     # {'shape': 5.2469, 'scale': 3.1571, 'loc': 53.0}
fit.l_moments.t3        # sample L-skewness
fit.rank                # 0 if the closest family was fitted
fit.used_fallback       # True when a closer family's estimator was invalid
fit.skipped             # [(family, why it was skipped), ...]

print(fit.summary())    # the whole thing as text, including the ranking
```

It evaluates its own distribution, so there is no need to look up which
`scipy` family and parameter layout the fitted name corresponds to:

```python
fit.cdf(31.0)           # P(X <= 31)
fit.sf(threshold)       # exceedance probability
fit.ppf(0.10)           # the 10th-percentile quantile
fit.rvs(size=1000)
fit.frozen()            # the underlying frozen scipy distribution
```

And it scores itself against data, handling the binning internally:

```python
fit.js_div(x_reference)       # Jensen-Shannon divergence
fit.ks_stat(x_reference)      # KS statistic -- no binning, no tuning knob
fit.plot(x_reference)         # diagnostic PDF/CDF figure
```

The free functions accept a fit in the same way, so an L-moment fit and a
maximum-likelihood fit can be compared on identical terms:

```python
from lmoments_uq import js_div, ks_stat
from scipy import stats

mle = stats.gamma(*stats.gamma.fit(x_scarce))
js_div(fit, x_reference), js_div(mle, x_reference)
ks_stat(fit, x_reference), ks_stat(mle, x_reference)
```

`js_div(p, q)` on two binned mass vectors still works exactly as before.

The identification steps return result objects too:

```python
from lmoments_uq import identify_dist, identify_dist_bootstrap, parameter_identify

ident = identify_dist(x)               # .best, .ranking, .l_moments, .top(3)
boot  = identify_dist_bootstrap(x)     # .status, .selection_frequencies, .t3_ci
cands = parameter_identify(x, k=3)     # .fits -> three full LMomentFit objects

# Rank competing families by how well each reproduces a reference sample
[(f.distribution, f.js_div(x_reference)) for f in cands.fits]
```

All of these still support the mapping access the 1.x API used
(`fit["distribution"]`, `ident["ranking"]`), so existing code keeps
working.

## Differences from the MATLAB version

Since 2.0.0 the two implementations are deliberately close: `JSDiv`,
`KLDiv` and the new `KSStat` all take a fit and a raw sample in MATLAB
too, and `parameter_identify` behaves the same way in both. What remains:

- Python returns result objects with methods; MATLAB returns multiple
  output arguments and structs, which is the idiomatic form there.
- `identify_dist` returns the full ranked distance to **all 9** candidate
  families (`ident.ranking`), not just the single closest match — a
  strict addition, useful for the UI's ranking table.
- The Gamma distribution's location shift is applied consistently in both
  `pdf_l` and `cdf_l` in both implementations (MATLAB's `CDF_l.m`
  historically omitted it; fixed as of the SoftwareX/JSS submission
  preparation).
- `kl_div(p, q)` returns `inf` when `p` has mass in a bin where `q` has
  none (the mathematically correct value); MATLAB's `KLDiv.m` drops such
  bins and can return a finite — even negative — divergence there. This
  cannot affect `js_div`, whose mixture term is positive wherever `p` is,
  and `js_div` was verified to machine precision against
  `scipy.spatial.distance.jensenshannon` (see the divergence tests).
- `weibul` is supported in `parameter_estimation`/`pdf_l`/`cdf_l`/`random_l`
  for completeness, but — matching the MATLAB version — is not one of the
  families `identify_dist` will pick automatically.

## Known limitations (inherited from the method/toolbox)

- Weibull is not offered for automatic identification (see above).
- Gumbel is the k=0 special case of the GEV curve, and Uniform is the k=1
  boundary case of the GP curve, on the L-moment ratio diagram; for
  samples near those special cases, `identify_dist` can pick either the
  special-case family or its generalizing family. This is a property of
  the diagram itself, not a bug (see `tests/test_lmoments.py` for how this
  is handled in testing).
