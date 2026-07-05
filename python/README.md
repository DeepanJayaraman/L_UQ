# L-moments UQ — Python port

A Python port of the [MATLAB UQ toolbox](../README.md) one directory up:
distribution-independent uncertainty quantification from scarce samples,
including data with extremes, using L-moments. Same method, same 9
supported distribution families, plus an interactive Streamlit UI.

No MATLAB installation was available while writing this port, so it was
**not** validated by diffing against MATLAB output. Instead, every
parameter-estimation formula and distribution-parameterization/sign
convention was validated by generating large synthetic samples from known
`scipy.stats` parameters and checking that this code recovers them — see
[`tests/test_lmoments.py`](tests/test_lmoments.py) (22 tests, all passing).
If you have MATLAB available, cross-checking a few outputs against the
original `.m` files would be a good next step.

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

## API

```python
from lmoments import identify_dist, parameter_estimation, pdf_l, cdf_l, random_l, js_div

result = identify_dist(x)              # {'best', 'ranking', 'L_sample'}
params = parameter_estimation(x, result['best'], *result['L_sample'])
pdf_vals = pdf_l(xgrid, result['best'], params)
```

## Differences from the MATLAB version

- `identify_dist` returns the full ranked distance to **all 9** candidate
  families (`result['ranking']`), not just the single closest match — a
  strict addition, useful for the UI's ranking table.
- `parameter_identify(x, k)` genuinely fits the top-`k` candidates; the
  MATLAB version's `K` argument silently breaks for `K>1` because
  `Identify_dist.m` never actually returns more than one candidate.
- The Gamma distribution's location shift is applied consistently in both
  `pdf_l` and `cdf_l` here; MATLAB's `CDF_l.m` omits it (see the main
  README's "Known limitations").
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
