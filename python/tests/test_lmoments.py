"""Validation tests for the Python L-moments port.

No MATLAB installation was available while porting this toolbox, so these
tests validate correctness against ground truth instead of against MATLAB
output: for each of the 9 supported families, draw a large synthetic
sample from known scipy.stats parameters, then check that (a) identify_dist
recovers the correct family and (b) parameter_estimation recovers
parameters close to the true ones. This exercises the full pipeline,
including the empirically-verified GEV/GP shape-parameter sign
conventions (see parameters.py docstring).
"""
import numpy as np
import pytest
from scipy import stats

from lmoments import (
    identify_dist, parameter_estimation, pdf_l, cdf_l, random_l, js_div, lmom,
)

N = 200_000
SEED = 12345


def _sample(rv):
    return rv.rvs(size=N, random_state=np.random.default_rng(SEED))


CASES = [
    ("uniform", stats.uniform(loc=-2.0, scale=7.0), {}),
    ("normal", stats.norm(loc=3.0, scale=2.0), {}),
    ("exponential", stats.expon(scale=2.5), {}),
    ("gumbel", stats.gumbel_r(loc=2.0, scale=0.5), {}),
    ("logistic", stats.logistic(loc=1.0, scale=0.7), {}),
    ("generalized extreme value", stats.genextreme(c=0.2, loc=0.0, scale=1.0), {}),
    ("generalized pareto", stats.genpareto(c=0.3, loc=0.0, scale=1.0), {}),
    ("lognormal", stats.lognorm(s=0.5, loc=0.0, scale=np.exp(0.0)), {}),
    ("gamma", stats.gamma(a=5.0, loc=3.0, scale=0.8), {}),
]


# Gumbel is exactly the k=0 special case of the GEV curve, and Uniform is
# exactly the k=1 boundary case of the GP curve, so their (T3, T4) points
# are geometrically coincident with (or extremely close to) a point on the
# generalizing family's curve. Which of the two gets picked can flip with
# sampling noise -- this is an inherent property of the L-moment ratio
# diagram itself (present in the original MATLAB implementation too), not
# a defect in this port, so these two cases accept either family.
_DEGENERATE_ALTERNATES = {
    "gumbel": {"gumbel", "generalized extreme value"},
    "uniform": {"uniform", "generalized pareto"},
}


@pytest.mark.parametrize("name,rv,_", CASES, ids=[c[0] for c in CASES])
def test_identify_dist_recovers_true_family(name, rv, _):
    x = _sample(rv)
    result = identify_dist(x)
    acceptable = _DEGENERATE_ALTERNATES.get(name, {name})
    assert result["best"] in acceptable, (
        f"expected one of {acceptable}, got {result['best']} (ranking={result['ranking'][:3]})"
    )


@pytest.mark.parametrize("name,rv,_", CASES, ids=[c[0] for c in CASES])
def test_parameter_estimation_matches_true_distribution(name, rv, _):
    x = _sample(rv)
    L = lmom(x, 4)
    L1, L2, T3, T4 = L[0], L[1], L[2] / L[1], L[3] / L[1]
    params = parameter_estimation(x, name, L1, L2, T3, T4)

    grid = np.linspace(*np.percentile(x, [1, 99]), 200)
    true_cdf = rv.cdf(grid)
    fitted_cdf = cdf_l(grid, name, params)

    max_abs_diff = np.max(np.abs(true_cdf - fitted_cdf))
    assert max_abs_diff < 0.02, f"{name}: max |CDF diff| = {max_abs_diff:.4f}"


def test_pdf_integrates_to_one_lognormal():
    rv = stats.lognorm(s=0.4, loc=0.0, scale=1.0)
    x = _sample(rv)
    L = lmom(x, 4)
    params = parameter_estimation(x, "lognormal", L[0], L[1], L[2] / L[1], L[3] / L[1])
    grid = np.linspace(1e-6, np.percentile(x, 99.9), 5000)
    pdf_vals = pdf_l(grid, "lognormal", params)
    integral = np.trapezoid(pdf_vals, grid)
    assert abs(integral - 1.0) < 0.02


def test_random_l_matches_requested_distribution():
    rv = stats.gamma(a=2.0, loc=0.0, scale=1.5)
    x = _sample(rv)
    L = lmom(x, 4)
    params = parameter_estimation(x, "gamma", L[0], L[1], L[2] / L[1], L[3] / L[1])
    resampled = random_l("gamma", params, 100_000)
    # Compare via JS divergence of histograms rather than raw samples.
    edges = np.linspace(*np.percentile(x, [0.5, 99.5]), 40)
    p, _ = np.histogram(x, bins=edges)
    q, _ = np.histogram(resampled, bins=edges)
    assert js_div(p + 1, q + 1) < 0.01  # +1 avoids empty-bin log(0) noise


def test_js_div_zero_for_identical_distributions():
    p = np.array([1.0, 2.0, 3.0, 4.0])
    assert js_div(p, p) == pytest.approx(0.0, abs=1e-12)


def test_js_div_positive_for_different_distributions():
    p = np.array([10.0, 1.0, 1.0, 1.0])
    q = np.array([1.0, 1.0, 1.0, 10.0])
    assert js_div(p, q) > 0.1
