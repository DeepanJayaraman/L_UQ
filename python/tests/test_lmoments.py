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

from lmoments_uq import (
    identify_dist, identify_dist_bootstrap, parameter_estimation,
    pdf_l, cdf_l, random_l, js_div, lmom,
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
    """The L-moment ratio diagram must point back to the generating family.

    With n = 200,000 the sample (T3, T4) point converges to the family's
    theoretical location on the diagram, so identification should be
    unambiguous (up to the geometric degeneracies documented above). A
    failure here means either the sample L-moment computation (lmom/pwm)
    or a family's reference point/curve polynomial in identify.py is
    wrong -- with scarce real-world samples the method has sampling noise,
    but at this n there is no such excuse.
    """
    x = _sample(rv)
    result = identify_dist(x)
    acceptable = _DEGENERATE_ALTERNATES.get(name, {name})
    assert result["best"] in acceptable, (
        f"expected one of {acceptable}, got {result['best']} (ranking={result['ranking'][:3]})"
    )


@pytest.mark.parametrize("name,rv,_", CASES, ids=[c[0] for c in CASES])
def test_parameter_estimation_matches_true_distribution(name, rv, _):
    """Estimated parameters must reproduce the generating distribution.

    Rather than comparing parameter values directly (parameterizations
    differ between Hosking's conventions and scipy's), this compares the
    *fitted CDF* against the true CDF across the 1st-99th percentile
    range. That end-to-end check catches every way the chain can go
    wrong at once: a bad L-moment estimator, a mistranscribed
    Hosking/Wallis formula in parameters.py, or a wrong parameter-array
    -> scipy mapping (including shape-parameter sign flips for GEV/GP,
    the subtlest porting hazard) in distributions.py. The 0.02 tolerance
    absorbs residual sampling noise at n = 200,000 plus the small bias of
    Hosking's rational-polynomial approximations (e.g. the GEV shape and
    lognormal sigma fits), which are accurate to ~1e-3 in CDF terms.
    """
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
    """A fitted PDF must integrate to ~1 over the sample's support.

    Any density must have unit total mass; a violation would indicate a
    broken parameter mapping in pdf_l (e.g. applying the 3-parameter
    lognormal's location shift twice, or confusing scipy's s/scale with
    Hosking's mu/sigma). Lognormal is the family chosen here because its
    shift-and-exponentiate parameterization is the easiest to get subtly
    wrong. Tolerance 0.02 covers the mass beyond the 99.9th-percentile
    integration cutoff plus trapezoid-rule error.
    """
    rv = stats.lognorm(s=0.4, loc=0.0, scale=1.0)
    x = _sample(rv)
    L = lmom(x, 4)
    params = parameter_estimation(x, "lognormal", L[0], L[1], L[2] / L[1], L[3] / L[1])
    grid = np.linspace(1e-6, np.percentile(x, 99.9), 5000)
    pdf_vals = pdf_l(grid, "lognormal", params)
    integral = np.trapezoid(pdf_vals, grid)
    assert abs(integral - 1.0) < 0.02


def test_random_l_matches_requested_distribution():
    """random_l must draw from the same distribution that pdf_l/cdf_l fit.

    Fit a gamma to data, then resample from the fit; the original and
    resampled histograms should be nearly indistinguishable (JSD < 0.01,
    where 0 is identical and 1 is disjoint). This guards the consistency
    between random_l's parameter mapping and the estimator's -- if
    random_l interpreted the parameter array differently (e.g. dropped
    the gamma location shift, as MATLAB's Random_l.m nearly does with its
    eps subtraction), samples would land in visibly different bins.
    """
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
    """JSD(P, P) must be exactly 0 -- the identity-of-indiscernibles axiom.

    When P == Q the mixture M equals both, every log ratio is log2(1) = 0,
    and both KL terms vanish. A nonzero value would mean the
    normalization or mixture computation introduces asymmetric error.
    """
    p = np.array([1.0, 2.0, 3.0, 4.0])
    assert js_div(p, p) == pytest.approx(0.0, abs=1e-12)


def test_js_div_positive_for_different_distributions():
    """JSD must be strictly positive for genuinely different distributions.

    Gibbs' inequality makes both KL(P||M) and KL(Q||M) nonnegative, with
    zero only when P == Q. These two histograms concentrate mass at
    opposite ends, so their divergence must be far from zero -- a small
    value here would mean mass is being silently dropped (e.g. by
    over-aggressive masking of bins).
    """
    p = np.array([10.0, 1.0, 1.0, 1.0])
    q = np.array([1.0, 1.0, 1.0, 10.0])
    assert js_div(p, q) > 0.1


def test_js_div_hand_computed_value():
    """Pin js_div to a value derivable by hand from the definition.

    For p=[.5,.5] and q=[.25,.75], the mixture is m=[.375,.625] and
    JSD = 0.5*KL(p||m) + 0.5*KL(q||m) expands to the expression below
    (~0.0487949). Unlike the property-based tests, this anchors the
    *absolute scale*: it would catch a wrong log base (natural log
    instead of log2 changes the value by a factor of ln 2), a dropped 0.5
    factor, or a mixture computed before normalization.
    """
    expected = 0.5 * (0.5 * np.log2(0.5 / 0.375) + 0.5 * np.log2(0.5 / 0.625)) \
        + 0.5 * (0.25 * np.log2(0.25 / 0.375) + 0.75 * np.log2(0.75 / 0.625))
    assert js_div([0.5, 0.5], [0.25, 0.75]) == pytest.approx(expected, abs=1e-14)


def test_js_div_matches_scipy_including_zero_bins():
    """Cross-validate js_div against scipy's independent implementation.

    500 randomized histograms, with ~30% of bins zeroed in each, so the
    comparison exercises the 0*log(0) = 0 convention and shared-zero
    bins -- exactly the situations that arise with scarce samples binned
    over a wide range (most bins empty, one extreme far away). Agreement
    to 1e-12 against code we didn't write is the strongest evidence the
    masking logic drops only the bins it mathematically should. Note
    scipy's jensenshannon returns the JS *distance* sqrt(JSD), hence the
    square.
    """
    from scipy.spatial.distance import jensenshannon
    rng = np.random.default_rng(0)
    for _ in range(500):
        nbins = rng.integers(2, 60)
        p = rng.random(nbins)
        q = rng.random(nbins)
        p[rng.random(nbins) < 0.3] = 0.0
        q[rng.random(nbins) < 0.3] = 0.0
        if p.sum() == 0 or q.sum() == 0:
            continue
        # scipy's jensenshannon returns the JS *distance* = sqrt(JSD)
        assert js_div(p, q) == pytest.approx(jensenshannon(p, q, base=2) ** 2, abs=1e-12)


def test_js_div_is_symmetric_and_bounded():
    """JSD must be symmetric and, with log2, bounded in [0, 1].

    Symmetry holds because the mixture M treats P and Q identically --
    unlike raw KL, which is directional. The [0, 1] bound is specific to
    the log2 base (JSDiv.m's header advertises exactly this range); the
    upper bound is attained exactly when the supports are disjoint, since
    then each distribution sees the mixture as half its own mass
    everywhere: KL(P||M) = log2(2) = 1. The app relies on this bound when
    presenting divergences as comparable 0-to-1 fit-quality scores.
    """
    rng = np.random.default_rng(1)
    p = rng.random(20)
    q = rng.random(20)
    assert js_div(p, q) == pytest.approx(js_div(q, p), abs=1e-14)
    assert 0.0 <= js_div(p, q) <= 1.0
    # exactly 1 for distributions with disjoint support (log2 base)
    assert js_div([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0, abs=1e-14)


def test_kl_div_nonnegative_and_inf_on_support_violation():
    """Standalone kl_div must satisfy Gibbs' inequality and the inf convention.

    Three requirements of the definition:
    (1) KL(P||Q) >= 0 always, with equality iff P == Q (Gibbs'
        inequality) -- checked over 200 random strictly-positive pairs.
    (2) If P has mass in a bin where Q has none, KL is +inf by
        definition. This is a regression test for a fixed bug: the
        MATLAB-inherited bin-dropping returned a finite (even negative,
        -0.5 for this input) divergence here.
    (3) Bins where P = 0 contribute nothing (the 0*log(0) = 0
        convention), regardless of Q's mass there -- for p=[1,0] vs
        q=[.5,.5], only the first bin counts: 1*log2(1/0.5) = 1 exactly.
    """
    from lmoments_uq import kl_div
    # (1) Gibbs' inequality: KL >= 0 always
    rng = np.random.default_rng(2)
    for _ in range(200):
        p = rng.random(15) + 1e-9
        q = rng.random(15) + 1e-9
        assert kl_div(p, q) >= 0.0
    # (2) p has mass where q has none -> true KL is infinite
    assert kl_div([0.5, 0.5], [1.0, 0.0]) == float("inf")
    # (3) p=0 bins contribute nothing (0*log0 = 0 convention)
    assert kl_div([1.0, 0.0], [0.5, 0.5]) == pytest.approx(1.0, abs=1e-14)


# ---------------------------------------------------------------------------
# Domain-guard and ranked-fallback behavior (added alongside the JSS
# submission preparation, after repeated-trials benchmarking showed that
# sampling noise alone can push a small sample's L-skewness outside a
# family's closed-form estimator domain).
# ---------------------------------------------------------------------------
from lmoments_uq import fit_best, ParameterEstimationError  # noqa: E402


def test_lognormal_negative_skew_raises_not_nan():
    """A near-symmetric sample identified as lognormal used to produce NaN
    parameters silently (log of a negative erf); it must now raise a
    catchable, informative error instead."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # T3 = 0 exactly
    L = lmom(x, 4)
    L1, L2, T3, T4 = L[0], L[1], L[2] / L[1], L[3] / L[1]
    with pytest.raises(ParameterEstimationError):
        parameter_estimation(x, "lognormal", L1, L2, T3, T4)


def test_degenerate_sample_raises():
    """A constant sample has L2 = 0; every family must refuse it clearly."""
    x = np.full(10, 3.14)
    L = lmom(x, 4)
    with pytest.raises(ParameterEstimationError):
        parameter_estimation(x, "normal", L[0], L[1],
                             np.nan, np.nan)


def test_fit_best_falls_back_to_feasible_family():
    """fit_best must return a valid fit even when the top-ranked family's
    estimator domain excludes the sample, by walking down the ranking."""
    rng = np.random.default_rng(99)
    # Small near-symmetric samples are the documented failure mode:
    # ~half will have slightly negative T3, and some of those land
    # closest to the lognormal curve, whose estimator then rejects them.
    hit_fallback = False
    for _ in range(200):
        x = stats.norm.rvs(size=10, random_state=rng)
        result = fit_best(x)
        assert np.all(np.isfinite(result["parameters"]))
        assert result["distribution"] in dict(result["ranking"])
        if result["skipped"]:
            hit_fallback = True
            assert result["rank"] > 0
    assert hit_fallback, (
        "expected at least one of 200 small normal samples to exercise "
        "the ranked-fallback path; if this stops happening the guard "
        "conditions may have changed")


def test_fit_best_agrees_with_direct_path_when_no_fallback_needed():
    """When the top-ranked family fits cleanly, fit_best must equal the
    plain identify_dist + parameter_estimation composition."""
    rng = np.random.default_rng(7)
    x = stats.lognorm.rvs(s=0.5, size=200, random_state=rng)
    result = fit_best(x)
    direct = identify_dist(x)
    assert result["distribution"] == direct["best"]
    L1, L2, T3, T4 = direct["L_sample"]
    np.testing.assert_allclose(
        result["parameters"],
        parameter_estimation(x, direct["best"], L1, L2, T3, T4))
    assert result["skipped"] == []
    assert result["rank"] == 0


# ---------------------------------------------------------------------
# Bootstrap identification (uncertainty-aware model selection).
# ---------------------------------------------------------------------

def test_bootstrap_frequencies_are_a_valid_distribution():
    """Selection frequencies must be nonnegative, sum to one, be sorted
    descending, and the point 'best' must be reported."""
    rng = np.random.default_rng(3)
    x = stats.lognorm.rvs(s=0.5, size=15, random_state=rng)
    r = identify_dist_bootstrap(x, n_boot=500, random_state=0)
    freqs = [f for _, f in r["selection_frequencies"]]
    assert all(f >= 0 for f in freqs)
    assert abs(sum(freqs) - 1.0) < 1e-9
    assert freqs == sorted(freqs, reverse=True)
    assert r["best"] in dict(r["point_ranking"])
    assert r["status"] in ("clear", "ambiguous")
    assert r["t3_ci"][0] <= r["t3_ci"][1]
    assert r["t4_ci"][0] <= r["t4_ci"][1]


def test_bootstrap_large_clean_sample_is_clear():
    """A large sample from a family with a distinctive, isolated
    ratio-diagram position (logistic) should be identified with high,
    clear bootstrap frequency."""
    rng = np.random.default_rng(1)
    x = stats.logistic.rvs(loc=1.0, scale=0.7, size=2000, random_state=rng)
    r = identify_dist_bootstrap(x, n_boot=500, random_state=0)
    assert r["best"] == "logistic"
    top_family, top_freq = r["selection_frequencies"][0]
    assert top_family == "logistic"
    assert top_freq > 0.8
    assert r["status"] == "clear"


def test_bootstrap_scarce_sample_has_wider_ci_than_large():
    """Sampling uncertainty in (t3, t4) must shrink with n: the bootstrap
    confidence intervals for a scarce sample must be substantially wider
    than for a large sample from the same parent. This is the core
    statistical content the bootstrap exposes."""
    parent = stats.gamma(a=2.0)
    xs = parent.rvs(size=12, random_state=np.random.default_rng(5))
    xl = parent.rvs(size=2000, random_state=np.random.default_rng(5))
    rs = identify_dist_bootstrap(xs, n_boot=500, random_state=0)
    rl = identify_dist_bootstrap(xl, n_boot=500, random_state=0)
    width = lambda ci: ci[1] - ci[0]
    assert width(rs["t3_ci"]) > 3 * width(rl["t3_ci"])
    assert width(rs["t4_ci"]) > 3 * width(rl["t4_ci"])


def test_bootstrap_ambiguity_flag_fires_on_scarce_extreme_sample():
    """The scarce lognormal-plus-genuine-extreme sample of the paper's
    worked example is identified as GEV by the point estimate, but the
    bootstrap must reveal that choice is not clear-cut (ambiguous)."""
    rng = np.random.default_rng(7)
    x = stats.lognorm.rvs(s=0.5, scale=1.0, size=12, random_state=rng)
    x = np.append(x, stats.lognorm.rvs(
        s=0.5, scale=1.0, size=100_000, random_state=rng).max())
    r = identify_dist_bootstrap(x, n_boot=500, random_state=0)
    assert r["best"] == "generalized extreme value"
    assert r["status"] == "ambiguous"
    assert r["selection_frequencies"][0][1] < 0.6


def test_bootstrap_rejects_too_few_observations():
    with pytest.raises(ValueError):
        identify_dist_bootstrap([1.0, 2.0, 3.0], n_boot=100)
