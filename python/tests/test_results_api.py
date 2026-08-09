"""Tests for the result objects introduced in 2.0.0 and the fit-aware
divergence API.

Two things are checked throughout: that the new attribute/method API does
what it says, and that the 1.x dict-style access it replaced still works,
so existing code and replication scripts keep running.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from lmoments_uq import (BootstrapIdentification, CandidateFits,
                         IdentificationResult, LMomentFit,
                         ParameterEstimationError, cdf_l, fit_best,
                         identify_dist, identify_dist_bootstrap, js_div,
                         kl_div, ks_stat, ks_test, parameter_identify)


@pytest.fixture
def scarce_and_reference():
    """A scarce sample with one genuine extreme, plus a large reference
    sample from the same parent -- the situation the toolbox targets."""
    rng = np.random.default_rng(7)
    x = stats.lognorm.rvs(s=0.5, scale=1.0, size=12, random_state=rng)
    x = np.append(x, stats.lognorm.rvs(s=0.5, scale=1.0, size=100_000,
                                       random_state=rng).max())
    ref = stats.lognorm.rvs(s=0.5, scale=1.0, size=5000,
                            random_state=np.random.default_rng(11))
    return x, ref


# ---------------------------------------------------------------------------
# LMomentFit: attributes, methods, and dict backwards compatibility
# ---------------------------------------------------------------------------

def test_fit_best_returns_result_object(scarce_and_reference):
    x, _ = scarce_and_reference
    fit = fit_best(x)
    assert isinstance(fit, LMomentFit)
    assert isinstance(fit.distribution, str)
    assert fit.parameters.ndim == 1
    assert fit.rank >= 0
    assert fit.data.size == x.size
    assert len(fit.ranking) == 9
    assert repr(fit).startswith("<LMomentFit ")
    assert fit.distribution in fit.summary()


def test_fit_dict_access_still_works(scarce_and_reference):
    """The 1.x API returned a dict; that access pattern must keep working."""
    x, _ = scarce_and_reference
    fit = fit_best(x)

    assert fit["distribution"] == fit.distribution
    np.testing.assert_array_equal(fit["parameters"], fit.parameters)
    assert fit["rank"] == fit.rank
    assert fit["distance"] == fit.distance
    assert fit["ranking"] == fit.ranking
    assert fit["skipped"] == fit.skipped
    assert list(fit["L_sample"]) == list(fit.l_moments)

    assert set(fit.keys()) == {"distribution", "parameters", "distance",
                               "rank", "L_sample", "ranking", "skipped"}
    assert "distribution" in fit
    assert fit.get("nonexistent", "default") == "default"
    assert dict(fit)["distribution"] == fit.distribution
    with pytest.raises(KeyError):
        fit["not_a_key"]


def test_parameters_dict_names_match_array(scarce_and_reference):
    x, _ = scarce_and_reference
    fit = fit_best(x)
    assert len(fit.parameter_names) == fit.parameters.size
    assert list(fit.parameters_dict.values()) == list(fit.parameters)


def test_fit_evaluates_its_own_distribution(scarce_and_reference):
    """fit.cdf must agree with the free function it replaces."""
    x, _ = scarce_and_reference
    fit = fit_best(x)
    grid = np.linspace(x.min(), x.max(), 50)

    np.testing.assert_allclose(
        fit.cdf(grid), cdf_l(grid, fit.distribution, fit.parameters))
    np.testing.assert_allclose(fit.sf(grid), 1 - fit.cdf(grid), atol=1e-12)
    # ppf inverts cdf
    q = np.linspace(0.05, 0.95, 19)
    np.testing.assert_allclose(fit.cdf(fit.ppf(q)), q, atol=1e-8)
    assert np.all(fit.pdf(grid) >= 0)
    assert fit.rvs(size=10, random_state=0).shape == (10,)
    lo, hi = fit.interval(0.9)
    assert lo < hi


def test_l_moments_named_access(scarce_and_reference):
    x, _ = scarce_and_reference
    fit = fit_best(x)
    L = fit.l_moments
    assert (L.L1, L.L2, L.t3, L.t4) == tuple(L)
    assert L.L2 > 0


def test_used_fallback_flag_tracks_rank():
    """A near-symmetric sample lands on the lognormal curve, whose
    estimator needs positive L-skewness, so fit_best must fall back."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=int(rng.integers(8, 16)))
    fit = fit_best(x)
    assert fit.used_fallback
    assert fit.rank > 0
    assert any(name == "lognormal" for name, _ in fit.skipped)
    assert "skipped" in fit.summary()


# ---------------------------------------------------------------------------
# Fit-aware divergences -- the API the JSS editor asked for
# ---------------------------------------------------------------------------

def test_js_div_accepts_fit_and_data(scarce_and_reference):
    """js_div(fit, data) and fit.js_div(data) must both work and agree."""
    x, ref = scarce_and_reference
    fit = fit_best(x)

    d_function = js_div(fit, ref)
    d_method = fit.js_div(ref)
    assert d_function == d_method
    assert 0.0 <= d_function <= 1.0


def test_js_div_fit_form_matches_manual_binning(scarce_and_reference):
    """The internal binning must reproduce what callers used to do by
    hand, so previously published numbers do not move."""
    x, ref = scarce_and_reference
    fit = fit_best(x)

    edges = np.linspace(ref.min(), ref.max(), 40)      # 39 bins
    p_data, _ = np.histogram(ref, bins=edges)
    p_fit = np.diff(cdf_l(edges, fit.distribution, fit.parameters))
    manual = js_div(p_data + 1e-12, p_fit + 1e-12)

    assert fit.js_div(ref) == manual


def test_js_div_argument_order_does_not_matter(scarce_and_reference):
    x, ref = scarce_and_reference
    fit = fit_best(x)
    assert js_div(fit, ref) == pytest.approx(js_div(ref, fit))


def test_js_div_two_vector_form_unchanged():
    """The low-level form must keep its old behaviour."""
    p = np.array([0.2, 0.3, 0.4, 0.1])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    assert js_div(p, p) == pytest.approx(0.0, abs=1e-12)
    assert js_div(p, q) == pytest.approx(js_div(q, p))
    assert 0.0 <= js_div(p, q) <= 1.0


def test_js_div_accepts_a_frozen_scipy_distribution(scarce_and_reference):
    """Comparing an MLE fit against the same reference must work too --
    the replication script scores both routes this way."""
    _, ref = scarce_and_reference
    frozen = stats.lognorm(s=0.5, scale=1.0)
    assert 0.0 <= js_div(frozen, ref) <= 1.0


def test_js_div_prefers_the_right_family(scarce_and_reference):
    _, ref = scarce_and_reference
    right = stats.lognorm(s=0.5, scale=1.0)
    wrong = stats.expon(scale=8.0)
    assert js_div(right, ref) < js_div(wrong, ref)


def test_js_div_rejects_two_fits(scarce_and_reference):
    x, _ = scarce_and_reference
    fit = fit_best(x)
    with pytest.raises(TypeError, match="two fits"):
        js_div(fit, fit)


def test_js_div_mismatched_vectors_error_mentions_the_fit_form():
    with pytest.raises(ValueError, match="js_div\\(fit, x\\)"):
        js_div(np.ones(5), np.ones(7))


def test_kl_div_fit_form(scarce_and_reference):
    x, ref = scarce_and_reference
    fit = fit_best(x)
    d = fit.kl_div(ref)
    assert np.isfinite(d) and d >= 0
    assert d == kl_div(fit, ref)


def test_custom_bins_and_range_change_the_value(scarce_and_reference):
    x, ref = scarce_and_reference
    fit = fit_best(x)
    assert fit.js_div(ref, bins=10) != fit.js_div(ref, bins=100)
    assert fit.js_div(ref, range=(ref.min(), ref.max())) == fit.js_div(ref)


# ---------------------------------------------------------------------------
# Kolmogorov-Smirnov statistic
# ---------------------------------------------------------------------------

def test_ks_stat_matches_a_direct_computation(scarce_and_reference):
    x, ref = scarce_and_reference
    fit = fit_best(x)

    s = np.sort(ref)
    n = s.size
    cdf = fit.cdf(s)
    expected = max(np.max(np.arange(1, n + 1) / n - cdf),
                   np.max(cdf - np.arange(0, n) / n))
    assert fit.ks_stat(ref) == pytest.approx(expected)
    assert fit.ks_stat(ref) == ks_stat(fit, ref)
    assert ks_stat(ref, fit) == ks_stat(fit, ref)   # order-insensitive


def test_ks_stat_is_small_for_the_true_parent():
    """For a large sample from the exact parent, D ~ 1/sqrt(n)."""
    ref = stats.norm.rvs(loc=3, scale=2, size=20000,
                         random_state=np.random.default_rng(4242))
    assert ks_stat(stats.norm(loc=3, scale=2), ref) < 0.02


def test_ks_stat_bounded_and_larger_for_a_wrong_family(scarce_and_reference):
    _, ref = scarce_and_reference
    right = ks_stat(stats.lognorm(s=0.5, scale=1.0), ref)
    wrong = ks_stat(stats.expon(scale=8.0), ref)
    assert 0.0 <= right <= 1.0
    assert right < wrong


def test_ks_test_returns_statistic_and_pvalue(scarce_and_reference):
    x, ref = scarce_and_reference
    fit = fit_best(x)
    result = fit.ks_test(ref)
    assert result.statistic == pytest.approx(fit.ks_stat(ref))
    assert 0.0 <= result.pvalue <= 1.0


# ---------------------------------------------------------------------------
# IdentificationResult / BootstrapIdentification / CandidateFits
# ---------------------------------------------------------------------------

def test_identify_dist_result_object(scarce_and_reference):
    x, _ = scarce_and_reference
    ident = identify_dist(x)
    assert isinstance(ident, IdentificationResult)
    assert ident.best == ident.ranking[0][0]
    assert len(ident.top(3)) == 3
    assert ident.distances[ident.best] == ident.ranking[0][1]
    # legacy access
    assert ident["best"] == ident.best
    assert ident["ranking"] == ident.ranking
    assert list(ident["L_sample"]) == list(ident.l_moments)
    assert ident.best in ident.summary()


def test_bootstrap_result_object(scarce_and_reference):
    x, _ = scarce_and_reference
    boot = identify_dist_bootstrap(x, n_boot=200, random_state=0)
    assert isinstance(boot, BootstrapIdentification)
    assert boot.status in {"clear", "ambiguous"}
    assert boot.is_ambiguous == (boot.status == "ambiguous")
    assert sum(boot.frequencies.values()) == pytest.approx(1.0)
    assert boot.t3_samples.size == boot.n_boot
    assert boot.t4_samples.size == boot.n_boot
    # legacy access
    assert boot["status"] == boot.status
    assert boot["selection_frequencies"] == boot.selection_frequencies
    assert boot.status in boot.summary()


def test_parameter_identify_returns_k_feasible_candidates(scarce_and_reference):
    x, _ = scarce_and_reference
    cands = parameter_identify(x, k=3)
    assert isinstance(cands, CandidateFits)
    assert len(cands.fits) == 3
    assert all(isinstance(f, LMomentFit) for f in cands.fits)
    assert [f.distance for f in cands.fits] == sorted(
        f.distance for f in cands.fits)
    assert cands.best is cands.fits[0]
    assert cands[0] is cands.fits[0]       # positional access
    assert cands["fits"] == cands.fits     # legacy access


def test_parameter_identify_agrees_with_fit_best(scarce_and_reference):
    x, _ = scarce_and_reference
    assert parameter_identify(x, k=1).best.distribution == fit_best(x).distribution


def test_parameter_identify_skips_infeasible_candidates():
    """Real data whose nearest family has an invalid estimator must not
    raise -- before 2.0.0 this crashed."""
    x = np.array([66., 70., 69., 68., 67., 72., 73., 70., 57., 63., 70.,
                  78., 67., 53., 67., 75., 70., 81., 76., 79., 75., 76., 58.])
    cands = parameter_identify(x, k=3)
    assert len(cands.fits) == 3
    assert any(name == "lognormal" for name, _ in cands.skipped)
    assert "lognormal" not in [f.distribution for f in cands.fits]


def test_parameter_identify_strict_mode_still_raises():
    x = np.array([66., 70., 69., 68., 67., 72., 73., 70., 57., 63., 70.,
                  78., 67., 53., 67., 75., 70., 81., 76., 79., 75., 76., 58.])
    with pytest.raises(ParameterEstimationError):
        parameter_identify(x, k=3, strict=True)


def test_candidates_can_be_scored_against_a_reference(scarce_and_reference):
    """The comparison the article makes: rank competing families by how
    well each reproduces a reference sample."""
    x, ref = scarce_and_reference
    cands = parameter_identify(x, k=3)
    scores = [(f.distribution, f.js_div(ref)) for f in cands.fits]
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for _, s in scores)


# ---------------------------------------------------------------------------
# Plotting (smoke tests -- matplotlib is an optional dependency)
# ---------------------------------------------------------------------------

def test_plot_methods_produce_figures(scarce_and_reference):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, ref = scarce_and_reference
    fit = fit_best(x)

    fig = fit.plot()
    assert len(fig.axes) == 2
    plt.close(fig)

    fig = fit.plot(reference=ref)
    assert len(fig.axes) == 2
    plt.close(fig)

    fig = identify_dist(x).plot()
    plt.close(fig)

    fig = identify_dist_bootstrap(x, n_boot=100, random_state=0).plot()
    assert len(fig.axes) == 2
    plt.close(fig)
