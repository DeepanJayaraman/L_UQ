"""Divergence and distance measures for comparing a fit with data.

Two calling conventions are supported.

**Fit against data** -- the common case when validating a scarce-sample
fit against a larger reference sample. Pass the fit and the data
directly; binning is handled internally::

    fit = fit_best(x_scarce)
    js_div(fit, x_full)          # or, equivalently, fit.js_div(x_full)
    ks_stat(fit, x_full)

**Two binned mass vectors** -- the low-level form, unchanged from earlier
releases, for callers who have already binned their data or who are
comparing two histograms::

    js_div(p_counts, q_counts)

Both divergences use log base 2, matching KLDiv.m / JSDiv.m, so the
Jensen-Shannon divergence lies in [0, 1].

This port operates on a single pair of 1-D probability/count vectors --
the only way KLDiv.m/JSDiv.m are exercised elsewhere in this toolbox.
MATLAB's KLDiv.m additionally accepts an n-row P matched row-wise against
Q, but does so via an index-deletion trick (`P(mask)=[]`) that silently
flattens P before summing for n>1, which does not generalize correctly;
that batch case is not replicated here.
"""
from __future__ import annotations

import numpy as np

#: Default number of histogram bins used when a fit is compared against a
#: raw sample. 39 bins (40 edges) spanning the data range reproduces the
#: binning used in the manuscript's worked examples.
DEFAULT_BINS = 39

#: Mass added to every bin before normalizing, so that a bin the fit
#: assigns no probability to does not make the divergence infinite.
DEFAULT_EPS = 1e-12


def _is_fit(obj) -> bool:
    """True for objects that expose a fitted CDF (an LMomentFit, or a
    frozen scipy distribution)."""
    return hasattr(obj, "cdf") and not isinstance(obj, np.ndarray)


def _binned(a, b, bins: int, eps: float, range=None):
    """Reduce a (fit, data) or (data, fit) pair to a comparable pair of
    binned probability vectors.

    Bin edges span the raw sample's range (or ``range`` when given). The
    sample contributes histogram counts; the fit contributes the
    probability mass it assigns to each bin, ``diff(cdf(edges))``.
    Returns ``(p, q)`` in the same order as the arguments.
    """
    fit_first = _is_fit(a)
    fit = a if fit_first else b
    data = np.asarray(b if fit_first else a, dtype=float).ravel()
    data = data[~np.isnan(data)]
    if data.size == 0:
        raise ValueError("no finite observations to compare the fit against")
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")

    if range is None:
        lo, hi = float(data.min()), float(data.max())
    else:
        lo, hi = float(range[0]), float(range[1])
    if not hi > lo:
        raise ValueError(
            f"cannot bin over a degenerate range [{lo}, {hi}]; pass an "
            "explicit `range=(lo, hi)`")

    edges = np.linspace(lo, hi, bins + 1)
    p_data, _ = np.histogram(data, bins=edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_fit = np.diff(np.asarray(fit.cdf(edges), dtype=float))
    p_fit = np.clip(np.nan_to_num(p_fit, nan=0.0), 0.0, None)

    p_data = p_data + eps
    p_fit = p_fit + eps
    return (p_fit, p_data) if fit_first else (p_data, p_fit)


def _resolve(p, q, bins: int, eps: float, range=None):
    """Dispatch the two calling conventions onto a pair of mass vectors."""
    if _is_fit(p) and _is_fit(q):
        raise TypeError(
            "comparing two fits directly is not supported; evaluate both "
            "against a common sample, e.g. js_div(fit_a, x), js_div(fit_b, x)")
    if _is_fit(p) or _is_fit(q):
        return _binned(p, q, bins, eps, range)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError(
            f"p and q must have the same number of bins (got {p.shape} and "
            f"{q.shape}); to compare a fit with a raw sample pass the fit "
            "itself, e.g. js_div(fit, x)")
    return p, q


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence of two already-normalized mass vectors, in bits."""
    if np.any((p > 0) & (q == 0)):
        return float("inf")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log2(p / q)
    mask = p > 0
    return float(np.sum(p[mask] * r[mask]))


def kl_div(p, q, *, bins: int = DEFAULT_BINS, eps: float = DEFAULT_EPS,
           range=None) -> float:
    """KL divergence D(P||Q), log base 2.

    Accepts either two binned count/probability vectors, or a fit and a
    raw sample in either order (see the module docstring). When a fit is
    involved the returned value is the divergence of the *first* argument
    from the second, after binning.

    Bins where p=0 contribute 0 (the standard 0*log(0) = 0 convention).
    If p has mass in a bin where q has none, the true divergence is
    infinite and this returns ``inf``. (This deliberately deviates from
    MATLAB's KLDiv.m, which drops such bins and can return a finite --
    even negative -- value; inside js_div the case is unreachable either
    way, because the mixture is positive wherever p is.)
    """
    p, q = _resolve(p, q, bins, eps, range)
    return _kl(p / p.sum(), q / q.sum())


def js_div(p, q, *, bins: int = DEFAULT_BINS, eps: float = DEFAULT_EPS,
           range=None) -> float:
    """Jensen-Shannon divergence, log base 2 (range [0, 1]).

    Accepts either two binned count/probability vectors, or a fit and a
    raw sample in either order::

        fit = fit_best(x_scarce)
        js_div(fit, x_full)

    Symmetric in its arguments, so the order does not matter.

    Parameters
    ----------
    bins : int
        Histogram bins used when a fit is compared against a raw sample.
        Ignored for the two-vector form.
    eps : float
        Mass added to each bin before normalizing, guarding against empty
        bins.
    range : tuple, optional
        Explicit ``(lo, hi)`` binning range. Defaults to the raw sample's
        range.
    """
    p, q = _resolve(p, q, bins, eps, range)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def ks_stat(fit, data) -> float:
    """Kolmogorov-Smirnov statistic between a fit and a raw sample.

    The largest absolute gap between the fitted CDF and the sample's
    empirical CDF,

        D = sup_x |F_fit(x) - F_n(x)|,

    evaluated exactly at the order statistics (checking both the left and
    right limits of the step function, as the supremum is attained at one
    of them). Unlike the divergence measures this needs no binning, so it
    carries no bin-width tuning parameter -- useful as a check that a
    JS-divergence comparison is not an artifact of the chosen bins.

    Either argument order is accepted.
    """
    if not _is_fit(fit):
        fit, data = data, fit
    if not _is_fit(fit):
        raise TypeError("ks_stat requires a fitted distribution and a sample")

    x = np.asarray(data, dtype=float).ravel()
    x = np.sort(x[~np.isnan(x)])
    n = x.size
    if n == 0:
        raise ValueError("no finite observations to compare the fit against")

    with np.errstate(invalid="ignore", divide="ignore"):
        cdf = np.asarray(fit.cdf(x), dtype=float)
    cdf = np.nan_to_num(cdf, nan=0.0)
    upper = np.arange(1, n + 1) / n - cdf   # F_n just after each point
    lower = cdf - np.arange(0, n) / n       # F_n just before each point
    return float(max(upper.max(), lower.max()))


def ks_test(fit, data):
    """KS statistic and p-value for a fit against a raw sample.

    Returns a ``scipy.stats`` result object with ``statistic`` and
    ``pvalue``. The p-value assumes the fitted parameters were *not*
    estimated from ``data``; when they were, it is optimistic (too large)
    and only :func:`ks_stat` should be reported. Validating a
    scarce-sample fit against an independent reference sample is the case
    where the p-value is meaningful.
    """
    from scipy import stats as _stats

    if not _is_fit(fit):
        fit, data = data, fit
    if not _is_fit(fit):
        raise TypeError("ks_test requires a fitted distribution and a sample")

    x = np.asarray(data, dtype=float).ravel()
    x = x[~np.isnan(x)]
    frozen = fit.frozen() if hasattr(fit, "frozen") else fit
    return _stats.kstest(x, frozen.cdf)
