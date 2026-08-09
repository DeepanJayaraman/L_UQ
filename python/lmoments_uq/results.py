"""Result objects returned by the fitting and identification routines.

Earlier releases returned plain ``dict`` objects, so exploring a fit meant
reaching into its internal structure (``fit["distribution"]``,
``fit["parameters"]``) and re-deriving everything else by hand. The classes
here expose the same information as documented attributes and methods --
evaluate the fitted CDF, score the fit against data, print a summary, draw
the diagnostic plot -- while still supporting the old key access, so code
written against the 1.x dict API keeps working unchanged.

    fit = fit_best(x_scarce)
    fit.distribution          # 'gamma'
    fit.parameters            # array([...])
    fit.parameters_dict       # {'shape': ..., 'scale': ..., 'loc': ...}
    fit.cdf(31.0)             # fitted CDF, no manual dispatch
    fit.js_div(x_full)        # binning handled internally
    fit.ks_stat(x_full)
    fit.plot(x_full)
    print(fit.summary())

    fit["distribution"]       # still works
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

from .distributions import PARAMETER_NAMES, _scipy_dist


class _KeyedResult:
    """Attribute-first result that also supports read-only mapping access.

    Mapping access exists purely for backwards compatibility with the 1.x
    dict-returning API and is not the recommended way to use these
    objects. ``collections.abc.Mapping`` is deliberately not used as a
    base class: its ``__eq__`` compares ``dict(self) == dict(other)``,
    which raises on the numpy arrays these results carry.
    """

    #: Mapping keys exposed for backwards compatibility, in display order.
    _keys: tuple = ()

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        return getattr(self, self._attr_for(key))

    def _attr_for(self, key: str) -> str:
        """Map a legacy dict key onto the attribute holding its value."""
        return key

    def keys(self) -> tuple:
        return self._keys

    def values(self) -> list:
        return [self[k] for k in self._keys]

    def items(self) -> list:
        return [(k, self[k]) for k in self._keys]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key) -> bool:
        return key in self._keys

    def __iter__(self) -> Iterator:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def to_dict(self) -> dict:
        """Plain dict of the legacy keys, e.g. for JSON/CSV serialization."""
        return dict(self.items())


class LMoments(tuple):
    """The four sample L-moment summaries, as a named 4-tuple.

    Indexing matches the legacy ``L_sample`` list ``[L1, L2, t3, t4]``.
    """

    __slots__ = ()

    def __new__(cls, L1, L2, t3, t4):
        return super().__new__(cls, (float(L1), float(L2), float(t3), float(t4)))

    L1 = property(lambda self: self[0], doc="First L-moment (the mean).")
    L2 = property(lambda self: self[1], doc="Second L-moment (L-scale).")
    t3 = property(lambda self: self[2], doc="L-skewness, L3/L2.")
    t4 = property(lambda self: self[3], doc="L-kurtosis, L4/L2.")

    def __repr__(self) -> str:
        return (f"LMoments(L1={self[0]:.6g}, L2={self[1]:.6g}, "
                f"t3={self[2]:.6g}, t4={self[3]:.6g})")


class LMomentFit(_KeyedResult):
    """A distribution fitted to a sample by the L-moment pipeline.

    Attributes
    ----------
    distribution : str
        Name of the fitted family.
    parameters : numpy.ndarray
        Estimated parameters, in the layout documented in
        :mod:`lmoments_uq.distributions` for this family.
    parameters_dict : dict
        The same parameters keyed by name.
    distance : float
        Ratio-diagram distance from the sample's ``(t3, t4)`` to this
        family's locus.
    rank : int
        Position of this family in the ratio-diagram ranking. ``0`` means
        it was the closest match; a positive rank means closer families
        were skipped because their estimator domain excluded this sample.
    l_moments : LMoments
        Sample L-moments ``(L1, L2, t3, t4)``.
    ranking : list of (str, float)
        All candidate families with their ratio-diagram distances,
        ascending.
    skipped : list of (str, str)
        ``(family, reason)`` for each closer family whose closed-form
        estimator was not valid for this sample.
    data : numpy.ndarray
        The sample the fit was estimated from (NaNs removed).
    """

    _keys = ("distribution", "parameters", "distance", "rank",
             "L_sample", "ranking", "skipped")

    def __init__(self, distribution, parameters, distance, rank,
                 l_moments, ranking, skipped, data):
        self.distribution = str(distribution)
        self.parameters = np.asarray(parameters, dtype=float)
        self.distance = float(distance)
        self.rank = int(rank)
        self.l_moments = (l_moments if isinstance(l_moments, LMoments)
                          else LMoments(*l_moments))
        self.ranking = list(ranking)
        self.skipped = list(skipped)
        self.data = np.asarray(data, dtype=float)
        self._frozen = None

    def _attr_for(self, key: str) -> str:
        return "l_moments" if key == "L_sample" else key

    # -- parameters ------------------------------------------------------

    @property
    def parameter_names(self) -> tuple:
        """Names of this family's parameters, positionally matching
        :attr:`parameters`."""
        return PARAMETER_NAMES[self.distribution]

    @property
    def parameters_dict(self) -> dict:
        return dict(zip(self.parameter_names,
                        (float(v) for v in self.parameters)))

    @property
    def used_fallback(self) -> bool:
        """True when the closest family on the ratio diagram was not the
        one fitted, because its estimator domain excluded this sample."""
        return self.rank > 0

    # -- distribution evaluation ----------------------------------------

    def frozen(self):
        """The fit as a frozen ``scipy.stats`` distribution.

        Gives access to the full scipy API (``mean``, ``interval``,
        ``expect``, ...) beyond the delegating methods below.
        """
        if self._frozen is None:
            self._frozen = _scipy_dist(self.distribution, self.parameters)
        return self._frozen

    def pdf(self, x) -> np.ndarray:
        """Fitted probability density at ``x``."""
        return self.frozen().pdf(x)

    def cdf(self, x) -> np.ndarray:
        """Fitted cumulative probability at ``x``."""
        return self.frozen().cdf(x)

    def sf(self, x) -> np.ndarray:
        """Fitted exceedance probability ``P(X > x)`` -- the quantity of
        interest in most reliability and risk calculations."""
        return self.frozen().sf(x)

    def ppf(self, q) -> np.ndarray:
        """Fitted quantile at probability ``q``."""
        return self.frozen().ppf(q)

    def rvs(self, size=1, random_state=None) -> np.ndarray:
        """Draw random variates from the fitted distribution."""
        return self.frozen().rvs(size=size, random_state=random_state)

    def interval(self, confidence=0.95) -> tuple:
        """Equal-tailed interval containing ``confidence`` mass."""
        return self.frozen().interval(confidence)

    # -- goodness of fit -------------------------------------------------

    def js_div(self, data, bins: int = 39, eps: float = 1e-12,
               range: tuple | None = None) -> float:
        """Jensen-Shannon divergence between this fit and ``data``.

        ``data`` is a raw sample -- typically a large reference sample the
        scarce-sample fit is being validated against. Binning is handled
        internally; see :func:`lmoments_uq.divergence.js_div`.
        """
        from .divergence import js_div as _js
        return _js(self, data, bins=bins, eps=eps, range=range)

    def kl_div(self, data, bins: int = 39, eps: float = 1e-12,
               range: tuple | None = None) -> float:
        """Kullback-Leibler divergence ``D(data || fit)``, in bits."""
        from .divergence import kl_div as _kl
        return _kl(self, data, bins=bins, eps=eps, range=range)

    def ks_stat(self, data) -> float:
        """Kolmogorov-Smirnov statistic against ``data``: the largest
        absolute gap between the fitted CDF and the sample's empirical
        CDF. Binning-free, unlike the divergence measures."""
        from .divergence import ks_stat as _ks
        return _ks(self, data)

    def ks_test(self, data):
        """KS statistic and its p-value against ``data``.

        The p-value is only interpretable when ``data`` is independent of
        the sample the fit came from. When the fit was estimated from
        ``data`` itself the null distribution no longer holds and the
        p-value is optimistic -- use :meth:`ks_stat` as a descriptive
        measure in that case.
        """
        from .divergence import ks_test as _ks_test
        return _ks_test(self, data)

    # -- presentation ----------------------------------------------------

    def summary(self) -> str:
        """Multi-line human-readable summary of the fit."""
        L = self.l_moments
        lines = [
            f"L-moment fit: {self.distribution}",
            f"  n observations   : {self.data.size}",
            f"  sample L-moments : L1={L.L1:.6g}  L2={L.L2:.6g}  "
            f"t3={L.t3:.6g}  t4={L.t4:.6g}",
            "  parameters       : "
            + ", ".join(f"{k}={v:.6g}" for k, v in self.parameters_dict.items()),
            f"  ratio-diagram    : rank {self.rank}, distance {self.distance:.6g}",
        ]
        if self.skipped:
            lines.append("  skipped (estimator domain):")
            for name, reason in self.skipped:
                lines.append(f"    - {name}: {reason}")
        lines.append("  candidate ranking:")
        for i, (name, dist) in enumerate(self.ranking):
            mark = " <-- fitted" if name == self.distribution else ""
            lines.append(f"    {i}. {name:26s} {dist:.6g}{mark}")
        return "\n".join(lines)

    def plot(self, reference=None, ax=None, bins: int | None = None,
             show_ecdf=True):
        """Diagnostic plot of the fit: density against a histogram of the
        sample, and the fitted CDF against the empirical CDF.

        Parameters
        ----------
        reference : array-like, optional
            A larger reference sample to overlay, for validating a
            scarce-sample fit against population truth.
        ax : matplotlib axes pair, optional
            Axes to draw into. A new figure is created when omitted.
        bins : int, optional
            Histogram bins for the density panel. Defaults to a
            square-root rule, which keeps a scarce sample's histogram from
            degenerating into isolated spikes.
        show_ecdf : bool
            Draw the sample's empirical CDF on the CDF panel.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "plotting requires matplotlib; install it with "
                "`pip install lmoments-uq[ui]`") from exc

        x = self.data
        ref = None if reference is None else np.asarray(reference, dtype=float)
        if bins is None:
            bins = int(min(20, max(5, np.sqrt(x.size))))

        lo = float(min(x.min(), ref.min())) if ref is not None else float(x.min())
        hi = float(max(x.max(), ref.max())) if ref is not None else float(x.max())
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        grid = np.linspace(lo - pad, hi + pad, 400)

        if ax is None:
            fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(10, 4))
        else:
            ax_pdf, ax_cdf = ax
            fig = ax_pdf.figure

        # density panel
        if ref is not None:
            ax_pdf.hist(ref, bins=50, density=True, alpha=0.25, color="0.5",
                        label=f"reference (n={ref.size})")
        ax_pdf.hist(x, bins=bins, density=True, alpha=0.45, color="tab:blue",
                    label=f"sample (n={x.size})")
        with np.errstate(invalid="ignore", divide="ignore"):
            ax_pdf.plot(grid, self.pdf(grid), lw=2, color="tab:red",
                        label=f"L-moment fit ({self.distribution})")
        ax_pdf.set_xlabel("x")
        ax_pdf.set_ylabel("density")
        ax_pdf.legend(fontsize=8)

        # CDF panel
        if ref is not None:
            r = np.sort(ref)
            ax_cdf.plot(r, np.arange(1, r.size + 1) / r.size, color="0.5",
                        lw=1.5, label=f"reference ECDF (n={ref.size})")
        if show_ecdf:
            s = np.sort(x)
            ax_cdf.step(s, np.arange(1, s.size + 1) / s.size, where="post",
                        color="tab:blue", label=f"sample ECDF (n={x.size})")
        with np.errstate(invalid="ignore", divide="ignore"):
            ax_cdf.plot(grid, self.cdf(grid), lw=2, color="tab:red",
                        label="L-moment fit")
        ax_cdf.set_xlabel("x")
        ax_cdf.set_ylabel("CDF")
        ax_cdf.set_ylim(-0.02, 1.02)
        ax_cdf.legend(fontsize=8)

        title = f"L-moment fit: {self.distribution}"
        if self.used_fallback:
            title += f" (rank {self.rank} fallback)"
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v:.4g}"
                           for k, v in self.parameters_dict.items())
        extra = f", rank={self.rank}" if self.used_fallback else ""
        return (f"<LMomentFit {self.distribution}({params}) "
                f"n={self.data.size}{extra}>")


class IdentificationResult(_KeyedResult):
    """Ratio-diagram identification of a sample's distribution family.

    Attributes
    ----------
    best : str
        Closest-matching family.
    ranking : list of (str, float)
        All candidate families with ratio-diagram distances, ascending.
    l_moments : LMoments
        Sample L-moments ``(L1, L2, t3, t4)``.
    """

    _keys = ("best", "ranking", "L_sample")

    def __init__(self, best, ranking, l_moments, data=None):
        self.best = str(best)
        self.ranking = list(ranking)
        self.l_moments = (l_moments if isinstance(l_moments, LMoments)
                          else LMoments(*l_moments))
        self.data = None if data is None else np.asarray(data, dtype=float)

    def _attr_for(self, key: str) -> str:
        return "l_moments" if key == "L_sample" else key

    @property
    def distances(self) -> dict:
        """Ratio-diagram distance keyed by family name."""
        return dict(self.ranking)

    def top(self, k: int = 3) -> list:
        """The ``k`` closest families as ``(name, distance)`` pairs."""
        return self.ranking[:k]

    def summary(self) -> str:
        L = self.l_moments
        lines = [
            f"Ratio-diagram identification: {self.best}",
            f"  sample L-moments : L1={L.L1:.6g}  L2={L.L2:.6g}  "
            f"t3={L.t3:.6g}  t4={L.t4:.6g}",
            "  candidate ranking:",
        ]
        for i, (name, dist) in enumerate(self.ranking):
            lines.append(f"    {i}. {name:26s} {dist:.6g}")
        return "\n".join(lines)

    def plot(self, ax=None):
        """Plot the sample's ``(t3, t4)`` against the ratio-diagram loci."""
        return _plot_ratio_diagram(
            [(self.l_moments.t3, self.l_moments.t4,
              f"sample (nearest: {self.best})")], ax=ax)

    def __repr__(self) -> str:
        L = self.l_moments
        return (f"<IdentificationResult best={self.best!r} "
                f"t3={L.t3:.4g} t4={L.t4:.4g}>")


class BootstrapIdentification(_KeyedResult):
    """Uncertainty-aware identification from bootstrap resampling.

    Attributes
    ----------
    best : str
        Point-estimate closest family.
    selection_frequencies : list of (str, float)
        How often each family was selected across resamples, descending.
    status : {'clear', 'ambiguous'}
        Whether the leading family is distinguishable from the runner-up
        under the configured thresholds.
    t3_ci, t4_ci : tuple of float
        95% percentile bootstrap intervals.
    point_ranking : list of (str, float)
        Ranking from the point estimate.
    n_boot : int
        Resamples that produced a valid identification.
    """

    _keys = ("best", "selection_frequencies", "status", "t3_ci", "t4_ci",
             "point_ranking", "n_boot")

    def __init__(self, best, selection_frequencies, status, t3_ci, t4_ci,
                 point_ranking, n_boot, t3_samples=None, t4_samples=None):
        self.best = str(best)
        self.selection_frequencies = list(selection_frequencies)
        self.status = str(status)
        self.t3_ci = tuple(float(v) for v in t3_ci)
        self.t4_ci = tuple(float(v) for v in t4_ci)
        self.point_ranking = list(point_ranking)
        self.n_boot = int(n_boot)
        self.t3_samples = (None if t3_samples is None
                           else np.asarray(t3_samples, dtype=float))
        self.t4_samples = (None if t4_samples is None
                           else np.asarray(t4_samples, dtype=float))

    @property
    def is_ambiguous(self) -> bool:
        return self.status == "ambiguous"

    @property
    def frequencies(self) -> dict:
        """Selection frequency keyed by family name."""
        return dict(self.selection_frequencies)

    def summary(self) -> str:
        lines = [
            f"Bootstrap identification: {self.best} (status: {self.status})",
            f"  resamples        : {self.n_boot}",
            f"  t3 95% interval  : [{self.t3_ci[0]:.4g}, {self.t3_ci[1]:.4g}]",
            f"  t4 95% interval  : [{self.t4_ci[0]:.4g}, {self.t4_ci[1]:.4g}]",
            "  selection frequencies:",
        ]
        for name, freq in self.selection_frequencies:
            if freq > 0:
                lines.append(f"    {name:26s} {100 * freq:5.1f}%")
        if self.is_ambiguous:
            lines.append("  The leading family is not clearly separated from "
                         "the runner-up;")
            lines.append("  treat the selected family as one plausible "
                         "candidate, not a decision.")
        return "\n".join(lines)

    def plot(self, ax=None):
        """Two-panel plot: the bootstrap ``(t3, t4)`` cloud over the
        ratio-diagram loci, and the family selection frequencies."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "plotting requires matplotlib; install it with "
                "`pip install lmoments-uq[ui]`") from exc

        if ax is None:
            fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
        else:
            axL, axR = ax
            fig = axL.figure

        cloud = None
        if self.t3_samples is not None and self.t4_samples is not None:
            cloud = (self.t3_samples, self.t4_samples)
        _plot_ratio_diagram([], ax=axL, cloud=cloud)
        axL.set_title(r"Bootstrap uncertainty in $(t_3, t_4)$")

        pairs = [(n, f) for n, f in self.selection_frequencies if f > 0][::-1]
        y = np.arange(len(pairs))
        axR.barh(y, [100 * f for _, f in pairs], color="tab:blue")
        axR.set_yticks(y)
        axR.set_yticklabels([n for n, _ in pairs], fontsize=9)
        axR.set_xlim(0, 100)
        axR.set_xlabel("bootstrap selection frequency (%)")
        axR.set_title(f"Family selection (status: {self.status})")
        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        top_name, top_freq = self.selection_frequencies[0]
        return (f"<BootstrapIdentification best={self.best!r} "
                f"status={self.status!r} top={top_name!r} "
                f"@{100 * top_freq:.0f}% of {self.n_boot}>")


class CandidateFits(_KeyedResult):
    """Top-``k`` candidate families with each one's parameter estimates.

    Attributes
    ----------
    fits : list of LMomentFit
        One fit per feasible candidate, in ranking order.
    ranking : list of (str, float)
    l_moments : LMoments
    skipped : list of (str, str)
        ``(family, reason)`` for each candidate passed over because its
        closed-form estimator was undefined for this sample.
    """

    _keys = ("fits", "L_sample", "ranking", "skipped")

    def __init__(self, fits, l_moments, ranking, skipped=()):
        self.fits = list(fits)
        self.l_moments = (l_moments if isinstance(l_moments, LMoments)
                          else LMoments(*l_moments))
        self.ranking = list(ranking)
        self.skipped = list(skipped)

    def _attr_for(self, key: str) -> str:
        return "l_moments" if key == "L_sample" else key

    def __getitem__(self, key):
        """Mapping access by legacy key, or positional access by index."""
        if isinstance(key, (int, np.integer, slice)):
            return self.fits[key]
        return super().__getitem__(key)

    @property
    def best(self) -> LMomentFit:
        """The highest-ranked candidate fit."""
        return self.fits[0]

    def summary(self) -> str:
        lines = [f"{len(self.fits)} candidate fit(s):"]
        for i, fit in enumerate(self.fits):
            params = ", ".join(f"{k}={v:.4g}"
                               for k, v in fit.parameters_dict.items())
            lines.append(f"  {i}. {fit.distribution:26s} "
                         f"d={fit.distance:.6g}  {params}")
        if self.skipped:
            lines.append("  skipped (estimator domain):")
            for name, reason in self.skipped:
                lines.append(f"    - {name}: {reason}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<CandidateFits {[f.distribution for f in self.fits]!r}>")


def _plot_ratio_diagram(points: Sequence[tuple], ax=None, cloud=None):
    """Draw the L-moment ratio diagram loci, optionally with sample points
    and a bootstrap cloud overlaid.

    ``points`` is a sequence of ``(t3, t4, label)``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "plotting requires matplotlib; install it with "
            "`pip install lmoments-uq[ui]`") from exc

    from .identify import (_L_FIXED, _CURVE_FAMILIES, _CURVE_T3_RANGE,
                           _t4_curve)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    for name, (t3f, t4f) in _L_FIXED.items():
        ax.plot(t3f, t4f, "o", color="black", markersize=4)
        ax.annotate(name, (t3f, t4f), textcoords="offset points",
                    xytext=(5, 4), fontsize=7)
    for col, name in enumerate(_CURVE_FAMILIES):
        lo, hi = _CURVE_T3_RANGE[name]
        grid = np.linspace(lo, hi, 200)
        ax.plot(grid, _t4_curve(grid, col), "-", lw=1.2, label=name)

    if cloud is not None:
        ax.scatter(cloud[0], cloud[1], s=6, alpha=0.12, color="tab:red",
                   label="bootstrap resamples")
    for t3, t4, label in points:
        ax.plot(t3, t4, "*", color="red", markersize=15, zorder=5, label=label)

    ax.set_xlabel(r"$t_3$ (L-skewness)")
    ax.set_ylabel(r"$t_4$ (L-kurtosis)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig
