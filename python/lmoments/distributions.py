"""PDF/CDF evaluation and random sampling for the distributions supported
by the L-moment parameter estimators in parameters.py.

Python port of PDF_l.m / CDF_l.m / Random_l.m, built on scipy.stats.
Parameter-array layouts and shape-parameter sign conventions match
parameter_estimation() in parameters.py and were verified empirically
against scipy (see python/tests/test_lmoments.py), since no MATLAB
installation was available while porting.

Unlike the MATLAB version, the Gamma branch here applies the same
location shift in both the PDF and the CDF (MATLAB's CDF_l.m omits the
shift that PDF_l.m applies -- see the main README's "Known limitations").
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _scipy_dist(name: str, parameter):
    p = np.atleast_1d(np.asarray(parameter, dtype=float))

    if name == "uniform":
        return stats.uniform(loc=p[0], scale=p[1] - p[0])
    if name == "normal":
        return stats.norm(loc=p[0], scale=p[1])
    if name == "exponential":
        return stats.expon(scale=p[0])
    if name == "gumbel":
        return stats.gumbel_r(loc=p[1], scale=p[0])
    if name == "logistic":
        return stats.logistic(loc=p[0], scale=p[1])
    if name == "generalized extreme value":
        return stats.genextreme(c=-p[0], loc=p[2], scale=p[1])
    if name == "generalized pareto":
        return stats.genpareto(c=p[0], loc=p[2], scale=p[1])
    if name == "lognormal":
        return stats.lognorm(s=p[1], loc=p[2], scale=np.exp(p[0]))
    if name == "gamma":
        return stats.gamma(a=p[0], loc=p[2], scale=p[1])
    if name == "weibul":
        return stats.weibull_min(c=p[1], loc=p[2], scale=p[0])
    raise ValueError(f"unsupported distribution: {name!r}")


def pdf_l(x, name: str, parameter) -> np.ndarray:
    return _scipy_dist(name, parameter).pdf(x)


def cdf_l(x, name: str, parameter) -> np.ndarray:
    return _scipy_dist(name, parameter).cdf(x)


def random_l(name: str, parameter, size) -> np.ndarray:
    return _scipy_dist(name, parameter).rvs(size=size)
