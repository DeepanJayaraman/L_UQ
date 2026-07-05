"""Distribution parameter estimation from sample L-moments.

Python port of Parameter_estimation.m / parameter_identify.m. Formulas
follow Hosking, J.R.M. (1990) and Hosking & Wallis (1997), Regional
Frequency Analysis: An Approach Based on L-Moments.

Sign conventions for the shape parameters of the generalized extreme
value and generalized Pareto families were verified empirically against
scipy.stats (fit on synthetic data of known parameters) rather than by
reading MATLAB's internal `cdf`/`pdf` conventions, since no MATLAB
installation was available while porting -- see python/tests/test_lmoments.py.
"""
from __future__ import annotations

from math import erf, pi

import numpy as np
from scipy.stats import norm as _norm
from scipy.special import gamma as gammafn

from .lmoments import lmom
from .identify import identify_dist

EULER_GAMMA = 0.5772156649


def parameter_estimation(x, distribution: str, L1: float, L2: float, T3: float, T4: float) -> np.ndarray:
    """Estimate `distribution`'s parameters from sample L-moments.

    Returns a numpy array; see distributions.py for how each family's
    array maps onto scipy.stats parameters.
    """
    x = np.asarray(x, dtype=float)

    if distribution == "uniform":
        return np.array([L1 - 3 * L2, L1 + 3 * L2])

    if distribution == "normal":
        return np.array([L1, np.sqrt(pi) * L2])

    if distribution == "exponential":
        return np.array([L1])

    if distribution == "gumbel":
        alpha = L2 / np.log(2)
        eta = L1 - EULER_GAMMA * alpha
        return np.array([alpha, eta])

    if distribution == "logistic":
        return np.array([L1, L2])

    if distribution == "generalized extreme value":
        z = (2 / (3 + T3)) - (np.log(2) / np.log(3))
        k = 7.8590 * z + 2.9554 * z ** 2
        alpha = L2 * k / ((1 - 2 ** (-k)) * gammafn(1 + k))
        eta = L1 + (alpha * (gammafn(1 + k) - 1)) / k
        return np.array([-k, alpha, eta])

    if distribution == "generalized pareto":
        k = (1 - 3 * T3) / (1 + T3)
        alpha = L2 * (1 + k) * (2 + k)
        eta = L1 - (2 + k) * L2
        return np.array([-k, alpha, eta])

    if distribution == "lognormal":
        z = np.sqrt(8 / 3) * _norm.ppf((1 + T3) / 2)
        sigma = 0.999281 * z - 0.006118 * z ** 3 + 0.000127 * z ** 5
        mu = np.log(L2 / erf(sigma / 2)) - sigma ** 2 / 2
        eta = L1 - np.exp(mu + sigma ** 2 / 2)
        return np.array([mu, sigma, eta])

    if distribution == "gamma":
        x_new = x - x.min() + np.finfo(float).eps
        l1n = lmom(x_new, 1)[0]
        t = L2 / l1n
        if 0 < t < 0.5:
            z = pi * t ** 2
            alpha = (1 - 0.3080 * z) / (z - 0.05812 * z ** 2 + 0.01765 * z ** 3)
        else:
            z = 1 - t
            alpha = (0.7213 * z - 0.5947 * z ** 2) / (1 - 2.1817 * z + 1.2113 * z ** 2)
        beta = l1n / alpha
        return np.array([alpha, beta, x.min()])

    if distribution == "weibul":
        k = (285.3 * T3 ** 6 - 658.6 * T3 ** 5 + 622.8 * T3 ** 4
             - 317.2 * T3 ** 3 + 98.52 * T3 ** 2 - 21.256 * T3 + 3.516)
        A = L2 / ((1 - 2 ** (-1 / k)) * gammafn(1 + 1 / k))
        B = L1 - A * gammafn(1 + 1 / k)
        return np.array([A, k, B])

    raise ValueError(f"unsupported distribution: {distribution!r}")


def parameter_identify(x, k: int = 1) -> dict:
    """Identify the top-`k` candidate distributions and estimate each one's
    parameters from the sample's L-moments.

    Unlike MATLAB's parameter_identify.m (whose `K` argument silently
    breaks unless Identify_dist returns exactly K candidates -- it never
    returns more than one), this version genuinely ranks all 9 supported
    families and estimates parameters for the requested top `k`.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    result = identify_dist(x)
    L1, L2, T3, T4 = result["L_sample"]

    top_k = result["ranking"][:k]
    fits = []
    for name, distance in top_k:
        params = parameter_estimation(x, name, L1, L2, T3, T4)
        fits.append({"distribution": name, "distance": distance, "parameters": params})

    return {
        "fits": fits,
        "L_sample": result["L_sample"],
        "ranking": result["ranking"],
    }
