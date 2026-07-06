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


class ParameterEstimationError(ValueError):
    """Raised when a sample's L-moments fall outside the domain where a
    family's closed-form L-moment estimator is valid.

    This makes domain violations explicit and catchable instead of
    silently propagating NaN (e.g. log of a non-positive argument in the
    lognormal formula when L-skewness is negative). Callers wanting
    automatic recovery should use `fit_best`, which falls back to the
    next-ranked family on the L-moment ratio diagram.
    """


def _validated(distribution: str, params: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(params)):
        raise ParameterEstimationError(
            f"{distribution}: closed-form L-moment estimate is not finite "
            f"for this sample's L-moments (got {params})")
    return params


def parameter_estimation(x, distribution: str, L1: float, L2: float, T3: float, T4: float) -> np.ndarray:
    """Estimate `distribution`'s parameters from sample L-moments.

    Returns a numpy array; see distributions.py for how each family's
    array maps onto scipy.stats parameters.

    Raises ParameterEstimationError when the sample's L-moments fall
    outside the domain where the family's closed-form estimator is valid
    (rather than returning NaN/invalid parameters).
    """
    x = np.asarray(x, dtype=float)

    if not (np.isfinite(L2) and L2 > 0):
        raise ParameterEstimationError(
            f"L2 must be positive and finite (got {L2}); "
            "constant or degenerate samples cannot be fitted")

    if distribution == "uniform":
        return _validated(distribution, np.array([L1 - 3 * L2, L1 + 3 * L2]))

    if distribution == "normal":
        return _validated(distribution, np.array([L1, np.sqrt(pi) * L2]))

    if distribution == "exponential":
        if L1 <= 0:
            raise ParameterEstimationError(
                f"exponential: mean L1 must be positive (got {L1})")
        return _validated(distribution, np.array([L1]))

    if distribution == "gumbel":
        alpha = L2 / np.log(2)
        eta = L1 - EULER_GAMMA * alpha
        return _validated(distribution, np.array([alpha, eta]))

    if distribution == "logistic":
        return _validated(distribution, np.array([L1, L2]))

    if distribution == "generalized extreme value":
        z = (2 / (3 + T3)) - (np.log(2) / np.log(3))
        k = 7.8590 * z + 2.9554 * z ** 2
        if k <= -1:
            # gamma(1+k) has poles at nonpositive integers and the GEV
            # mean does not exist for k <= -1; the Hosking approximation
            # is outside its stated validity here.
            raise ParameterEstimationError(
                f"generalized extreme value: estimated shape k={k:.4f} <= -1 "
                f"(T3={T3:.4f} outside the approximation's validity range)")
        denom = (1 - 2 ** (-k)) * gammafn(1 + k)
        if not np.isfinite(denom) or denom == 0:
            raise ParameterEstimationError(
                f"generalized extreme value: singular scale denominator "
                f"for estimated shape k={k:.4f}")
        alpha = L2 * k / denom
        eta = L1 + (alpha * (gammafn(1 + k) - 1)) / k
        if alpha <= 0:
            raise ParameterEstimationError(
                f"generalized extreme value: nonpositive scale alpha={alpha:.4f}")
        return _validated(distribution, np.array([-k, alpha, eta]))

    if distribution == "generalized pareto":
        if T3 <= -1 or T3 >= 1:
            raise ParameterEstimationError(
                f"generalized pareto: T3={T3:.4f} outside (-1, 1)")
        k = (1 - 3 * T3) / (1 + T3)
        alpha = L2 * (1 + k) * (2 + k)
        eta = L1 - (2 + k) * L2
        if alpha <= 0:
            raise ParameterEstimationError(
                f"generalized pareto: nonpositive scale alpha={alpha:.4f} "
                f"(estimated shape k={k:.4f})")
        return _validated(distribution, np.array([-k, alpha, eta]))

    if distribution == "lognormal":
        if T3 <= 0:
            # This parameterization covers the positively skewed
            # three-parameter lognormal only; T3 <= 0 drives sigma <= 0
            # and log(L2/erf(sigma/2)) out of its domain. This is the
            # single most common failure mode observed in repeated-trials
            # benchmarking (near-symmetric samples whose ratio-diagram
            # position lands on the lognormal curve by sampling noise).
            raise ParameterEstimationError(
                f"lognormal: requires positive L-skewness (got T3={T3:.4f}); "
                "this parameterization covers positively skewed samples only")
        z = np.sqrt(8 / 3) * _norm.ppf((1 + T3) / 2)
        sigma = 0.999281 * z - 0.006118 * z ** 3 + 0.000127 * z ** 5
        ratio = L2 / erf(sigma / 2)
        if not np.isfinite(ratio) or ratio <= 0:
            raise ParameterEstimationError(
                f"lognormal: invalid L2/erf(sigma/2)={ratio} for sigma={sigma:.4f}")
        mu = np.log(ratio) - sigma ** 2 / 2
        eta = L1 - np.exp(mu + sigma ** 2 / 2)
        return _validated(distribution, np.array([mu, sigma, eta]))

    if distribution == "gamma":
        x_new = x - x.min() + np.finfo(float).eps
        l1n = lmom(x_new, 1)[0]
        if l1n <= 0:
            raise ParameterEstimationError(
                f"gamma: shifted-sample mean must be positive (got {l1n})")
        t = L2 / l1n
        if not (0 < t < 1):
            raise ParameterEstimationError(
                f"gamma: L-CV of shifted sample t={t:.4f} outside (0, 1)")
        if t < 0.5:
            z = pi * t ** 2
            alpha = (1 - 0.3080 * z) / (z - 0.05812 * z ** 2 + 0.01765 * z ** 3)
        else:
            z = 1 - t
            alpha = (0.7213 * z - 0.5947 * z ** 2) / (1 - 2.1817 * z + 1.2113 * z ** 2)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ParameterEstimationError(
                f"gamma: nonpositive/invalid shape alpha={alpha} (t={t:.4f})")
        beta = l1n / alpha
        return _validated(distribution, np.array([alpha, beta, x.min()]))

    if distribution == "weibul":
        k = (285.3 * T3 ** 6 - 658.6 * T3 ** 5 + 622.8 * T3 ** 4
             - 317.2 * T3 ** 3 + 98.52 * T3 ** 2 - 21.256 * T3 + 3.516)
        if not np.isfinite(k) or k <= 0:
            raise ParameterEstimationError(
                f"weibul: nonpositive/invalid shape k={k} (T3={T3:.4f})")
        A = L2 / ((1 - 2 ** (-1 / k)) * gammafn(1 + 1 / k))
        B = L1 - A * gammafn(1 + 1 / k)
        if not np.isfinite(A) or A <= 0:
            raise ParameterEstimationError(
                f"weibul: nonpositive/invalid scale A={A} (shape k={k:.4f})")
        return _validated(distribution, np.array([A, k, B]))

    raise ValueError(f"unsupported distribution: {distribution!r}")


def fit_best(x) -> dict:
    """Identify and fit the best *feasible* family for a sample.

    Walks the L-moment ratio diagram ranking from `identify_dist` and
    returns the first family whose closed-form estimator succeeds --
    falling back to the next-ranked candidate whenever a sample's
    L-moments land outside a family's estimator domain (see
    ParameterEstimationError). Repeated-trials benchmarking showed such
    domain violations occur for a nontrivial share of small samples by
    sampling noise alone, so callers using automatic identification
    should prefer this over identify_dist + parameter_estimation.

    Returns {'distribution', 'parameters', 'distance', 'rank',
    'L_sample', 'ranking', 'skipped'} where 'skipped' lists
    (family, reason) pairs for any higher-ranked families that failed.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    result = identify_dist(x)
    L1, L2, T3, T4 = result["L_sample"]

    skipped = []
    for rank, (name, distance) in enumerate(result["ranking"]):
        try:
            params = parameter_estimation(x, name, L1, L2, T3, T4)
        except ParameterEstimationError as exc:
            skipped.append((name, str(exc)))
            continue
        return {
            "distribution": name,
            "parameters": params,
            "distance": distance,
            "rank": rank,
            "L_sample": result["L_sample"],
            "ranking": result["ranking"],
            "skipped": skipped,
        }

    raise ParameterEstimationError(
        "no supported family's closed-form estimator is valid for this "
        f"sample's L-moments (L1={L1:.4f}, L2={L2:.4f}, T3={T3:.4f}, "
        f"T4={T4:.4f}); tried, in ranking order: "
        + "; ".join(f"{n}: {r}" for n, r in skipped))


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
