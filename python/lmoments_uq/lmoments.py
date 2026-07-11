"""Sample L-moments.

Python port of lmom.m / LegendreShiftPoly.m (Kobus N. Bekker; Peter Roche).
Uses the closed-form relation between probability-weighted moments (PWMs)
and L-moments (Hosking, 1990, eq. 2.3) instead of literally re-deriving the
shifted Legendre polynomial recurrence; the two are mathematically
equivalent and this form is easier to verify and extend to arbitrary order.
"""
from __future__ import annotations

import numpy as np
from scipy.special import comb


def pwm(x: np.ndarray, n_moments: int) -> np.ndarray:
    """Unbiased sample probability-weighted moments b_0..b_{n_moments-1}.

    b_r = (1/n) * sum_{j=r+1}^{n} [C(j-1,r) / C(n-1,r)] * x_(j)

    where x_(j) is the j-th order statistic (1-indexed, ascending).
    """
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = x.size
    if n_moments > n:
        raise ValueError(f"need at least {n_moments} samples, got {n}")
    b = np.empty(n_moments)
    b[0] = x.mean()
    for r in range(1, n_moments):
        j = np.arange(r + 1, n + 1)  # 1-indexed order-statistic index
        weight = comb(j - 1, r) / comb(n - 1, r)
        b[r] = np.sum(weight * x[j - 1]) / n
    return b


def lmom(x: np.ndarray, n_moments: int = 4) -> np.ndarray:
    """Sample L-moments L1..L_{n_moments} of data vector x.

    Matches MATLAB's ``lmom(X, nL)``: returns a length-``n_moments`` array
    ``[L1, L2, ..., L_nL]``.
    """
    b = pwm(x, n_moments)
    L = np.empty(n_moments)
    for r in range(n_moments):
        k = np.arange(r + 1)
        p = (-1.0) ** (r - k) * comb(r, k) * comb(r + k, k)
        L[r] = np.sum(p * b[: r + 1])
    return L


def l_moment_ratios(x: np.ndarray) -> dict:
    """L1, L2, L-skewness (T3) and L-kurtosis (T4) of a sample.

    Equivalent to the ``L_sample = [L1_S, L2_S, T3_S, T4_S]`` computed
    inline at the top of MATLAB's ``Identify_dist.m``.
    """
    L = lmom(x, 4)
    L1, L2, L3, L4 = L
    return {"L1": L1, "L2": L2, "T3": L3 / L2, "T4": L4 / L2}
