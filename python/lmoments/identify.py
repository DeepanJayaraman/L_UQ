"""Distribution identification via the L-moment ratio diagram.

Python port of Identify_dist.m. Supports the same 9 distribution families
as the MATLAB version. 'weibul' is intentionally excluded from automatic
identification, matching the upstream MATLAB code (its candidate-search
loop only ever spans these 9 families) -- see the repository README's
"Known limitations" for details.
"""
from __future__ import annotations

import numpy as np

from .lmoments import lmom

DISTRIBUTIONS = [
    "uniform", "normal", "exponential", "gumbel", "logistic",
    "generalized extreme value", "generalized pareto", "lognormal", "gamma",
]

# Fixed (T3, T4) reference points on the L-moment ratio diagram.
_L_FIXED = {
    "uniform": (0.0, 0.0),
    "normal": (0.0, 0.1230),
    "exponential": (1 / 3, 1 / 6),
    "gumbel": (0.1699, 0.1504),
    "logistic": (0.0, 1 / 6),
}

# Polynomial coefficients approximating T4(T3) on the L-moment ratio
# diagram for the four "curved" families (columns 0-3). Column 4 is the
# Weibull approximation, kept for completeness but unused (see README).
_COEFF = np.array([
    [0.10701, 0, 0.12282, 0.12240, 0.107159752418765],
    [0.11090, 0.20196, 0, 0, -0.113630607247195],
    [0.84838, 0.95924, 0.77518, 0.30115, 0.842338510249911],
    [-0.06669, -0.20096, 0, 0, 0.0921732557597174],
    [0.00567, 0.04061, 0.12279, 0.95812, 0.0427093600517927],
    [-0.04208, 0, 0, 0, -0.0155156500400596],
    [0.03673, 0, -0.13638, -0.57488, -0.0321156200448707],
    [0, 0, 0, 0, 0.0363854870768234],
    [0, 0, 0.11368, 0.19383, 0.0398640323858772],
])

_CURVE_FAMILIES = ["generalized extreme value", "generalized pareto", "lognormal", "gamma"]
_CURVE_T3_RANGE = {
    "generalized extreme value": (-0.6, 0.9),
    "generalized pareto": (-0.9, 0.9),
    "lognormal": (-0.9, 0.9),
    "gamma": (-0.9, 0.9),
}


def _t4_curve(t3: np.ndarray, col: int) -> np.ndarray:
    a = _COEFF[:, col]
    return sum(a[i] * t3 ** i for i in range(9))


def _closest_on_curve(t3_s: float, t4_s: float, col: int, t3_range: tuple) -> float:
    t3_grid = np.arange(t3_range[0], t3_range[1] + 1e-9, 0.01)
    t4_grid = _t4_curve(t3_grid, col)
    return float(np.min(np.hypot(t3_grid - t3_s, t4_grid - t4_s)))


def identify_dist(x) -> dict:
    """Identify the closest-matching distribution family for sample x.

    Returns a dict with keys:
      - 'best': name of the closest-matching family (str)
      - 'ranking': all 9 families as (name, distance) sorted ascending --
        an addition over the MATLAB version, which only ever returns the
        single closest match
      - 'L_sample': [L1, L2, T3, T4]
    """
    L = lmom(x, 4)
    L1, L2, L3, L4 = L
    T3, T4 = L3 / L2, L4 / L2

    distances = {
        name: float(np.hypot(T3 - t3f, T4 - t4f))
        for name, (t3f, t4f) in _L_FIXED.items()
    }
    for col, name in enumerate(_CURVE_FAMILIES):
        distances[name] = _closest_on_curve(T3, T4, col, _CURVE_T3_RANGE[name])

    ranking = sorted(distances.items(), key=lambda kv: kv[1])
    return {
        "best": ranking[0][0],
        "ranking": ranking,
        "L_sample": [float(L1), float(L2), float(T3), float(T4)],
    }
