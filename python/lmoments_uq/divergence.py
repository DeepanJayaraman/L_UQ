"""Kullback-Leibler and Jensen-Shannon divergence between two binned
probability distributions (log base 2, matching KLDiv.m / JSDiv.m).

This port operates on a single pair of 1-D probability/count vectors --
the only way KLDiv.m/JSDiv.m are exercised elsewhere in this toolbox.
MATLAB's KLDiv.m additionally accepts an n-row P matched row-wise against
Q, but does so via an index-deletion trick (`P(mask)=[]`) that silently
flattens P before summing for n>1, which does not generalize correctly;
that batch case is not replicated here.
"""
from __future__ import annotations

import numpy as np


def kl_div(p, q) -> float:
    """KL divergence D(P||Q), log base 2, for two 1-D count/probability vectors.

    Bins where p=0 contribute 0 (the standard 0*log(0) = 0 convention).
    If p has mass in a bin where q has none, the true divergence is
    infinite and this returns ``inf``. (This deliberately deviates from
    MATLAB's KLDiv.m, which drops such bins and can return a finite --
    even negative -- value; inside js_div the case is unreachable either
    way, because the mixture is positive wherever p is.)
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same number of bins")
    p = p / p.sum()
    q = q / q.sum()
    if np.any((p > 0) & (q == 0)):
        return float("inf")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log2(p / q)
    mask = p > 0
    return float(np.sum(p[mask] * r[mask]))


def js_div(p, q) -> float:
    """Jensen-Shannon divergence, log base 2 (range [0, 1])."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same number of bins")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
