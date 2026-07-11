"""Illustrative example for the L-moments based UQ toolbox (Python port of
demo_example.m).

Workflow:
  1. Draw a scarce sample (n=12) from a known distribution and inject one
     extreme value, generated as the maximum of 1e5 draws from the same
     parent (a genuine ~99.999th-percentile event; the extreme-generation
     scheme of Jayaraman & Ramu 2021).
  2. Identify the best-fit distribution family from the L-moment ratio
     diagram and estimate its parameters from L-moments.
  3. Fit the same family to the data using conventional moments/MLE
     (scipy's `.fit`) for comparison.
  4. Plot both fitted PDFs/CDFs against the sample, and compute the
     Jensen-Shannon divergence of each fit from the empirical distribution.

Run: python demo_example.py [--no-show]
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from lmoments_uq import identify_dist, parameter_estimation, pdf_l, cdf_l, js_div

# MATLAB name -> scipy.stats distribution used for the conventional/MLE fit.
# (MATLAB's 'ExtremeValue' models the smallest-extreme/min case, opposite of
# this toolbox's 'gumbel', which models the largest-extreme/max case; treat
# a gumbel comparison as approximate only, same caveat as in demo_example.m.)
_MLE_DIST = {
    "uniform": stats.uniform,
    "normal": stats.norm,
    "exponential": stats.expon,
    "gumbel": stats.gumbel_r,
    "logistic": stats.logistic,
    "generalized extreme value": stats.genextreme,
    "generalized pareto": stats.genpareto,
    "lognormal": stats.lognorm,
    "gamma": stats.gamma,
}


def run(seed: int = 7, show: bool = True):
    rng = np.random.default_rng(seed)

    n = 12
    true_mu, true_sigma = 0.0, 0.5
    x = stats.lognorm.rvs(s=true_sigma, scale=np.exp(true_mu), size=n, random_state=rng)
    # Injected extreme: the maximum of 1e5 draws from the same parent --
    # a genuine rare event (~the population's 99.999th percentile),
    # matching the extreme-generation scheme of Jayaraman & Ramu (2021).
    extreme = stats.lognorm.rvs(s=true_sigma, scale=np.exp(true_mu),
                                size=100_000, random_state=rng).max()
    x = np.append(x, extreme)

    result = identify_dist(x)
    fitted_dist = result["best"]
    L1, L2, T3, T4 = result["L_sample"]
    print(f"L-moment ratio diagram identified distribution: {fitted_dist}")
    print(f"Full ranking: {result['ranking']}")

    param_L = parameter_estimation(x, fitted_dist, L1, L2, T3, T4)
    print(f"L-moment parameter estimate: {param_L}")

    mle_family = _MLE_DIST[fitted_dist]
    mle_params = mle_family.fit(x)
    print(f"Conventional-moment (MLE) parameter estimate: {mle_params}")

    lo = max(x.min() - 1, 1e-6)
    hi = x.max() * 1.05
    xgrid = np.linspace(lo, hi, 400)

    pdf_L = pdf_l(xgrid, fitted_dist, param_L)
    pdf_conv = mle_family.pdf(xgrid, *mle_params)
    cdf_L = cdf_l(xgrid, fitted_dist, param_L)
    cdf_conv = mle_family.cdf(xgrid, *mle_params)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(x, density=True, alpha=0.3, label="Sample histogram")
    axes[0].plot(xgrid, pdf_L, lw=2, label="L-moment fit")
    axes[0].plot(xgrid, pdf_conv, "--", lw=2, label="Conventional-moment (MLE) fit")
    axes[0].axvline(x[-1], color="k", ls=":", label="extreme sample")
    axes[0].set_xlabel("X"); axes[0].set_ylabel("PDF")
    axes[0].set_title(f"PDF fit: {fitted_dist}")
    axes[0].legend()

    x_sorted = np.sort(x)
    f_emp = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    axes[1].step(x_sorted, f_emp, where="post", color="k", label="Empirical CDF")
    axes[1].plot(xgrid, cdf_L, lw=2, label="L-moment fit")
    axes[1].plot(xgrid, cdf_conv, "--", lw=2, label="Conventional-moment (MLE) fit")
    axes[1].set_xlabel("X"); axes[1].set_ylabel("CDF")
    axes[1].set_title(f"CDF fit: {fitted_dist}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("demo_example_output.png", dpi=150)
    fig.savefig("demo_example_output.pdf")
    print("Saved figure to demo_example_output.png / .pdf")

    edges = np.linspace(lo, hi, 30)
    p_emp, _ = np.histogram(x, bins=edges)
    p_L = np.diff(cdf_l(edges, fitted_dist, param_L))
    p_conv = np.diff(mle_family.cdf(edges, *mle_params))

    jsd_L = js_div(p_emp + 1e-12, p_L + 1e-12)
    jsd_conv = js_div(p_emp + 1e-12, p_conv + 1e-12)
    print(f"Jensen-Shannon divergence, empirical vs L-moment fit:       {jsd_L:.4f}")
    print(f"Jensen-Shannon divergence, empirical vs conventional fit:   {jsd_conv:.4f}")

    if show:
        plt.show()
    return fitted_dist, jsd_L, jsd_conv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true", help="don't open an interactive plot window")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(seed=args.seed, show=not args.no_show)
