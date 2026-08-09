"""Generate fixed, concrete reference cases for the R-side comparison
against `lmom` (comparison/compare_lmom.R).

Rather than trying to match numpy's and R's random-number streams (fragile
and unnecessary), this script draws each sample once in Python, writes the
raw data to CSV, and records this toolbox's own L-moments/parameter
estimates alongside it. `compare_lmom.R` then loads the *same* CSV data
and independently computes L-moments/parameters via the `lmom` package,
so the comparison is apples-to-apples on identical data.

Only families with a direct `lmom::pel*` equivalent are included here
(uniform and logistic have trivial closed forms identical by construction
in both implementations -- see the note in compare_lmom.R).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "L-UQ" / "python"))

from lmoments_uq import lmom, parameter_estimation, cdf_l  # noqa: E402

CASES_DIR = HERE / "reference_cases"
CASES_DIR.mkdir(exist_ok=True)

# (case name, our family name, scipy generator, kwargs, n, inject extreme?)
CASES = [
    ("normal_n20_extreme", "normal", stats.norm, dict(loc=0, scale=1), 20, True),
    ("exponential_n15_extreme", "exponential", stats.expon, dict(loc=0, scale=2), 15, True),
    ("gumbel_n20_extreme", "gumbel", stats.gumbel_r, dict(loc=0, scale=1), 20, True),
    ("gev_n25_extreme", "generalized extreme value", stats.genextreme,
     dict(c=0.15, loc=0, scale=1), 25, True),
    ("gpa_n25_extreme", "generalized pareto", stats.genpareto,
     dict(c=0.15, loc=0, scale=1), 25, True),
    ("lognormal_n12_extreme", "lognormal", stats.lognorm,
     dict(s=0.5, loc=0, scale=1), 12, True),
    ("gamma_n20_extreme", "gamma", stats.gamma, dict(a=2.0, loc=0, scale=1), 20, True),
]


def main():
    rng = np.random.default_rng(42)
    index = []
    summary_rows = []
    for name, family, dist, kwargs, n, inject in CASES:
        x = dist.rvs(size=n, random_state=rng, **kwargs)
        if inject:
            span = x.max() - x.min() if x.max() > x.min() else abs(x.max()) + 1
            x = np.append(x, x.max() + 6 * span)

        L = lmom(x, 4)
        L1, L2, T3, T4 = L[0], L[1], L[2] / L[1], L[3] / L[1]
        params = parameter_estimation(x, family, L1, L2, T3, T4)

        # Evaluate CDF at fixed sample-derived quantile points. Comparing
        # evaluated CDF values (rather than raw parameter vectors) avoids
        # the GEV/GP shape-parameter sign/ordering convention mismatch
        # between this toolbox and `lmom` documented in python/README.md
        # -- CDF(x) is convention-independent, so genuine numerical
        # agreement (or disagreement) shows up unambiguously here.
        qpoints = np.quantile(x, [0.1, 0.3, 0.5, 0.7, 0.9])
        cdf_vals = cdf_l(qpoints, family, params)

        np.savetxt(CASES_DIR / f"{name}.csv", x, delimiter=",", header="x", comments="")
        ref = dict(family=family, n=int(n), L1=L1, L2=L2, T3=T3, T4=T4,
                   python_params=list(map(float, params)),
                   qpoints=list(map(float, qpoints)),
                   python_cdf_at_qpoints=list(map(float, cdf_vals)))
        (CASES_DIR / f"{name}_reference.json").write_text(json.dumps(ref, indent=2))
        index.append(name)
        padded = list(params) + [np.nan] * (3 - len(params))
        summary_rows.append([name, family, n, L1, L2, T3, T4, *padded,
                              *qpoints, *cdf_vals])
        print(f"{name}: family={family} L=({L1:.6f},{L2:.6f},{T3:.6f},{T4:.6f}) "
              f"params={np.round(params, 6).tolist()}")

    (CASES_DIR / "index.json").write_text(json.dumps(index, indent=2))

    import csv as csv_mod
    with open(CASES_DIR / "reference_params.csv", "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(["case", "family", "n", "L1", "L2", "T3", "T4",
                    "python_param1", "python_param2", "python_param3",
                    "q10", "q30", "q50", "q70", "q90",
                    "python_cdf_q10", "python_cdf_q30", "python_cdf_q50",
                    "python_cdf_q70", "python_cdf_q90"])
        w.writerows(summary_rows)

    print(f"\nWrote {len(index)} reference cases to {CASES_DIR}")
    print(f"Wrote consolidated {CASES_DIR / 'reference_params.csv'} for the R script to diff against")


if __name__ == "__main__":
    main()
