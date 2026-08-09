"""L-moments based uncertainty quantification from scarce samples including extremes.

Python port of the MATLAB toolbox at https://github.com/DeepanJayaraman/L-UQ.

Typical use::

    from lmoments_uq import fit_best

    fit = fit_best(x_scarce)      # identify the family and estimate parameters
    print(fit.summary())
    fit.sf(threshold)             # exceedance probability from the fit
    fit.js_div(x_reference)       # goodness of fit against a reference sample
    fit.ks_stat(x_reference)
    fit.plot(x_reference)
"""
from .lmoments import lmom, pwm, l_moment_ratios
from .identify import identify_dist, identify_dist_bootstrap, DISTRIBUTIONS
from .parameters import (parameter_estimation, parameter_identify,
                         fit_best, ParameterEstimationError)
from .distributions import pdf_l, cdf_l, random_l, PARAMETER_NAMES
from .divergence import kl_div, js_div, ks_stat, ks_test
from .results import (LMomentFit, IdentificationResult, BootstrapIdentification,
                      CandidateFits, LMoments)

__all__ = [
    "lmom", "pwm", "l_moment_ratios",
    "identify_dist", "identify_dist_bootstrap", "DISTRIBUTIONS",
    "parameter_estimation", "parameter_identify",
    "fit_best", "ParameterEstimationError",
    "pdf_l", "cdf_l", "random_l", "PARAMETER_NAMES",
    "kl_div", "js_div", "ks_stat", "ks_test",
    "LMomentFit", "IdentificationResult", "BootstrapIdentification",
    "CandidateFits", "LMoments",
]

__version__ = "2.0.0"
