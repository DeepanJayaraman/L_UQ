"""L-moments based uncertainty quantification from scarce samples including extremes.

Python port of the MATLAB toolbox at https://github.com/DeepanJayaraman/L-UQ.
"""
from .lmoments import lmom, pwm, l_moment_ratios
from .identify import identify_dist, DISTRIBUTIONS
from .parameters import (parameter_estimation, parameter_identify,
                         fit_best, ParameterEstimationError)
from .distributions import pdf_l, cdf_l, random_l
from .divergence import kl_div, js_div

__all__ = [
    "lmom", "pwm", "l_moment_ratios",
    "identify_dist", "DISTRIBUTIONS",
    "parameter_estimation", "parameter_identify",
    "fit_best", "ParameterEstimationError",
    "pdf_l", "cdf_l", "random_l",
    "kl_div", "js_div",
]

__version__ = "1.0.1"
