% Unit tests for the MATLAB UQ toolbox, mirroring the coverage of the
% Python suite (python/tests/test_lmoments.py): for each supported
% family, draw a large synthetic sample from known parameters, then
% check that (a) Identify_dist recovers the generating family and
% (b) the fitted CDF from the L-moment parameter estimates tracks the
% empirical CDF closely. Also covers the shifted-gamma PDF/CDF
% consistency fix, explicit three-parameter Weibull support, the
% divergence functions, and the Parameter_estimation domain guards.
%
% Run from the repository root:
%   results = runtests('tests');  assertSuccess(results);
%
% Requires the Statistics and Machine Learning Toolbox (same
% requirement as the toolbox itself).

function tests = test_uq_matlab
tests = functiontests(localfunctions);
end

function setupOnce(testCase)  %#ok<INUSD>
% Make the toolbox functions visible when running from tests/.
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
rng(12345, 'twister');
end

% ---------------------------------------------------------------------
% Identification: large samples must point back to the generating family.
% Gumbel/GEV and Uniform/GP are geometrically degenerate pairs on the
% ratio diagram (Gumbel IS the k=0 point of the GEV curve; Uniform IS
% the k=1 boundary of the GP curve), so those accept either family --
% same policy as the Python tests.
% ---------------------------------------------------------------------

function test_identify_normal(testCase)
% Normal/gamma is a third degenerate pair: the zero-skew limit of the
% shifted gamma (Pearson III) IS the normal, so its ratio-diagram curve
% passes through the normal point and a symmetric sample can land
% closer to either by sampling noise alone.
X = random('normal', 3, 2, 200000, 1);
D = Identify_dist(X);
verifyTrue(testCase, any(strcmp(D{1}, {'normal', 'gamma'})));
end

function test_identify_exponential(testCase)
X = random('exponential', 2.5, 200000, 1);
D = Identify_dist(X);
verifyEqual(testCase, D{1}, 'exponential');
end

function test_identify_gumbel_or_gev(testCase)
X = random('generalized extreme value', 0, 0.5, 2, 200000, 1);
D = Identify_dist(X);
verifyTrue(testCase, any(strcmp(D{1}, ...
    {'gumbel', 'generalized extreme value'})));
end

function test_identify_logistic(testCase)
% MATLAB's random() has no 'logistic'; inverse-CDF sample it.
U = rand(200000, 1);
X = 1.0 + 0.7*log(U./(1-U));
D = Identify_dist(X);
verifyEqual(testCase, D{1}, 'logistic');
end

function test_identify_lognormal(testCase)
X = random('lognormal', 0, 0.5, 200000, 1);
D = Identify_dist(X);
verifyEqual(testCase, D{1}, 'lognormal');
end

function test_identify_gamma(testCase)
X = 3 + random('gamma', 5, 0.8, 200000, 1);
D = Identify_dist(X);
verifyEqual(testCase, D{1}, 'gamma');
end

function test_identify_uniform_or_gp(testCase)
X = random('uniform', -2, 5, 200000, 1);
D = Identify_dist(X);
verifyTrue(testCase, any(strcmp(D{1}, ...
    {'uniform', 'generalized pareto'})));
end

% ---------------------------------------------------------------------
% Fitted-CDF accuracy: the L-moment fit of the identified family must
% track the empirical CDF of a large sample closely (max deviation is a
% Kolmogorov-Smirnov-style statistic; 0.02 is generous for n = 200k yet
% catches any parameterization/sign-convention mistake immediately).
% ---------------------------------------------------------------------

function verify_fit_tracks_ecdf(testCase, X, family)
% Fits `family` (passed explicitly so this check is independent of any
% identification ambiguity between geometrically close families) and
% verifies the fitted CDF tracks the empirical CDF.
L4 = lmom(X, 4);
P = Parameter_estimation(X, family, L4(1), L4(2), L4(3)/L4(2), L4(4)/L4(2));
Xs = sort(X);
n = numel(Xs);
idx = (1000:1000:n)';
emp = idx / n;
fit = CDF_l(Xs(idx), family, P);
verifyLessThan(testCase, max(abs(fit(:) - emp(:))), 0.02);
end

function test_fit_normal(testCase)
verify_fit_tracks_ecdf(testCase, random('normal', 3, 2, 200000, 1), 'normal');
end

function test_fit_lognormal(testCase)
verify_fit_tracks_ecdf(testCase, ...
    random('lognormal', 0, 0.5, 200000, 1), 'lognormal');
end

function test_fit_gev(testCase)
verify_fit_tracks_ecdf(testCase, ...
    random('generalized extreme value', 0.2, 1, 0, 200000, 1), ...
    'generalized extreme value');
end

function test_fit_gamma_shifted(testCase)
% Exercises the shifted-gamma branch, including the CDF_l location-shift
% fix (CDF_l previously omitted the X - P(3) shift PDF_l applies).
verify_fit_tracks_ecdf(testCase, ...
    3 + random('gamma', 5, 0.8, 200000, 1), 'gamma');
end

% ---------------------------------------------------------------------
% Shifted-gamma PDF/CDF consistency (the historical CDF_l bug): the
% numerical derivative of CDF_l must match PDF_l on a shifted sample.
% ---------------------------------------------------------------------

function test_gamma_pdf_cdf_consistent(testCase)
X = 3 + random('gamma', 5, 0.8, 200000, 1);
[~, L] = Identify_dist(X);
P = Parameter_estimation(X, 'gamma', L(1), L(2), L(3), L(4));
xg = linspace(min(X)+0.1, max(X)-0.1, 500);
h = 1e-5;
dCDF = (CDF_l(xg+h, 'gamma', P) - CDF_l(xg-h, 'gamma', P)) / (2*h);
pdfv = PDF_l(xg, 'gamma', P);
verifyLessThan(testCase, max(abs(dCDF(:) - pdfv(:))), 1e-3);
end

% ---------------------------------------------------------------------
% Explicit three-parameter Weibull support (not auto-identified, but
% fully usable by name): round-trip parameters through PDF/CDF/Random.
% ---------------------------------------------------------------------

function test_weibull_explicit_roundtrip(testCase)
B_true = 2; A_true = 1.5; k_true = 1.8;
X = B_true + random('weibull', A_true, k_true, 200000, 1);
L = lmom(X, 4);
P = Parameter_estimation(X, 'weibul', L(1), L(2), L(3)/L(2), L(4)/L(2));
% Parameter = [A, k, B]
verifyLessThan(testCase, abs(P(1) - A_true), 0.1);
verifyLessThan(testCase, abs(P(2) - k_true), 0.15);
verifyLessThan(testCase, abs(P(3) - B_true), 0.1);
% CDF at the median of the sample should be ~0.5
med = median(X);
verifyLessThan(testCase, abs(CDF_l(med, 'weibul', P) - 0.5), 0.02);
% PDF must be the derivative of CDF
h = 1e-5;
dCDF = (CDF_l(med+h, 'weibul', P) - CDF_l(med-h, 'weibul', P)) / (2*h);
verifyLessThan(testCase, abs(dCDF - PDF_l(med, 'weibul', P)), 1e-4);
% Random_l draws must have approximately the right median
Z = Random_l('weibul', P, 100000, 1);
verifyLessThan(testCase, abs(median(Z) - med), 0.05);
end

% ---------------------------------------------------------------------
% Divergence functions.
% ---------------------------------------------------------------------

function test_jsdiv_zero_for_identical(testCase)
p = [0.2 0.3 0.5];
verifyLessThan(testCase, JSDiv(p, p), 1e-12);
end

function test_jsdiv_symmetric_bounded(testCase)
p = [0.7 0.2 0.1]; q = [0.1 0.2 0.7];
d1 = JSDiv(p, q); d2 = JSDiv(q, p);
verifyLessThan(testCase, abs(d1 - d2), 1e-12);
verifyGreaterThan(testCase, d1, 0);
verifyLessThan(testCase, d1, 1);  % bounded by log(2) in nats / 1 bit
end

% ---------------------------------------------------------------------
% Domain guards added alongside the JSS submission preparation.
% ---------------------------------------------------------------------

function test_lognormal_negative_skew_errors(testCase)
X = (1:5)';   % T3 = 0 exactly
L = lmom(X, 4);
verifyError(testCase, ...
    @() Parameter_estimation(X, 'lognormal', L(1), L(2), L(3)/L(2), L(4)/L(2)), ...
    'LUQ:ParameterEstimation:lognormalNegativeSkew');
end

function test_degenerate_sample_errors(testCase)
X = 3.14 * ones(10, 1);
verifyError(testCase, ...
    @() Parameter_estimation(X, 'normal', 3.14, 0, NaN, NaN), ...
    'LUQ:ParameterEstimation:invalidL2');
end

% ---------------------------------------------------------------------
% fit_best: guarded fit with ranked fallback (MATLAB counterpart of the
% Python fit_best; see the paper's Algorithm 1).
% ---------------------------------------------------------------------

function test_fit_best_fallback(testCase)
% Near-symmetric sample whose nearest ratio-diagram family is lognormal
% with (slightly) negative L-skewness: fit_best must skip lognormal and
% return the next-ranked valid family, recording the skip.
X = [-0.1321, 0.6404, 0.1049, -0.5357, 0.3616, 1.3040, ...
      0.9471, -0.7037, -1.2654, -0.6233, 0.0413, -2.3250, ...
     -0.2188, -1.2459];
[fam, P, skipped] = fit_best(X);
verifyNotEqual(testCase, fam, 'lognormal');
verifyNotEmpty(testCase, P);
verifyTrue(testCase, any(strcmp(skipped(:, 1), 'lognormal')));
end

function test_fit_best_clean_sample(testCase)
X = random('normal', 3, 2, 5000, 1);
[fam, P, skipped] = fit_best(X);
verifyNotEmpty(testCase, fam);
verifyNotEmpty(testCase, P);
verifyEmpty(testCase, skipped);
end

% ---------------------------------------------------------------------
% Bootstrap identification (uncertainty-aware model selection).
% ---------------------------------------------------------------------

function test_bootstrap_frequencies_valid(testCase)
X = random('gamma', 2, 1, 15, 1);
r = Identify_dist_bootstrap(X, 400);
verifyEqual(testCase, sum(r.frequencies), 1, 'AbsTol', 1e-9);
verifyTrue(testCase, all(r.frequencies >= 0));
verifyTrue(testCase, issorted(r.frequencies, 'descend'));
verifyTrue(testCase, any(strcmp({'clear','ambiguous'}, r.status)));
verifyLessThanOrEqual(testCase, r.t3_ci(1), r.t3_ci(2));
end

function test_bootstrap_large_logistic_is_clear(testCase)
X = random('logistic', 1.0, 0.7, 2000, 1);
r = Identify_dist_bootstrap(X, 400);
verifyEqual(testCase, r.best, 'logistic');
verifyEqual(testCase, r.families{1}, 'logistic');
verifyGreaterThan(testCase, r.frequencies(1), 0.8);
verifyEqual(testCase, r.status, 'clear');
end

function test_bootstrap_scarce_wider_ci_than_large(testCase)
Xs = random('gamma', 2, 1, 12, 1);
Xl = random('gamma', 2, 1, 2000, 1);
rs = Identify_dist_bootstrap(Xs, 400);
rl = Identify_dist_bootstrap(Xl, 400);
verifyGreaterThan(testCase, diff(rs.t3_ci), 3 * diff(rl.t3_ci));
verifyGreaterThan(testCase, diff(rs.t4_ci), 3 * diff(rl.t4_ci));
end
