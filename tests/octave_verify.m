% Octave-based verification of the MATLAB toolbox sources.
%
% MATLAB's unittest framework (functiontests/localfunctions, used by
% test_uq_matlab.m) is not available in GNU Octave, so this script
% re-implements the same coverage as a plain script that runs under a
% bare Octave installation -- no packages required. It exercises the
% *actual* .m sources shipped in this repository:
%
%   A. Machine-precision equivalence against the Python implementation
%      on the 7 fixed reference cases in JSS/comparison/reference_cases
%      (the same cases the R lmom comparison used): sample L-moments,
%      estimated parameters, and fitted CDF values must match the
%      stored Python outputs.
%   B. Identification: large inverse-CDF samples must point back to the
%      generating family (degenerate pairs accepted, same policy as the
%      MATLAB/Python suites).
%   C. Domain guards: invalid L-moments must raise errors, not NaN.
%   D. Divergence properties: JSDiv in [0,1], symmetric, zero on
%      identical inputs.
%   E. Explicit three-parameter Weibull: estimate/PDF/CDF/Random round
%      trip.
%
% Run:  octave --no-gui --eval "run('tests/octave_verify.m')"
%
% Octave is not MATLAB: a pass here strongly suggests (but does not
% prove) the code runs in MATLAB; the CI workflow's matlab-actions job
% remains the authoritative check.

1;  % script file marker

function check(cond, name)
  global n_pass n_fail failures
  if cond
    n_pass = n_pass + 1;
    printf("PASS  %s\n", name);
  else
    n_fail = n_fail + 1;
    failures{end+1} = name;
    printf("FAIL  %s\n", name);
  end
end

global n_pass n_fail failures
n_pass = 0; n_fail = 0; failures = {};

% No `pkg load statistics` here: since version 2.0.0 the toolbox depends
% only on functions core to both MATLAB and Octave. If any of these
% checks starts failing with "'xyz' undefined", a Statistics-toolbox call
% has crept back into the sources.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'));          % the toolbox .m files
% Reference cases ship with the repository so that this check runs from a
% bare clone. The sibling path is kept as a fallback for the development
% tree, where the cases are generated.
refdir = fullfile(here, 'reference_cases');
if ~exist(refdir, 'dir')
  refdir = fullfile(here, '..', '..', 'JSS', 'comparison', 'reference_cases');
end

printf("Octave %s, no external packages loaded\n\n", OCTAVE_VERSION);

%% -------------------------------------------------------------- Part A --
printf("--- A. Equivalence vs. Python reference cases ---\n");
if ~exist(refdir, 'dir')
  printf("SKIP  reference_cases directory not found (%s)\n", refdir);
else
  cases = jsondecode(fileread(fullfile(refdir, 'index.json')));
  for ci = 1:numel(cases)
    stem = cases{ci};
    ref = jsondecode(fileread(fullfile(refdir, [stem '_reference.json'])));
    xt = dlmread(fullfile(refdir, [stem '.csv']), ',', 1, 0);
    x = xt(:);
    fam = ref.family;

    L = lmom(x, 4);
    T3 = L(3)/L(2); T4 = L(4)/L(2);
    ok_lmom = abs(L(1)-ref.L1) < 1e-9 && abs(L(2)-ref.L2) < 1e-9 && ...
              abs(T3-ref.T3) < 1e-9 && abs(T4-ref.T4) < 1e-9;
    check(ok_lmom, sprintf("%s: lmom matches Python L1,L2,T3,T4", stem));

    P = Parameter_estimation(x, fam, L(1), L(2), T3, T4);
    ok_par = numel(P) == numel(ref.python_params) && ...
             max(abs(P(:) - ref.python_params(:))) < 1e-8;
    check(ok_par, sprintf("%s: Parameter_estimation matches Python", stem));

    try
      C = CDF_l(ref.qpoints(:)', fam, P);
      ok_cdf = max(abs(C(:) - ref.python_cdf_at_qpoints(:))) < 1e-8;
      check(ok_cdf, sprintf("%s: CDF_l matches Python at 5 qpoints", stem));
    catch err
      check(false, sprintf("%s: CDF_l errored: %s", stem, err.message));
    end
  end
end

%% -------------------------------------------------------------- Part B --
printf("\n--- B. Identification on large samples ---\n");
rand('state', 12345); randn('state', 12345);
N = 200000;

% Normal/gamma is a degenerate pair: the zero-skew limit of the shifted
% gamma (Pearson III) IS the normal, so its curve passes through the
% normal point and a symmetric sample can land closer to either.
X = 3 + 2*randn(N,1);
D = Identify_dist(X);
check(any(strcmp(D{1}, {'normal','gamma'})), ...
      "identify normal (or gamma, degenerate pair)");

X = -2.5*log(rand(N,1));
D = Identify_dist(X);
check(strcmp(D{1}, 'exponential'), "identify exponential");

X = 1.0 - 0.5*log(-log(rand(N,1)));   % Gumbel(eta=1, alpha=0.5)
D = Identify_dist(X);
check(any(strcmp(D{1}, {'gumbel','generalized extreme value'})), ...
      "identify gumbel (or GEV, degenerate pair)");

U = rand(N,1);
X = 1.0 + 0.7*log(U./(1-U));          % logistic
D = Identify_dist(X);
check(strcmp(D{1}, 'logistic'), "identify logistic");

X = 2 + 3*rand(N,1);                  % uniform(2,5)
D = Identify_dist(X);
check(any(strcmp(D{1}, {'uniform','generalized pareto'})), ...
      "identify uniform (or GP, degenerate pair)");

X = exp(0.0 + 0.5*randn(N,1));        % lognormal
D = Identify_dist(X);
check(any(strcmp(D{1}, {'lognormal','generalized extreme value','gamma'})), ...
      "identify lognormal (or adjacent curve)");

%% -------------------------------------------------------------- Part C --
printf("\n--- C. Domain guards raise errors ---\n");
% Near-symmetric sample forced through the lognormal estimator: T3 <= 0.
x = [1 2 3 4 5 6 7 8 9 10];
L = lmom(x, 4); T3 = L(3)/L(2); T4 = L(4)/L(2);
guard_hit = false;
try
  Parameter_estimation(x, 'lognormal', L(1), L(2), -0.05, T4);
catch
  guard_hit = true;
end
check(guard_hit, "lognormal guard: error on T3 <= 0 (not NaN)");

guard_hit = false;
try
  Parameter_estimation(x, 'normal', L(1), -1.0, T3, T4);
catch
  guard_hit = true;
end
check(guard_hit, "global guard: error on L2 <= 0");

%% -------------------------------------------------------------- Part D --
printf("\n--- D. Divergence properties ---\n");
p = [0.2 0.3 0.4 0.1]; q = [0.25 0.25 0.25 0.25];
j1 = JSDiv(p, q); j2 = JSDiv(q, p);
check(abs(j1 - j2) < 1e-12, "JSDiv symmetric");
check(j1 >= 0 && j1 <= 1, "JSDiv in [0, 1]");
check(abs(JSDiv(p, p)) < 1e-12, "JSDiv(p, p) = 0");
check(KLDiv(p, p) < 1e-12, "KLDiv(p, p) = 0");

%% -------------------------------------------------------------- Part E --
printf("\n--- E. Explicit three-parameter Weibull ---\n");
A_true = 2.0; k_true = 1.5; B_true = 10.0;
U = rand(N,1);
X = B_true + A_true * (-log(1-U)).^(1/k_true);
L = lmom(X, 4); T3 = L(3)/L(2); T4 = L(4)/L(2);
P = Parameter_estimation(X, 'weibul', L(1), L(2), T3, T4);
check(abs(P(1) - A_true) < 0.1 && abs(P(2) - k_true) < 0.1 && ...
      abs(P(3) - B_true) < 0.1, ...
      sprintf("weibul round-trip: est [%.3f %.3f %.3f] vs true [2 1.5 10]", ...
              P(1), P(2), P(3)));
try
  grid = linspace(B_true + 0.01, B_true + 8, 50);
  pv = PDF_l(grid, 'weibul', P);
  cv = CDF_l(grid, 'weibul', P);
  rv = Random_l('weibul', P, 1000, 1);
  check(all(isfinite(pv)) && all(diff(cv) >= -1e-12) && ...
        all(isfinite(rv)) && min(rv) >= B_true - 0.01, ...
        "weibul PDF_l/CDF_l/Random_l branches work");
catch err
  check(false, sprintf("weibul PDF/CDF/Random errored: %s", err.message));
end

%% -------------------------------------------------------------- Part F --
printf("\n--- F. fit_best ranked fallback ---\n");
% The paper's Section 4.3 sample: nearest family is lognormal but its
% L-skewness is (slightly) negative, so fit_best must skip it and fall
% back to the next-ranked valid family.
x = [-0.1321, 0.6404, 0.1049, -0.5357, 0.3616, 1.3040, ...
      0.9471, -0.7037, -1.2654, -0.6233, 0.0413, -2.3250, ...
     -0.2188, -1.2459];
try
  [fam, P, skipped, Ls] = fit_best(x);
  check(~strcmp(fam, 'lognormal') && ~isempty(P), ...
        sprintf("fit_best falls back past lognormal (returned %s)", fam));
  check(~isempty(skipped) && any(strcmp(skipped(:,1), 'lognormal')), ...
        "fit_best records lognormal in skipped list");
catch err
  check(false, sprintf("fit_best errored: %s", err.message));
  check(false, "fit_best skipped-list check unreachable");
end
% And on a well-behaved sample it should return rank-1 with nothing skipped.
X = 3 + 2*randn(5000, 1);
[fam, P, skipped] = fit_best(X);
check(~isempty(P) && isempty(skipped), ...
      sprintf("fit_best clean sample: %s, nothing skipped", fam));

%% -------------------------------------------------------------- Part G --
printf("\n--- G. bootstrap identification ---\n");
X = 1.0 + 0.7*log(rand(2000,1)./(1-rand(2000,1)));   % logistic
r = Identify_dist_bootstrap(X, 400);
check(abs(sum(r.frequencies) - 1) < 1e-9, "bootstrap frequencies sum to 1");
check(issorted(r.frequencies, 'descend'), "bootstrap frequencies sorted descending");
check(strcmp(r.families{1}, 'logistic') && r.frequencies(1) > 0.8 && ...
      strcmp(r.status, 'clear'), ...
      sprintf("bootstrap: large logistic is clear (%s %.2f)", ...
              r.families{1}, r.frequencies(1)));
Xs = 3 + 2*randn(12,1); Xl = 3 + 2*randn(2000,1);
rs = Identify_dist_bootstrap(Xs, 400); rl = Identify_dist_bootstrap(Xl, 400);
check(diff(rs.t3_ci) > 3*diff(rl.t3_ci), ...
      sprintf("bootstrap: scarce t3 CI (%.3f) >> large (%.3f)", ...
              diff(rs.t3_ci), diff(rl.t3_ci)));

%% -------------------------------------------------------------- Part H --
printf("\n--- H. Fit-aware divergences, KS statistic, percentile ---\n");

% A scarce fit scored against a large reference sample, the way the
% worked examples do it. JSDiv/KLDiv/KSStat take the fit and the raw
% sample directly; no histogram is built by hand.
rand('state', 4242); randn('state', 4242);
Xref = 3 + 2*randn(20000, 1);
Xsc  = 3 + 2*randn(25, 1);
[famH, PH] = fit_best(Xsc);

jd = JSDiv(famH, PH, Xref);
kd = KLDiv(famH, PH, Xref);
ks = KSStat(famH, PH, Xref);
check(isscalar(jd) && jd >= 0 && jd <= 1, ...
      sprintf("JSDiv(fit, data) in [0,1] (%.4f)", jd));
check(isscalar(kd) && isfinite(kd) && kd >= 0, ...
      sprintf("KLDiv(fit, data) finite and non-negative (%.4f)", kd));
check(isscalar(ks) && ks >= 0 && ks <= 1, ...
      sprintf("KSStat(fit, data) in [0,1] (%.4f)", ks));

% A fit of the right family against its own parent should beat a
% deliberately wrong one on both measures.
jd_wrong = JSDiv('exponential', [8.0], Xref);
ks_wrong = KSStat('exponential', [8.0], Xref);
check(jd < jd_wrong, ...
      sprintf("JSDiv prefers the fitted family (%.4f < %.4f)", jd, jd_wrong));
check(ks < ks_wrong, ...
      sprintf("KSStat prefers the fitted family (%.4f < %.4f)", ks, ks_wrong));

% The two-vector form must still work and agree with the manual binning
% the fit-and-data form performs internally.
[Pfit, Pdata] = luq_bin_fit(famH, PH, Xref, 39);
check(abs(JSDiv(Pdata, Pfit) - jd) < 1e-12, ...
      "JSDiv(fit, data) equals JSDiv on the same bins");

% KS statistic against the exact CDF of the sampled family is small for
% a large sample, and exactly zero-ish for a perfect match.
ks_exact = KSStat('normal', [3, 2], Xref);
check(ks_exact < 0.02, ...
      sprintf("KSStat of the true parent on n=20000 is small (%.4f)", ks_exact));

% luq_percentile against a known answer: for 1:100 the linear-
% interpolation convention puts the 50th percentile at 50.5.
check(abs(luq_percentile(1:100, 50) - 50.5) < 1e-12, ...
      "luq_percentile(1:100, 50) = 50.5");
check(abs(luq_percentile(1:100, 0) - 1) < 1e-12 && ...
      abs(luq_percentile(1:100, 100) - 100) < 1e-12, ...
      "luq_percentile endpoints exact");

% luq_dist: CDF monotone, PDF non-negative, and inv the inverse of cdf,
% across every supported family.
fams = {'uniform', 'normal', 'exponential', 'gumbel', 'logistic', ...
        'generalized extreme value', 'generalized pareto', ...
        'lognormal', 'gamma', 'weibul'};
pars = {[0.5 4], [3 2], [2.5], [1.7 4], [1 0.8], ...
        [0.25 1.5 3], [0.3 2 1], [0.4 0.7 2], [2.5 1.3 1], [3 1.8 0.5]};
qq = 0.05:0.05:0.95;
all_ok = true;
for fi = 1:numel(fams)
  xq = luq_dist('inv', fams{fi}, pars{fi}, qq);
  cq = luq_dist('cdf', fams{fi}, pars{fi}, xq);
  dq = luq_dist('pdf', fams{fi}, pars{fi}, xq);
  ok = all(isfinite(xq)) && all(abs(cq - qq) < 1e-8) && all(dq >= 0);
  if ~ok
    printf("      round-trip failed for %s\n", fams{fi});
    all_ok = false;
  end
end
check(all_ok, "luq_dist: cdf(inv(q)) = q and pdf >= 0 for all 10 families");

% parameter_identify returns K feasible candidates, closest first, and
% its first candidate agrees with fit_best.
[names_k, pars_k, d_k] = parameter_identify(Xsc, 3);
check(numel(names_k) == 3 && numel(pars_k) == 3 && numel(d_k) == 3, ...
      "parameter_identify returns K candidates");
check(issorted(d_k), "parameter_identify candidates ordered by distance");
check(strcmp(names_k{1}, famH), ...
      sprintf("parameter_identify(X,1) agrees with fit_best (%s)", names_k{1}));

%% ----------------------------------------------------------------------
printf("\n=== %d passed, %d failed ===\n", n_pass, n_fail);
if n_fail > 0
  for i = 1:numel(failures)
    printf("  FAILED: %s\n", failures{i});
  end
  error("octave_verify: %d check(s) failed", n_fail);
end
