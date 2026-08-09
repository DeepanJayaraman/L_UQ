% Kolmogorov-Smirnov statistic between a fitted family and a raw sample.
%
%   D = KSStat(Name, Parameter, Data)
%
% Name      - fitted family name
% Parameter - its parameter vector (see Parameter_estimation.m)
% Data      - raw sample the fit is being compared against
%
% D = sup_x |F_fit(x) - F_n(x)|, the largest absolute gap between the
% fitted CDF and the sample's empirical CDF, evaluated exactly at the
% order statistics -- checking both the left and right limits of the step
% function, since the supremum is attained at one of them.
%
% Unlike JSDiv this needs no binning, so it carries no bin-width tuning
% parameter. It is useful as a check that a JS-divergence comparison is
% not an artifact of the chosen bins.
%
% MATLAB counterpart of the Python ks_stat(fit, data) / fit.ks_stat(data).
% No p-value is returned: when the parameters were estimated from the same
% data the null distribution of D no longer holds, and reporting a
% nominal p-value would be misleading. Use D descriptively, or validate
% against an independent reference sample.
%
% Copy right
% ADOPT Lab, IIT Madras, India

function D = KSStat(Name, Parameter, Data)

if nargin < 3
    error('LUQ:KSStat:nargin', 'usage: KSStat(Name, Parameter, Data)');
end

X = Data(:);
X = X(~isnan(X));
n = numel(X);
if n == 0
    error('LUQ:KSStat:empty', ...
        'no finite observations to compare the fit against');
end

Xs = sort(X);
F = luq_dist('cdf', Name, Parameter, Xs);
F = F(:);
F(~isfinite(F)) = 0;

upper = (1:n)' / n - F;      % empirical CDF just after each point
lower = F - (0:n-1)' / n;    % empirical CDF just before each point
D = max(max(upper), max(lower));
