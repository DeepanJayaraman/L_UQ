% Uncertainty-aware distribution identification via bootstrap.
%
% MATLAB counterpart of the Python identify_dist_bootstrap. Identify_dist
% reports a single closest family, but for scarce samples the sample
% ratios (t3, t4) carry substantial sampling uncertainty and several
% ratio-diagram loci lie close together, so the reported best family may
% not be statistically distinguishable from the runner-up. This routine
% resamples X with replacement Nboot times, re-identifies on each
% resample, and summarizes how often each family is selected, with
% percentile confidence intervals for (t3, t4) and an ambiguity flag.
%
%   result = Identify_dist_bootstrap(X)
%   result = Identify_dist_bootstrap(X, Nboot, clearFreq, clearMargin)
%
% X          - sample vector (NaN entries dropped)
% Nboot      - number of bootstrap resamples (default 1000)
% clearFreq  - min selection frequency for a 'clear' call (default 0.5)
% clearMargin- min lead over the 2nd family for 'clear' (default 0.15)
%
% result is a struct with fields:
%   best          - point-estimate closest family (Identify_dist(X))
%   families      - 1x9 cell of family names
%   frequencies   - 1x9 selection frequencies (same order as families),
%                   descending
%   status        - 'clear' or 'ambiguous'
%   t3_ci, t4_ci  - 1x2 [lo hi] 95% percentile bootstrap intervals
%   Nboot         - number of resamples that produced a valid fit
%
% Copy right
% ADOPT Lab, IIT Madras, India

function result = Identify_dist_bootstrap(X, Nboot, clearFreq, clearMargin)

if nargin < 2 || isempty(Nboot),       Nboot = 1000;      end
if nargin < 3 || isempty(clearFreq),   clearFreq = 0.5;   end
if nargin < 4 || isempty(clearMargin), clearMargin = 0.15; end

X = X(:);
X = X(~isnan(X));
n = numel(X);
if n < 4
    error('LUQ:IdentifyBootstrap:tooFew', ...
        'need at least 4 observations, got %d', n);
end

dist = {'uniform','normal','exponential','gumbel','logistic',...
    'generalized extreme value','generalized pareto','lognormal','gamma'};

pointD = Identify_dist(X);
best = pointD{1};

counts = zeros(1, numel(dist));
t3s = zeros(Nboot, 1);
t4s = zeros(Nboot, 1);
valid = 0;
for b = 1:Nboot
    Xb = X(randi(n, n, 1));
    try
        [Db, Lb] = Identify_dist(Xb);
    catch
        continue
    end
    t3b = Lb(3); t4b = Lb(4);
    if ~isfinite(t3b) || ~isfinite(t4b)
        continue
    end
    idx = find(strcmp(dist, Db{1}), 1);
    if isempty(idx), continue; end
    counts(idx) = counts(idx) + 1;
    valid = valid + 1;
    t3s(valid) = t3b;
    t4s(valid) = t4b;
end

if valid == 0
    error('LUQ:IdentifyBootstrap:allFailed', ...
        'all bootstrap resamples failed to identify a family');
end
t3s = t3s(1:valid);
t4s = t4s(1:valid);

freqs = counts / valid;
[freqs, order] = sort(freqs, 'descend');
families = dist(order);

topFreq = freqs(1);
secondFreq = 0; if numel(freqs) > 1, secondFreq = freqs(2); end
if topFreq >= clearFreq && (topFreq - secondFreq) >= clearMargin
    status = 'clear';
else
    status = 'ambiguous';
end

result = struct();
result.best = best;
result.families = families;
result.frequencies = freqs;
result.status = status;
result.t3_ci = [prctile(t3s, 2.5), prctile(t3s, 97.5)];
result.t4_ci = [prctile(t4s, 2.5), prctile(t4s, 97.5)];
result.Nboot = valid;
