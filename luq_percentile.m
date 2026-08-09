% Percentile of a sample by linear interpolation between order statistics.
%
%   V = luq_percentile(X, P)
%
% X - data vector (NaN entries are dropped)
% P - percentile in [0, 100], scalar or vector
% V - interpolated percentile(s), same shape as P
%
% Replaces the Statistics toolbox / statistics package function prctile,
% so that Identify_dist_bootstrap runs on a bare MATLAB or Octave
% installation.
%
% Note that this is *not* a drop-in reimplementation of prctile: it uses
% the linear interpolation convention of numpy.percentile, placing the
% i-th of n order statistics at percentile 100*(i-1)/(n-1), whereas
% prctile places it at 100*(i-0.5)/n. The numpy convention is used here
% so that the bootstrap intervals reported by the MATLAB, Octave and
% Python implementations agree; against prctile the difference is of
% order one order statistic and vanishes as the number of bootstrap
% resamples grows.
%
% Copy right
% ADOPT Lab, IIT Madras, India

function V = luq_percentile(X, P)

X = X(:);
X = X(~isnan(X));
n = numel(X);
if n == 0
    error('LUQ:percentile:empty', 'no non-NaN observations');
end
if any(P(:) < 0 | P(:) > 100)
    error('LUQ:percentile:range', 'percentiles must lie in [0, 100]');
end

Xs = sort(X);
if n == 1
    V = repmat(Xs, size(P));
    return
end

% Work in a column internally: Xs is a column, so indexing it with a row
% vector of positions would otherwise broadcast into a matrix.
Pc = P(:);
H = (n - 1) * (Pc / 100);         % 0-based position in the sorted sample
Lo = min(floor(H), n - 2);        % keep Lo+2 a valid index at P = 100
Frac = H - Lo;

V = Xs(Lo + 1) + Frac .* (Xs(Lo + 2) - Xs(Lo + 1));
V = reshape(V, size(P));
