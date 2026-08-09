% Reduce a fitted family and a raw sample to a comparable pair of binned
% probability vectors, for use by JSDiv and KLDiv.
%
%   [Pfit, Pdata] = luq_bin_fit(Name, Parameter, Data, nbins)
%
% Name      - fitted family name
% Parameter - its parameter vector (see Parameter_estimation.m)
% Data      - raw sample the fit is being compared against
% nbins     - number of equal-width bins spanning the data range
%
% Pfit      - probability mass the fit assigns to each bin
% Pdata     - histogram counts of Data in the same bins
%
% Both vectors have 1e-12 added to every bin, so a bin the fit assigns no
% probability to cannot make the divergence infinite. This is the same
% binning rule the Python port applies inside js_div, so MATLAB, Octave
% and Python return identical divergences for the same fit and data.
%
% Copy right
% ADOPT Lab, IIT Madras, India

function [Pfit, Pdata] = luq_bin_fit(Name, Parameter, Data, nbins)

if nargin < 4 || isempty(nbins)
    nbins = 39;
end
if nbins < 1
    error('LUQ:binFit:nbins', 'nbins must be at least 1, got %g', nbins);
end

Data = Data(:);
Data = Data(~isnan(Data));
if isempty(Data)
    error('LUQ:binFit:empty', ...
        'no finite observations to compare the fit against');
end

lo = min(Data);
hi = max(Data);
if ~(hi > lo)
    error('LUQ:binFit:degenerate', ...
        'cannot bin over a degenerate data range [%g, %g]', lo, hi);
end

edges = linspace(lo, hi, nbins + 1);

% Histogram counts without histcounts/histc, which differ between
% MATLAB versions and Octave: bin k collects lo + (k-1)*w <= x < lo + k*w,
% with the final bin closed at the top so max(Data) is counted.
w = (hi - lo) / nbins;
idx = floor((Data - lo) / w) + 1;
idx = min(max(idx, 1), nbins);
Pdata = zeros(1, nbins);
for k = 1:numel(idx)
    Pdata(idx(k)) = Pdata(idx(k)) + 1;
end

Cfit = luq_dist('cdf', Name, Parameter, edges);
Pfit = diff(Cfit(:).');
Pfit(~isfinite(Pfit)) = 0;
Pfit = max(Pfit, 0);

Pfit = Pfit + 1e-12;
Pdata = Pdata + 1e-12;
