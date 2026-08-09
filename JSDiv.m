% JS divergence with log2 base (range [0,1])
%
% Two calling conventions are supported.
%
% 1. Fit against raw data -- the common case when validating a
%    scarce-sample fit against a larger reference sample. Binning is
%    handled internally:
%
%       [Distribution, Parameter] = fit_best(X_scarce);
%       d = JSDiv(Distribution, Parameter, X_full);
%       d = JSDiv(Distribution, Parameter, X_full, nbins);
%
%    This mirrors the Python API's js_div(fit, x_full) / fit.js_div(x_full),
%    which needs no manual histogram construction either.
%
% 2. Two binned mass vectors -- the low-level form, unchanged from
%    earlier releases:
%
%       d = JSDiv(P, Q)
%
%    P and Q are automatically normalised to sum to one over rows.
%    P = n x nbins, Q = 1 x nbins, dist = n x 1.
%
% The default binning in form 1 is 39 equal-width bins spanning the range
% of the data, with 1e-12 added to every bin so an empty bin cannot make
% the divergence infinite -- the same convention as the Python port, so
% the two implementations return identical values.
%
% Copy right
% Adopt Lab, IIT madras, India

function dist = JSDiv(varargin)

if nargin >= 3 && ischar(varargin{1})
    % Form 1: JSDiv(Name, Parameter, Data [, nbins])
    Name = varargin{1};
    Parameter = varargin{2};
    Data = varargin{3};
    if nargin >= 4
        nbins = varargin{4};
    else
        nbins = 39;
    end
    [P, Q] = luq_bin_fit(Name, Parameter, Data, nbins);
elseif nargin == 2
    % Form 2: JSDiv(P, Q)
    P = varargin{1};
    Q = varargin{2};
    if size(P,2) ~= size(Q,2)
        error('the number of columns in P and Q should be the same');
    end
else
    error('LUQ:JSDiv:usage', ...
        ['usage: JSDiv(P, Q) for binned mass vectors, or ', ...
         'JSDiv(Name, Parameter, Data [, nbins]) for a fit against data']);
end

% normalizing the P and Q
Q = Q ./ sum(Q);
Q = repmat(Q, [size(P,1) 1]);
P = P ./ repmat(sum(P,2), [1 size(P,2)]);
M = 0.5 .* (P + Q);
dist = 0.5 .* KLDiv(P,M) + 0.5 * KLDiv(Q,M);
