% Identify the top-K candidate distributions and estimate each one's
% parameters from the sample's L-moments.
%
%   [Distribution_type, Parameter, D_sorted, L_sample, skipped] = ...
%       parameter_identify(X, K)
%
% X                 - sample vector (NaN entries are dropped)
% K                 - number of feasible candidate families to fit
%                     (default 1)
%
% Distribution_type - 1xK cell of family names, closest first
% Parameter         - 1xK cell of parameter vectors, one per family
%                     (layouts as documented in Parameter_estimation.m)
% D_sorted          - 1xK ratio-diagram distances for those families
% L_sample          - [L1, L2, T3, T4] of the sample
% skipped           - Nx2 cell {family, reason} for candidates passed
%                     over because their closed-form estimator was
%                     undefined for this sample
%
% Candidates whose estimator domain excludes the sample are skipped and
% the search continues down the ranking, so parameter_identify(X, 1)
% agrees with fit_best(X). Fewer than K fits are returned only when the
% ranking is exhausted.
%
% This is the MATLAB counterpart of the Python parameter_identify and
% behaves the same way. Before version 2.0.0 this function looped over
% Identify_dist's output as though it returned K candidates; since
% Identify_dist only ever returns the single closest family, any call
% with K > 1 raised an index-out-of-bounds error, and K = 1 passed a
% 1x1 cell rather than a char to Parameter_estimation. Both are fixed
% here, and the output arguments have changed accordingly -- see NEWS.
%
% Copy right
% ADOPT Lab, IIT Madras, India.

function [Distribution_type, Parameter, D_sorted, L_sample, skipped] = ...
    parameter_identify(X, K)

if nargin < 2 || isempty(K)
    K = 1;
end

X = X(:);
X = X(~isnan(X));

% Same nine automatic-identification families, in the same order, as
% Identify_dist.m (Weibull is excluded there by design; request it
% explicitly via Parameter_estimation if needed).
dist = {'uniform','normal','exponential','gumbel','logistic',...
    'generalized extreme value',...
    'generalized pareto','lognormal','gamma'};

[~, L_sample, D] = Identify_dist(X);
L1 = L_sample(1); L2 = L_sample(2); T3 = L_sample(3); T4 = L_sample(4);

[Dsort, order] = sort(D(:));

Distribution_type = {};
Parameter = {};
D_sorted = [];
skipped = cell(0, 2);

for i = 1:numel(order)
    if numel(Distribution_type) >= K
        break
    end
    name = dist{order(i)};
    try
        P = Parameter_estimation(X, name, L1, L2, T3, T4);
        Distribution_type{end+1} = name;      %#ok<AGROW>
        Parameter{end+1} = P;                 %#ok<AGROW>
        D_sorted(end+1) = Dsort(i);           %#ok<AGROW>
    catch err
        skipped(end+1, :) = {name, err.message}; %#ok<AGROW>
    end
end

if isempty(Distribution_type)
    error('LUQ:parameterIdentify:noValidFamily', ...
        ['no supported family''s closed-form estimator is valid for this ', ...
         'sample''s L-moments (L1=%g, L2=%g, T3=%g, T4=%g)'], ...
        L1, L2, T3, T4);
end
end
