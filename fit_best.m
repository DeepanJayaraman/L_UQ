% Guarded identification and estimation with ranked fallback.
%
% MATLAB counterpart of the Python fit_best() (see the paper's
% Algorithm 1): walks the L-moment ratio-diagram ranking from
% Identify_dist and returns the first family whose closed-form
% estimator is valid, skipping any family whose domain guard rejects
% the sample's L-moments (Parameter_estimation raises an error instead
% of returning NaN). Recommended over Identify_dist +
% Parameter_estimation for any unattended use.
%
%   [Distribution, Parameter, skipped, L_sample] = fit_best(X)
%
% X            - sample vector (NaN entries are dropped)
% Distribution - name of the first valid family, in ranking order
% Parameter    - its closed-form parameter estimate
% skipped      - N x 2 cell {family, reason} for every higher-ranked
%                family whose estimator domain was violated
% L_sample     - [L1, L2, T3, T4] of the sample
%
% Errors only if no supported family's estimator is valid.
%
% Copy right
% ADOPT Lab, IIT Madras, India

function [Distribution, Parameter, skipped, L_sample] = fit_best(X)

% Same nine automatic-identification families, in the same order, as
% Identify_dist.m (Weibull is excluded there by design; request it
% explicitly via Parameter_estimation if needed).
dist = {'uniform','normal','exponential','gumbel','logistic',...
    'generalized extreme value',...
    'generalized pareto','lognormal','gamma'};

X = X(:);
X = X(~isnan(X));

[~, L_sample, D] = Identify_dist(X);
L1 = L_sample(1); L2 = L_sample(2); T3 = L_sample(3); T4 = L_sample(4);

[~, order] = sort(D(:));

skipped = cell(0, 2);
for idx = order'
    name = dist{idx};
    try
        Parameter = Parameter_estimation(X, name, L1, L2, T3, T4);
        Distribution = name;
        return
    catch err
        skipped(end+1, :) = {name, err.message}; %#ok<AGROW>
    end
end

reasons = sprintf('%s: %s; ', skipped{:, 1}, skipped{:, 2});
error('LUQ:fitBest:noValidFamily', ...
    ['no supported family''s closed-form estimator is valid for this ', ...
     'sample''s L-moments (L1=%g, L2=%g, T3=%g, T4=%g); tried: %s'], ...
    L1, L2, T3, T4, reasons);
