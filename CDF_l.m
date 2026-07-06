% CDF estimation

% Reference
% 1. J.R.M. Hosking, L-moments: analysis and estimation of distributions using linear combinations of order statistics
% 2. J.R.M. Hosking , J.R. Wallis,Regional Frequency Analysis: An approach based on L-moments.

% X - sample
% Name - Distribution name
% Parameter - Distribution parameter

% Copy right
% ADOPT Lab, IIT Madras, India

function CDF = CDF_l(X,Name,Parameter)
P = Parameter;
if strcmp(Name, 'exponential')
    CDF = cdf(Name,X,P(1));
end
if any(strcmp(Name,{'uniform','normal','logistic'}))
    CDF = cdf(Name,X,P(1),P(2));
end
if any(strcmp(Name,{'generalized extreme value','generalized pareto'}))
    CDF = cdf(Name,X,P(1),P(2),P(3));
end
if strcmp(Name,'lognormal')
    X = X - P(3);
    CDF = cdf('lognormal',X,P(1),P(2));
end
if strcmp(Name,'gumbel')
    CDF = cdf('generalized extreme value',X,0,P(1),P(2));
end
if strcmp(Name,'gamma')
    % Same location shift as PDF_l.m's gamma branch, so PDF and CDF are
    % consistent for shifted-Gamma fits (Parameter = [Alpha,Beta,min(X)]).
    X = X-P(3)+eps;
    CDF = cdf('gamma',X,P(1),P(2));
end
if strcmp(Name,'weibul')
    % Parameter = [A,k,B]: A - scale, k - shape, B - location.
    % MATLAB's 'weibull' takes (scale, shape); the location shift makes
    % this the three-parameter Weibull used by Parameter_estimation.m.
    CDF = cdf('weibull',X-P(3),P(1),P(2));
end