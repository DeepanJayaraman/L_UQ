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
if any(strcmp(Name,{'uniform','normal','logistic','gamma'}))
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
    CDF = cdf ('gamma',X,P(1),P(2));
end