% Random number generation

% Reference
% 1. J.R.M. Hosking, L-moments: analysis and estimation of distributions using linear combinations of order statistics
% 2. J.R.M. Hosking , J.R. Wallis,Regional Frequency Analysis: An approach based on L-moments.

% Name - Distribution name
% Parameter - Distribution parameter
% (m,n) - Size

% Copy right
% ADOPT Lab, IIT Madras, India

function X = Random_l(Name,Parameter,m,n)
P = Parameter;
if strcmp(Name, 'exponential')
    X = random(Name,P(1),m,n);
end
if any(strcmp(Name,{'uniform','normal','logistic'}))
    X = random(Name,P(1),P(2),m,n);
end
if any(strcmp(Name,{'generalized extreme value','generalized pareto'}))
    X = random(Name,P(1),P(2),P(3),m,n);
end
if strcmp(Name,'lognormal')
    Z = random('lognormal',P(1),P(2),m,n);
    X = Z+P(3);
end
if strcmp(Name,'gumbel')
    X = random('generalized extreme value',0,P(1),P(2),m,n);
end
if strcmp(Name,'gamma')
    Z = random (Name,P(1),P(2),m,n);
    X = Z+P(3)-eps;
end