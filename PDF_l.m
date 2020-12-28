% PDF estimation

% Reference
% 1. J.R.M. Hosking, L-moments: analysis and estimation of distributions using linear combinations of order statistics
% 2. J.R.M. Hosking , J.R. Wallis,Regional Frequency Analysis: An approach based on L-moments.

% X - sample
% Name - Distribution name
% Parameter - Distribution parameter

% Copy right
% ADOPT Lab, IIT Madras, India

function PDF = PDF_l(X,Name,Parameter)
P = Parameter;
if strcmp(Name, 'exponential')
    Name = string(Name);
%     X = X-P(2);
    PDF = pdf(Name,X,P(1));
end
if any(strcmp(Name,{'uniform','normal','logistic'}))
    Name = string(Name);
    PDF = pdf(Name,X,P(1),P(2));
end
if any(strcmp(Name,{'generalized extreme value','generalized pareto'}))
    Name = string(Name);
    PDF = pdf(Name,X,P(1),P(2),P(3));
end
if strcmp(Name,'lognormal')
    X = X - P(3);
    PDF = pdf('lognormal',X,P(1),P(2));
end
if strcmp(Name,'gumbel')
    PDF = pdf('generalized extreme value',X,0,P(1),P(2));
end
if strcmp(Name,'gamma')
    X = X-P(3)+eps;
    PDF = pdf('gamma',X,P(1),P(2));
    % If start point unknown
    % Parameter = [Sigma,Mu,Eta,Gamma];
    %     if P(4)==0
    %         PDF = pdf('normal',P(2),P(1));
    %     elseif P(4)<0
    %         X = P(3)-X;
    %     PDF = pdf ('gamma',X,P(1),P(2));
    %     else
    %         X = X-P(3);
    %     PDF = pdf ('gamma',X,P(1),P(2));
    %     end
end
if strcmp(Name,'weibul')
    % A - Scale
    % B - Location
    % k - Shape
    % Parameter = [A,k,B]
    k = P(2);
    A = P(1);
    B = P(3);
    PDF = wblpdf3(X,k,A,B);
end