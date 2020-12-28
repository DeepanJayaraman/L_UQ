% Parameter estimation

% Reference
% 1. J.R.M. Hosking, L-moments: analysis and estimation of distributions using linear combinations of order statistics
% 2. J.R.M. Hosking , J.R. Wallis,Regional Frequency Analysis: An approach based on L-moments.
% X - Sample
% Distribution - Approximated distribution from sample
% L1, L2 - L-moments
% T3,T4 - L-moment ratios
% Copy right
% Deepan Jayaraman and Palaniappan ramu - ADOPT Lab, IIT Madras, India.

function [Parameter] = Parameter_estimation(X,Distribution,L1,L2,T3,T4)

dist = {'uniform','normal','exponential','gumbel','logistic',...
    'generalized extreme value',...
    'generalized pareto','lognormal','gamma','weibul'};

% D = ~cellfun('isempty',(strfind(dist,Distribution)));
D = strcmp(dist,Distribution);
Position = find(D);
if Position == 1    % Uniform distribution
    Alpha = L1-3*L2;
    Beta = L1+3*L2;
    Parameter = [Alpha,Beta];
end
if Position == 2    % Normal distribution
    Mean = L1;
    Standard_deviation = sqrt(pi)*L2;
    Parameter = [Mean,Standard_deviation];
end
if Position == 3    % Exponential
    Alpha = L1;
    Parameter = L1;
    %     Alpha = 2*L2;
    %     Eta = L1-Alpha; % Lower endpoint of the distribution
    %     Parameter = [Alpha,Eta];
end
if Position == 4    % Gumbel or Type I extreme value distribution (k,hape parameter = 0)
    Alpha = L2/log(2);  % Scale parameter
    G =  0.5772156649;  % Euler constant
    Eta = L1- G*Alpha ; % Location parameter
    Parameter = [Alpha,Eta];
end
if Position == 5    % Logistic
    Alpha = L2;
    Eta = L1;
    Parameter = [Eta,Alpha];
end
if Position == 6    % Generalized extreme value type II & III
    % only for -2 > k > 0.8
    z = (2/(3+T3))-(log(2)/log(3));
    k = 7.8590*z+2.9554*z^2; % Shape
    Alpha = L2*k/((1-2^(-k))*gamma(1+k)); %Scale
    Eta = L1+((Alpha*(gamma(1+k)-1))/k); % Loacation
    % negative k is taken as shape parameter from sample tests conducted
    Parameter = [-k,Alpha,Eta];
end
if Position == 7    % Generalized pareto
    % to find eta
    % Xnew = X-min(X); % GP parameter estimation consider the start point as 0, k=0
    % Lnew = lmom(Xnew,1);
    % Startpoint = L1-Lnew; % this subtraction approximatly gives the start point of GP
    % Estimated
    % k = (Lnew/L2)-2;
    % Alpha = L2*(1+k)*(2+k); % L1 = (2+k)*L2
    % Eta = L1-Alpha/(1+k); % always zero
    % negative k is taken as shape parameter from sample tests conducted
    % Parameter = [-k,Alpha,Startpoint];
    
    % If eta(Startpoint) is unknown
    k = (1-3*T3)/(1+T3);
    Alpha = L2*(1+k)*(2+k);
    Eta = L1 - (2+k)*L2; % Startpoint
    % negative k is taken as shape parameter from sample tests conducted
    Parameter = [-k,Alpha,Eta];
    
end
if Position == 8    % Lognormal distribution
    Z = sqrt(8/3)*norminv((1+T3)/2);
    Sigma = 0.999281*Z-0.006118*Z^3+0.000127*Z^5;
    Mu = log(L2/erf(Sigma/2))-Sigma^2/2;
    Eta = L1-exp(Mu+Sigma^2/2);
    Parameter = [Mu,Sigma,Eta];
    
    %     Wrong from ref 2
    %     E0 = 2.0466534; E1 = -3.6544371; E2 = 1.8396733;
    %     E3 = -0.20360244; F1 = -2.0182173; F2 = 1.2420401;
    %     F3 = -0.21741801;
    %
    %     k = -T3*(E0+E1*T3^2+E2*T3^4+E3*T3^6)/(1+F1*T3^2+F2*T3^4+F3*T3^6); % Shape
    %     Alpha = L2*k*exp(-k^2/2)/(1-2*norm(-k/sqrt(2))); %Scale
    %     Eta = L1-(Alpha/k)*(1-exp(k^2/2));
    %     Parameter = [Eta,Alpha,k];
end

if Position == 9   % Gamma or  Generalized Pearson type III
    Xnew = X-min(X)+eps; % start point or location
    L1 = lmom(Xnew,1); % No change in L2,L3,L4
    t = L2/L1;
    if t>0 && t<0.5
        z=pi*t^2;
        Alpha = (1-0.3080*z)/(z-0.05812*z^2+0.01765*z^3); % shape
    end
    if t>=0.5 && t<1
        z = 1-t;
        Alpha = (0.7213*z-0.5947*z^2)/(1-2.1817*z+1.2113*z^2);
    end
    Beta = L1/Alpha; % Scale
    Parameter = [Alpha,Beta,min(X)];
    
    % If eta(Startpoint) is unknown
    
    %     if abs(T3)>0 && abs(T3)<1/3
    %         z = 3*pi*T3^2;
    %         Alpha = (1+0.2906*z)/(z+0.1882*z^2+0.0442*z^3);
    %     else
    %         z = 1-abs(T3);
    %         Alpha = (0.36067*z-0.59567*z^2+0.25361*z^3)/(1-2.78861*z+2.56096*z^2-0.77045*z^3);
    %     end
    %     Gamma = 2*Alpha^(0.5)*sign(T3); % Shape
    %     Sigma = L2*(pi^0.5)*(Alpha^0.5)*gamma(Alpha)/gamma(Alpha+0.5); % Scale
    %     Mu = L1; % Location
    %
    %     if Gamma == 0
    %         Eta = 0;
    %     else
    %         Eta = Mu-2*Sigma/Gamma;
    %     end
    %
    %     Parameter = [Sigma,Mu,Eta,Gamma];
    
end
% if Position == 10    % Generalized Logistic
%     k = -T3;
%     Alpha = L2/(gamma(1+k)*gamma(1-k));
%     Eta = L1+(L2-Alpha)/k;
%     Parameter = [k,Alpha,Eta];
% end


if Position == 10 % weibul distribution
    
    % A - Scale
    % B - Location
    % k - Shape
    
    % in reference instead T3 they gave L3
    k = 285.3*T3^6-658.6*T3^5+622.8*T3^4-...
        317.2*T3^3+98.52*T3^2-21.256*T3+3.516;
    A = L2/((1-2^(-1/k))*gamma(1+1/k));
    B = L1-A*gamma(1+1/k);
    Parameter = [A,k,B];
end