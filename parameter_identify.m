% Parameter identification from sample
% X - sample
% K - Number of distributions (Helps to know possible distributions)
% Copy right
% ADOPT Lab, IIT Madras, India.


function [P,Distribution_type,L_sample,D,D_min,Parameter1] = parameter_identify(X,K)
X(isnan(X))=[];

[Distribution_type,L_sample,D,D_min] = Identify_dist(X,K);
L1= L_sample(1);L2 = L_sample(2);T3 = L_sample(3);T4 = L_sample(4);
for k = 1:K
    Parameter1 = Parameter_estimation(X,Distribution_type(k),L1,L2,T3,T4);
    C = max(size(Parameter1));
    switch C
        case 3
        P(1,k)=struct('P',Parameter1);
        case 2
        P(1,k)=struct('P',[Parameter1,0]);
        case 1
        P(1,k)=struct('P',[Parameter1,0,0]);
    end
end
end