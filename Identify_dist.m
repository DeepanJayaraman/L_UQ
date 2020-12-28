% Distribution identification 

% Reference
% 1. J.R.M. Hosking, L-moments: analysis and estimation of distributions using linear combinations of order statistics
% 2. J.R.M. Hosking , J.R. Wallis,Regional Frequency Analysis: An approach based on L-moments.

% X - sample
% k - No of distributions to match

% Copy right
% ADOPT Lab, IIT Madras, India

function [Distribution_type,L_sample,D,D_min] = Identify_dist(X,k)


% L_S = [L-skewness L-kurtosis] obtained from sample
% L_A = [L-skewness L-kurtosis] obtained from L ratio diagram


dist = {'uniform','normal','exponential','gumbel','logistic',...
    'generalized extreme value',...
    'generalized pareto','lognormal','gamma','weibul'};

L_U = [0,0];            % Uniform [skewness, Kurtosis]
L_N = [0,0.1230];       % Normal [0,0.1226]
L_E = [1/3,1/6];        % Exponential
L_G = [0.1699,0.1504];  % Gumbel
L_L = [0,1/6];          % Logistic


% Compute L moments
L = lmom(X,4);
L1_S = L(1);
L2_S = L(2);
T3_S = L(3)/L(2);         % L - skewness
T4_S = L(4)/L(2);         % L - kurtosis
L_sample = [L1_S,L2_S,T3_S,T4_S];
% L_S = round([T3_S,T4_S],4);
L_S = [T3_S,T4_S];
% L_dist = round([L_U;L_N;L_E;L_G;L_L],4);
L_dist = [L_U;L_N;L_E;L_G;L_L];
% Preallocation
D = zeros(9,1);

for i = 1:length(L_dist)
    D(i) = norm (L_dist(i,:) - L_S); % Euclidean norm
end
%% Generalized distributions
% syms T3 T4 Euclidean_distance
% T3_S, T4_S known ratio
% T3, T4 unknown and T4 is a function of T3
options = optimoptions(@fmincon,'Display','off');
for j =1:4  % 5 if u need to use weibul
    i = j+5;
    if i==6
        [~,DD] = fmincon(@(T3)dist_euc(L_S,T3,j),0.1,[],[],[],[],-0.6,0.9,[],options);
    else
        [~,DD] = fmincon(@(T3)dist_euc(L_S,T3,j),0.1,[],[],[],[],-0.9,0.9,[],options);
    end
    %     [~,DD] = fminunc(@(T3)dist_euc(L_S,T3,j),0,options);
    D(i) = DD;
end
% if prod(sign(X(:))+eps == -1)<0
%     D([3,7,9])=1;
% end
D = round(D,4);
PP = find(D == min(D(:)));
% Position = PP(randperm(length(PP),1));
Position = PP(length(PP));
D_min = D(Position);
% [D_min,Position] = mink(D,3);
Distribution_type = dist(Position);
end

% Minimum distance function
function DD = dist_euc(L_S,T3,j)
L1 = [T3,T4(T3,j)];
L1 = round(L1,4);
DD = norm(L1-L_S);
    function T4 = T4(T3,j)
        Coeff = [0.107010000000000,0,0.122820000000000,0.122400000000000,0.107159752418765;0.110900000000000,0.201960000000000,0,0,-0.113630607247195;0.848380000000000,0.959240000000000,0.775180000000000,0.301150000000000,0.842338510249911;-0.0666900000000000,-0.200960000000000,0,0,0.0921732557597174;0.00567000000000000,0.0406100000000000,0.122790000000000,0.958120000000000,0.0427093600517927;-0.0420800000000000,0,0,0,-0.0155156500400596;0.0367300000000000,0,-0.136380000000000,-0.574880000000000,-0.0321156200448707;0,0,0,0,0.0363854870768234;0,0,0.113680000000000,0.193830000000000,0.0398640323858772];
        %  Coeff = [0.10701,0,0.12282,0.12240;0.11090,0.20196,0,0;
%             0.84838,0.95924,0.77518,0.30115;
%             -0.066690,-0.20096,0,0;0.0056700,0.040610,0.12279,0.95812;
%             -0.042080,0,0,0;0.036730,0,-0.13638,-0.57488;0,0,0,0;
%             0,0,0.11368,0.19383];
        A = Coeff(:,j);
        T4 = A(1)+A(2)*T3+A(3)*(T3^2)+A(4)*(T3^3)+...
            A(5)*(T3^4)+A(6)*(T3^5)+A(7)*(T3^6)+...
            A(8)*(T3^7)+A(9)*(T3^8);
    end
end
