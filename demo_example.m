%% demo_example.m
% Illustrative example for the L-moments based Uncertainty Quantification
% (UQ) toolbox.
%
% Workflow demonstrated:
%   1. Draw a scarce sample (n = 12) from a known distribution and inject
%      one extreme value (outlier), mimicking field/experimental data
%      where only a few observations are available and one is anomalous.
%   2. Identify the best-fit distribution family from the L-moment ratio
%      diagram (Identify_dist.m) and estimate its parameters from
%      L-moments (Parameter_estimation.m).
%   3. Fit the same family to the data using conventional moments / MLE
%      (MATLAB's fitdist) for comparison.
%   4. Plot both fitted PDFs/CDFs against the sample and compute the
%      Jensen-Shannon divergence of each fit from the empirical
%      distribution, to quantify how much the extreme value distorts
%      each approach.
%
% Requires: MATLAB Statistics and Machine Learning Toolbox.
%
% Background:
% Jayaraman D, Ramu P. L-moments-based uncertainty quantification for
% scarce samples including extremes. Structural and Multidisciplinary
% Optimization. 2021 Aug;64(2):505-39.

clear; clc; close all
rng(7);

%% 1. Generate a scarce sample with one injected extreme value
n = 12;
true_mu = 0; true_sigma = 0.5;
X = random('lognormal', true_mu, true_sigma, n, 1);
% Injected extreme: the maximum of 1e5 draws from the same parent -- a
% genuine rare event (~the population's 99.999th percentile), matching
% the extreme-generation scheme of Jayaraman & Ramu (2021).
X(end+1) = max(random('lognormal', true_mu, true_sigma, 100000, 1));

%% 2. Identify distribution family and estimate parameters from L-moments
[Distribution_type, L_sample] = Identify_dist(X);
fitted_dist = Distribution_type{1};
L1 = L_sample(1); L2 = L_sample(2); T3 = L_sample(3); T4 = L_sample(4);
fprintf('L-moment ratio diagram identified distribution: %s\n', fitted_dist);

Param_L = Parameter_estimation(X, fitted_dist, L1, L2, T3, T4);

%% 3. Conventional-moments (MLE) fit of the same family, for comparison
% Note: MATLAB's 'ExtremeValue' models the smallest-extreme (min) case,
% while this toolbox's 'gumbel' models the largest-extreme (max) case;
% treat that particular comparison as approximate only.
name_map = containers.Map( ...
    {'uniform','normal','exponential','gumbel','logistic', ...
     'generalized extreme value','generalized pareto','lognormal','gamma'}, ...
    {'Uniform','Normal','Exponential','ExtremeValue','Logistic', ...
     'GeneralizedExtremeValue','GeneralizedPareto','Lognormal','Gamma'});
pd_conventional = fitdist(X, name_map(fitted_dist));

%% 4. Evaluate PDF/CDF over a grid and plot both fits against the sample
xgrid = linspace(max(min(X)-1,1e-6), max(X)*1.05, 400);
pdf_L    = PDF_l(xgrid, fitted_dist, Param_L);
pdf_conv = pdf(pd_conventional, xgrid);

figure('Name', 'L-moment vs conventional-moment fit');

subplot(1,2,1)
histogram(X, 'Normalization', 'pdf', 'FaceAlpha', 0.3); hold on
plot(xgrid, pdf_L, 'LineWidth', 2)
plot(xgrid, pdf_conv, '--', 'LineWidth', 2)
xline(X(end), ':k', 'extreme sample')
legend('Sample histogram', 'L-moment fit', 'Conventional-moment (MLE) fit', ...
    'Location', 'best')
xlabel('X'); ylabel('PDF'); title(['PDF fit: ' fitted_dist])

cdf_L    = CDF_l(xgrid, fitted_dist, Param_L);
cdf_conv = cdf(pd_conventional, xgrid);
[f_emp, x_emp] = ecdf(X);

subplot(1,2,2)
stairs(x_emp, f_emp, 'k'); hold on
plot(xgrid, cdf_L, 'LineWidth', 2)
plot(xgrid, cdf_conv, '--', 'LineWidth', 2)
legend('Empirical CDF', 'L-moment fit', 'Conventional-moment (MLE) fit', ...
    'Location', 'best')
xlabel('X'); ylabel('CDF'); title(['CDF fit: ' fitted_dist])

%% 5. Quantitative comparison via Jensen-Shannon divergence
edges  = linspace(max(min(X)-1,1e-6), max(X)*1.05, 30);
P_emp  = histcounts(X, edges);
P_L    = diff(CDF_l(edges, fitted_dist, Param_L));
P_conv = diff(cdf(pd_conventional, edges));

JSD_L    = JSDiv(P_emp, P_L);
JSD_conv = JSDiv(P_emp, P_conv);

fprintf('Jensen-Shannon divergence, empirical vs L-moment fit:       %.4f\n', JSD_L);
fprintf('Jensen-Shannon divergence, empirical vs conventional fit:   %.4f\n', JSD_conv);
