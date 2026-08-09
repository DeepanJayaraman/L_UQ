%% demo_octave.m
% Portable walkthrough of the L-UQ toolbox.
%
% Runs unmodified in GNU Octave and in MATLAB, and needs no additional
% packages: neither the Octave statistics package nor MATLAB's Statistics
% and Machine Learning Toolbox. demo_example.m is the richer MATLAB-only
% demo (it compares against fitdist's maximum-likelihood fit and is what
% the article's demo figure comes from); this file is what ships in the
% Octave package.
%
% Functions exercised: lmom, Identify_dist, Parameter_estimation,
% fit_best, parameter_identify, Identify_dist_bootstrap, PDF_l, CDF_l,
% Random_l, JSDiv, KLDiv, KSStat.
%
% Run:  demo_octave
%
% Background:
% Jayaraman D, Ramu P. L-moments-based uncertainty quantification for
% scarce samples including extremes. Structural and Multidisciplinary
% Optimization. 2021 Aug;64(2):505-39.
%
% Copy right
% ADOPT Lab, IIT Madras, India

clear; close all

% Octave and MATLAB seed their generators differently; both accept this.
rand('state', 7);   %#ok<RAND>

%% 1. A scarce sample containing one genuine extreme
%
% 12 draws from a lognormal parent, plus the maximum of 1e5 draws from
% the same parent -- a real ~99.999th-percentile event, not contamination
% (the extreme-generation scheme of Jayaraman & Ramu 2021).

true_mu = 0; true_sigma = 0.5;
X = Random_l('lognormal', [true_mu, true_sigma, 0], 12, 1);
X(end+1) = max(Random_l('lognormal', [true_mu, true_sigma, 0], 100000, 1));

fprintf('\n=== 1. Scarce sample ===\n');
fprintf('n = %d, min = %.4f, max = %.4f\n', numel(X), min(X), max(X));

%% 2. Sample L-moments
%
% L1 and L2 are location and scale; the ratios T3 (L-skewness) and T4
% (L-kurtosis) are the shape summaries identification works with.

L = lmom(X, 4);
fprintf('\n=== 2. Sample L-moments ===\n');
fprintf('L1 = %.4f  L2 = %.4f  L3 = %.4f  L4 = %.4f\n', L(1), L(2), L(3), L(4));
fprintf('T3 = %.4f  T4 = %.4f\n', L(3)/L(2), L(4)/L(2));

%% 3. Identification on the L-moment ratio diagram

[Distribution_type, L_sample, D] = Identify_dist(X);
fprintf('\n=== 3. Ratio-diagram identification ===\n');
fprintf('closest family: %s\n', Distribution_type{1});

dist_names = {'uniform','normal','exponential','gumbel','logistic', ...
    'generalized extreme value','generalized pareto','lognormal','gamma'};
[sorted_d, order] = sort(D(:));
fprintf('ranking (distance):\n');
for i = 1:numel(order)
    fprintf('  %d. %-26s %.6f\n', i-1, dist_names{order(i)}, sorted_d(i));
end

%% 4. Guarded fit with ranked fallback
%
% fit_best walks that ranking and returns the first family whose
% closed-form estimator is actually valid for this sample, reporting any
% it had to skip.

[Distribution, Parameter, skipped, L_sample] = fit_best(X);
fprintf('\n=== 4. fit_best ===\n');
fprintf('fitted family: %s\n', Distribution);
fprintf('parameters   : %s\n', mat2str(Parameter, 6));
if isempty(skipped)
    fprintf('skipped      : none (closest family was valid)\n');
else
    fprintf('skipped:\n');
    for i = 1:size(skipped, 1)
        fprintf('  %s: %s\n', skipped{i,1}, skipped{i,2});
    end
end

%% 5. Competing candidates
%
% parameter_identify fits more than one candidate, so families that sit
% close together on the diagram can be compared rather than silently
% collapsed to the single nearest one.

fprintf('\n=== 5. Top-3 candidates ===\n');
[names_k, params_k, d_k] = parameter_identify(X, 3);
for i = 1:numel(names_k)
    fprintf('  %-26s d=%.6f  %s\n', names_k{i}, d_k(i), mat2str(params_k{i}, 6));
end
% Each candidate can be scored against the reference sample on the same
% footing -- see section 8 below for the reference.

%% 6. How certain is the identification?
%
% For a scarce sample the ratio-diagram position is itself uncertain.
% Bootstrapping shows how often each family would have been chosen.

fprintf('\n=== 6. Bootstrap identification ===\n');
boot = Identify_dist_bootstrap(X, 500);
fprintf('point estimate : %s\n', boot.best);
fprintf('status         : %s\n', boot.status);
fprintf('t3 95%% CI      : [%.4f, %.4f]\n', boot.t3_ci(1), boot.t3_ci(2));
fprintf('t4 95%% CI      : [%.4f, %.4f]\n', boot.t4_ci(1), boot.t4_ci(2));
fprintf('selection frequencies:\n');
for i = 1:numel(boot.families)
    if boot.frequencies(i) > 0
        fprintf('  %-26s %5.1f%%\n', boot.families{i}, 100*boot.frequencies(i));
    end
end

%% 7. Evaluating the fitted distribution

xgrid = linspace(max(min(X)-1, 1e-6), max(X)*1.05, 400);
pdf_L = PDF_l(xgrid, Distribution, Parameter);
cdf_L = CDF_l(xgrid, Distribution, Parameter);

fprintf('\n=== 7. Fitted distribution ===\n');
fprintf('P(X <= median of sample) = %.4f\n', ...
    CDF_l(median(X), Distribution, Parameter));
fprintf('PDF integrates to %.4f over the plotting grid\n', trapz(xgrid, pdf_L));

%% 8. Goodness of fit against a large reference sample
%
% The scarce fit is scored against 20,000 draws from the true parent --
% the situation the toolbox is designed for, where a small sample must
% stand in for a population that is expensive to observe.
%
% Note that JSDiv and KSStat take the fit and the raw data directly: no
% histogram has to be built by hand.

X_reference = Random_l('lognormal', [true_mu, true_sigma, 0], 20000, 1);

fprintf('\n=== 8. Goodness of fit vs. a 20,000-point reference ===\n');
fprintf('JS divergence : %.4f\n', JSDiv(Distribution, Parameter, X_reference));
fprintf('KL divergence : %.4f\n', KLDiv(Distribution, Parameter, X_reference));
fprintf('KS statistic  : %.4f\n', KSStat(Distribution, Parameter, X_reference));

%% 9. Plots
%
% hist/stairs/plot only -- all available in both Octave and MATLAB.

figure('Name', 'L-UQ demo');

subplot(1,2,1)
[counts, centers] = hist(X, 8);
width = centers(2) - centers(1);
bar(centers, counts / (numel(X) * width), 1.0); hold on
plot(xgrid, pdf_L, 'LineWidth', 2)
xlabel('X'); ylabel('PDF'); title(['PDF fit: ' Distribution])
legend('sample histogram', 'L-moment fit', 'location', 'northeast')

subplot(1,2,2)
Xs = sort(X(:));
stairs(Xs, (1:numel(Xs))'/numel(Xs), 'k'); hold on
plot(xgrid, cdf_L, 'LineWidth', 2)
xlabel('X'); ylabel('CDF'); title(['CDF fit: ' Distribution])
legend('empirical CDF', 'L-moment fit', 'location', 'southeast')

fprintf('\nDone.\n');
