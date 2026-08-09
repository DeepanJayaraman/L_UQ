% CDF estimation

% Reference
% 1. J.R.M. Hosking, L-moments: analysis and estimation of distributions using linear combinations of order statistics
% 2. J.R.M. Hosking , J.R. Wallis,Regional Frequency Analysis: An approach based on L-moments.

% X - sample
% Name - Distribution name
% Parameter - Distribution parameter

% Evaluation is delegated to luq_dist, which writes every family out in
% closed form. Earlier releases called MATLAB's name-dispatching cdf(),
% which requires the Statistics and Machine Learning Toolbox and has no
% portable equivalent in GNU Octave; the parameter layouts, location
% shifts and numerical results are unchanged. In particular the gamma
% branch still applies the same location shift as PDF_l.m, so PDF and
% CDF stay consistent for shifted-Gamma fits.

% Copy right
% ADOPT Lab, IIT Madras, India

function CDF = CDF_l(X,Name,Parameter)

CDF = luq_dist('cdf', Name, Parameter, X);
