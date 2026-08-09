% PDF estimation

% Reference
% 1. J.R.M. Hosking, L-moments: analysis and estimation of distributions using linear combinations of order statistics
% 2. J.R.M. Hosking , J.R. Wallis,Regional Frequency Analysis: An approach based on L-moments.

% X - sample
% Name - Distribution name
% Parameter - Distribution parameter

% Evaluation is delegated to luq_dist, which writes every family out in
% closed form. Earlier releases called MATLAB's name-dispatching pdf(),
% which requires the Statistics and Machine Learning Toolbox and has no
% portable equivalent in GNU Octave; the parameter layouts, location
% shifts and numerical results are unchanged.

% Copy right
% ADOPT Lab, IIT Madras, India

function PDF = PDF_l(X,Name,Parameter)

PDF = luq_dist('pdf', Name, Parameter, X);
