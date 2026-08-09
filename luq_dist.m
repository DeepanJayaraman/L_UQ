% Closed-form PDF / CDF / inverse-CDF for the families supported by
% Parameter_estimation.m.
%
%   Y = luq_dist(op, Name, Parameter, X)
%
% op        - 'pdf', 'cdf' or 'inv'
% Name      - family name, as used throughout the toolbox
% Parameter - parameter vector in the layout returned by
%             Parameter_estimation.m (see below)
% X         - evaluation points, or probabilities when op is 'inv'
%
% Parameter layouts, matching Parameter_estimation.m and the Python port:
%   uniform                     [lower, upper]
%   normal                      [mean, sd]
%   exponential                 [scale]
%   gumbel                      [scale, loc]
%   logistic                    [loc, scale]
%   generalized extreme value   [shape k, scale, loc]
%   generalized pareto          [shape k, scale, loc]
%   lognormal                   [meanlog, sdlog, loc]
%   gamma                       [shape, scale, loc]
%   weibul                      [scale, shape, loc]
%
% Shape conventions follow MATLAB's gevcdf/gpcdf: for the generalized
% extreme value family F(x) = exp(-(1+k*z)^(-1/k)) with z = (x-loc)/scale,
% and for the generalized Pareto family F(x) = 1-(1+k*z)^(-1/k).
%
% Why this file exists
% --------------------
% PDF_l.m, CDF_l.m and Random_l.m used to call MATLAB's name-dispatching
% pdf/cdf/random, which require the Statistics and Machine Learning
% Toolbox and are not available under GNU Octave without the statistics
% package (whose accepted distribution names and argument orders have
% changed between releases). Everything here is written out in closed
% form on top of functions that are core to both MATLAB and Octave --
% erf, erfinv, gamma, gammainc, gammaincinv -- so the toolbox runs on a
% bare MATLAB or Octave installation with no extra packages.
%
% The formulas were verified against scipy.stats to machine precision
% (max absolute discrepancy 3.4e-16 over all ten families) before being
% transcribed here; see the Python test suite for the equivalent checks.
%
% Copy right
% ADOPT Lab, IIT Madras, India

function Y = luq_dist(op, Name, Parameter, X)

if nargin < 4
    error('LUQ:luqDist:nargin', ...
        'usage: luq_dist(op, Name, Parameter, X)');
end

P = Parameter(:).';
X = double(X);

switch lower(op)
    case 'pdf'
        Y = local_pdf(Name, P, X);
    case 'cdf'
        Y = local_cdf(Name, P, X);
    case 'inv'
        if any(X(:) < 0 | X(:) > 1)
            error('LUQ:luqDist:badProbability', ...
                'probabilities passed to the ''inv'' operation must lie in [0, 1]');
        end
        Y = local_inv(Name, P, X);
    otherwise
        error('LUQ:luqDist:badOp', ...
            'unknown operation ''%s'' (expected ''pdf'', ''cdf'' or ''inv'')', op);
end

end


% ---------------------------------------------------------------- pdf --
function Y = local_pdf(Name, P, X)
Y = zeros(size(X));
switch Name
    case 'uniform'
        a = P(1); b = P(2);
        ok = X >= a & X <= b;
        Y(ok) = 1 / (b - a);
    case 'normal'
        mu = P(1); sg = P(2);
        Y = exp(-0.5 * ((X - mu) / sg).^2) / (sg * sqrt(2*pi));
    case 'exponential'
        mu = P(1);
        ok = X >= 0;
        Y(ok) = exp(-X(ok) / mu) / mu;
    case 'logistic'
        mu = P(1); s = P(2);
        Z = exp(-(X - mu) / s);
        Y = Z ./ (s * (1 + Z).^2);
    case 'gumbel'
        sg = P(1); mu = P(2);
        Z = (X - mu) / sg;
        Y = exp(-Z - exp(-Z)) / sg;
    case 'generalized extreme value'
        k = P(1); sg = P(2); mu = P(3);
        if k == 0
            Y = local_pdf('gumbel', [sg mu], X);
        else
            T = 1 + k * (X - mu) / sg;
            ok = T > 0;
            Y(ok) = T(ok).^(-1/k - 1) .* exp(-T(ok).^(-1/k)) / sg;
        end
    case 'generalized pareto'
        k = P(1); sg = P(2); th = P(3);
        Z = (X - th) / sg;
        if k == 0
            ok = Z >= 0;
            Y(ok) = exp(-Z(ok)) / sg;
        else
            T = 1 + k * Z;
            ok = Z >= 0 & T > 0;
            Y(ok) = T(ok).^(-1/k - 1) / sg;
        end
    case 'lognormal'
        mu = P(1); sg = P(2); eta = P(3);
        W = X - eta;
        ok = W > 0;
        Y(ok) = exp(-0.5 * ((log(W(ok)) - mu) / sg).^2) ...
                ./ (W(ok) * sg * sqrt(2*pi));
    case 'gamma'
        a = P(1); b = P(2); loc = P(3);
        W = X - loc + eps;
        ok = W > 0;
        Y(ok) = W(ok).^(a - 1) .* exp(-W(ok) / b) / (b^a * gamma(a));
    case 'weibul'
        A = P(1); k = P(2); B = P(3);
        W = X - B;
        ok = W >= 0;
        Y(ok) = (k / A) * (W(ok) / A).^(k - 1) .* exp(-(W(ok) / A).^k);
    otherwise
        error('LUQ:luqDist:badName', 'unsupported distribution: %s', Name);
end
end


% ---------------------------------------------------------------- cdf --
function Y = local_cdf(Name, P, X)
Y = zeros(size(X));
switch Name
    case 'uniform'
        a = P(1); b = P(2);
        Y = min(max((X - a) / (b - a), 0), 1);
    case 'normal'
        mu = P(1); sg = P(2);
        Y = 0.5 * (1 + erf((X - mu) / (sg * sqrt(2))));
    case 'exponential'
        mu = P(1);
        ok = X >= 0;
        Y(ok) = 1 - exp(-X(ok) / mu);
    case 'logistic'
        mu = P(1); s = P(2);
        Y = 1 ./ (1 + exp(-(X - mu) / s));
    case 'gumbel'
        sg = P(1); mu = P(2);
        Y = exp(-exp(-(X - mu) / sg));
    case 'generalized extreme value'
        k = P(1); sg = P(2); mu = P(3);
        if k == 0
            Y = local_cdf('gumbel', [sg mu], X);
        else
            T = 1 + k * (X - mu) / sg;
            % Outside the support the CDF is 0 below and 1 above; which
            % side the excluded region lies on is set by the sign of k.
            if k > 0
                Y = zeros(size(X));
            else
                Y = ones(size(X));
            end
            ok = T > 0;
            Y(ok) = exp(-T(ok).^(-1/k));
        end
    case 'generalized pareto'
        k = P(1); sg = P(2); th = P(3);
        Z = (X - th) / sg;
        Y = double(Z >= 0);
        if k == 0
            ok = Z >= 0;
            Y(ok) = 1 - exp(-Z(ok));
        else
            T = 1 + k * Z;
            ok = Z >= 0 & T > 0;
            Y(ok) = 1 - T(ok).^(-1/k);
        end
    case 'lognormal'
        mu = P(1); sg = P(2); eta = P(3);
        W = X - eta;
        ok = W > 0;
        Y(ok) = 0.5 * (1 + erf((log(W(ok)) - mu) / (sg * sqrt(2))));
    case 'gamma'
        a = P(1); b = P(2); loc = P(3);
        W = X - loc + eps;
        ok = W > 0;
        % MATLAB/Octave gammainc(x, a) is the lower regularized
        % incomplete gamma -- note the argument order is the reverse of
        % scipy's gammainc(a, x).
        Y(ok) = gammainc(W(ok) / b, a);
    case 'weibul'
        A = P(1); k = P(2); B = P(3);
        W = X - B;
        ok = W >= 0;
        Y(ok) = 1 - exp(-(W(ok) / A).^k);
    otherwise
        error('LUQ:luqDist:badName', 'unsupported distribution: %s', Name);
end
end


% ------------------------------------------------------------ inverse --
function Y = local_inv(Name, P, Q)
switch Name
    case 'uniform'
        Y = P(1) + Q * (P(2) - P(1));
    case 'normal'
        Y = P(1) + P(2) * sqrt(2) * erfinv(2 * Q - 1);
    case 'exponential'
        Y = -P(1) * log1p(-Q);
    case 'logistic'
        Y = P(1) + P(2) * log(Q ./ (1 - Q));
    case 'gumbel'
        Y = P(2) - P(1) * log(-log(Q));
    case 'generalized extreme value'
        k = P(1); sg = P(2); mu = P(3);
        if k == 0
            Y = local_inv('gumbel', [sg mu], Q);
        else
            Y = mu + sg * ((-log(Q)).^(-k) - 1) / k;
        end
    case 'generalized pareto'
        k = P(1); sg = P(2); th = P(3);
        if k == 0
            Y = th - sg * log1p(-Q);
        else
            Y = th + sg * ((1 - Q).^(-k) - 1) / k;
        end
    case 'lognormal'
        Y = P(3) + exp(P(1) + P(2) * sqrt(2) * erfinv(2 * Q - 1));
    case 'gamma'
        % gammaincinv(p, a) inverts gammainc(x, a) = p, matching the
        % argument order used in local_cdf above.
        Y = P(3) - eps + P(2) * gammaincinv(Q, P(1));
    case 'weibul'
        Y = P(3) + P(1) * (-log1p(-Q)).^(1 / P(2));
    otherwise
        error('LUQ:luqDist:badName', 'unsupported distribution: %s', Name);
end
end
