function dist=KLDiv(varargin)
%  dist = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
%  distributions
%  P and Q  are automatically normalised to have the sum of one on rows
% have the length of one at each
% P =  n x nbins
% Q =  1 x nbins or n x nbins(one to one)
% dist = n x 1
%
%  dist = KLDiv(Name, Parameter, Data [, nbins]) is the same divergence
%  between a fitted family and a raw sample, with the binning handled
%  internally by luq_bin_fit -- the counterpart of the Python API's
%  kl_div(fit, x) / fit.kl_div(x). Data is the first argument of the
%  divergence, i.e. this returns D(data || fit).

if nargin >= 3 && ischar(varargin{1})
    if nargin >= 4
        nbins = varargin{4};
    else
        nbins = 39;
    end
    [Q, P] = luq_bin_fit(varargin{1}, varargin{2}, varargin{3}, nbins);
elseif nargin == 2
    P = varargin{1};
    Q = varargin{2};
else
    error('LUQ:KLDiv:usage', ...
        ['usage: KLDiv(P, Q) for binned mass vectors, or ', ...
         'KLDiv(Name, Parameter, Data [, nbins]) for a fit against data']);
end
% if size(P,2)~=size(Q,2)
%     error('the number of columns in P and Q should be the same');
% end
% if sum(~isfinite(P(:))) + sum(~isfinite(Q(:)))
%    error('the inputs contain non-finite values!') 
% end
% normalizing the P and Q
if size(Q,1)==1
    Q = Q ./sum(Q);
    P = P ./repmat(sum(P,2),[1 size(P,2)]);
    R = log2(P./repmat(Q,[size(P,1) 1]));
    P(or(isnan(R),isinf(R)))=[];
    R(or(isnan(R),isinf(R)))=[];
    dist =  sum(P.*R,2);
    
elseif size(Q,1)==size(P,1)
    
    Q = Q ./repmat(sum(Q,2),[1 size(Q,2)]);
    P = P ./repmat(sum(P,2),[1 size(P,2)]);
    R = log2(P./Q);
    P(or(isnan(R),isinf(R)))=[];
    R(or(isnan(R),isinf(R)))=[];

    dist =  sum(P.*R,2);
end
% resolving the case when P(i)==0
dist(isnan(dist))=0;
end


