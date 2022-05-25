% Copyright (c) 2017 Zi Wang
% Copyright (c) 2014, J.M. Hernandez-Lobato, M.W. Hoffman, Z. Ghahramani
% This function is adapted from the code for the paper
% Hernández-Lobato J. M., Hoffman M. W. and Ghahramani Z.
% Predictive Entropy Search for Efficient Global Optimization of Black-box
% Functions, In NIPS, 2014.
% https://bitbucket.org/jmh233/codepesnips2014
function [locations, samples] = sampleMaximumNomarg(nK, xx, ...
                 yy, sigma0, sigma, l, xmin, xmax, nFeatures)
% This function returns sampled maximum values for the posterior GP 
% conditioned on current obervations. We construct random features and 
% optimize functions drawn from the posterior GP.
% nM is the number of sampled GP hyper-parameter settings.
% nK is the number of sampled maximum values.
% xx, yy are the current observations.
% sigma0, sigma, l are the hyper-parameters of the Gaussian kernel.
% xmin, xmax are the lower and upper bounds for the search space.
% nFeatures is the number of random features sampled to approximate the
% GP.
% epsilon is an offset on the sampled max-value.

d = size(xx, 2);

n_samples = nK;

samples = zeros(nK, 1)*-1e10;
locations = zeros(nK, d)*-1e10;

% Draw weights for the random features.
        
W = randn(nFeatures, d) .* repmat(sqrt(l), nFeatures, 1);
b = 2 * pi * rand(nFeatures, 1);

% Compute the features for xx.
Z = sqrt(2 * sigma / nFeatures) * cos(W * xx' + ...
    repmat(b, 1, size(xx, 1)));

% Draw the coefficient theta.
noise = randn(nFeatures, 1);
if (size(xx, 1) < nFeatures)
    % We adopt the formula $theta \sim \N(Z(Z'Z + \sigma^2 I)^{-1} y, 
    % I-Z(Z'Z + \sigma^2 I)Z')$.
    Sigma = Z' * Z + sigma0 * eye(size(xx, 1));
    mu = Z*chol2invchol(Sigma)*yy;
    [U, D] = eig(Sigma);
    D = diag(D);
    R = (sqrt(D) .* (sqrt(D) + sqrt(sigma0))).^-1;
    theta = noise - (Z * (U * (R .* (U' * (Z' * noise))))) + mu;
else
    % $theta \sim \N((ZZ'/\sigma^2 + I)^{-1} Z y / \sigma^2,
    % (ZZ'/\sigma^2 + I)^{-1})$.
    Sigma = chol2invchol(Z*Z' / sigma0 + eye(nFeatures));
    mu = Sigma * Z * yy / sigma0;
    var = chol(Sigma);
    
end


noise = randn(nFeatures, n_samples);

theta = noise - (Z * (U * (R .* (U' * (Z' * noise))))) + mu;

% Obtain a function sampled from the posterior GP.

targetVector = @(x) (theta' * sqrt(2 * sigma / nFeatures) * ...
    cos(W * x' + repmat(b, 1, size(x, 1))))';



% Optimize the function.
%[location, sample]= globalMaximization(target, xmin, xmax, xx);
[locations, samples]= budgetMaximization(targetVector, xmin, xmax, xx);


% If the optimization failed, we manually set the
% sample to be max(yy) + epsilon to make sure our samples are
% upperbounds on the underlying function.
%if sample < max(yy) + 5*sqrt(sigma0(i))
%    samples(i, j) = max(yy) + 5*sqrt(sigma0(i));
        %end
end

