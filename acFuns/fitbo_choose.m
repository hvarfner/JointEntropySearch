% Copyright (c) 2017 Zi Wang
function [optimum, optval] = fitbo_choose(nM, nK, xx, yy, KernelMatrixInv, ...
    guesses, eta, sigma0, sigma, l, xmin, xmax, nFeatures, epsilon)
% This function returns the next evaluation point using MES-R.
% nM is the number of sampled GP hyper-parameter settings.
% nK is the number of sampled maximum values.
% xx, yy are the current observations.
% KernelMatrixInv is the gram maxtrix inverse under different GP
% hyper-parameters.
% guesses are the inferred points to recommend to evaluate.
% sigma0, sigma, l are the hyper-parameters of the Gaussian kernel.
% xmin, xmax are the lower and upper bounds for the search space.
% nFeatures is the number of random features sampled to approximate the GP.

% Sample posterior max-values of the function with random features.

d = size(xx, 2);

l = repmat(l, nK, 1);
sigma0 = repmat(sigma0, nK, 1);
sigma = repmat(sigma, nK, 1);
KernelMatrixInv = repmat(KernelMatrixInv, 1, nK)';

eta = reshape(eta, nM * nK, 1);

acfun = @(x) evaluateFITBO(x, xx, yy, KernelMatrixInv, l, sigma, sigma0, eta, nM * nK);
[optimum, optval] = globalMaximization(acfun, xmin, xmax, guesses);
