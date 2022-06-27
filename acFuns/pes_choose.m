
function [optimum, optval] = pes_choose(nM, nK, xx, yy, KernelMatrixInv, ...
    guesses, sigma0, sigma, l, xmin, xmax, nFeatures, epsilon)
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

l = repmat(l, nK, 1);
sigma0 = repmat(sigma0, nK, 1);
sigma = repmat(sigma, nK, 1);
[ m hessians ] = sampleMaximumHessian(nM*nK, xx, yy, sigma0, sigma, l, xmin, xmax, nFeatures);

% We call the ep method
ret = initializeEPapproximation(xx, yy, m, l, sigma, sigma0, hessians);

% We define the cost function to be optimized

acfun = @(x) evaluateEPobjective(ret, x);
[optimum, optval] = globalMinimization(acfun, xmin, xmax, [guesses; m]);
