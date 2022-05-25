% Copyright (c) 2017 Zi Wang
function [optimum, optval] = jes_choose(nM, nK, xx, yy, KernelMatrixInv, ...
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

d = size(xx, 2);

for m = 1 : nM
    [locs_m, maxes_m] = sampleMaximumNomarg(nK, xx, yy, sigma0(m), sigma(m), l(m,:), xmin, xmax, nFeatures);
    maxes(m, :) = maxes_m;
    locs(m, :, :) = locs_m;
end

%[locs, maxes] = sampleMaximumValues(nM, nK, xx, yy, sigma0, sigma, l, xmin, ...
%    xmax, nFeatures);

locs_reshaped = reshape(locs, [], size(xx, 2));
% Define the acquisition function (and gradient) of MES.

acfun = @(x) evaluateJES(x, locs, maxes, xx, yy, KernelMatrixInv, l, sigma, sigma0);

%gridSize = round(20000 * sqrt(d));
%Xgrid = repmat(xmin', gridSize, 1) + repmat((xmax - xmin)', gridSize, 1) ...
%    .* rand(gridSize, d);
%[out,idx] = sort(Xgrid);
%csvwrite('plot/jesacq.csv', [Xgrid(idx) acfun(Xgrid(idx))]); 

%X = linspace(0, 1, 101);
%[X, Y] = meshgrid(X);
%X = reshape(X, [], 1);
%Y = reshape(Y, [], 1);
%XY = [X Y];
%f = acfun(XY);
%XYF = [X Y f];
%csvwrite(append('acq_optima/optima', append(string(size(yy, 1)), '.csv')), locs);
%csvwrite(append('acq_optima/acq', append(string(size(yy, 1)), '.csv')), XYF);
%csvwrite(append('acq_optima/locs', append(string(size(yy, 1)), '.csv')), xx);
%csvwrite(append('acq_optima/vals', append(string(size(yy, 1)), '.csv')), yy);

% Optimize the acquisition function.
[optimum, optval] = globalMaximization(acfun, xmin, xmax, [guesses;xx; locs_reshaped]);
