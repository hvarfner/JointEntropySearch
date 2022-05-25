% Copyright (c) 2017 Zi Wang
function [f, g] = evaluateMES(x, locs, maxes, xx, yy, KernelMatrixInv, ...
    l, sigma, sigma0)
% This function computes the acquisition function value f and gradient g at 
% the queried point x using MES given sampled function values maxes, and
% observations xx (size T x d), yy (size T x 1).
% See also: mean_var.m
eps = 1e-16;
i = 1;
nK = size(maxes, 2);
nM = size(KernelMatrixInv, 2);
if nargout == 2
    
    f = 0;
    g = 0;
    sx = size(x,1);
    % Notice that if gradient is queried, sx == 1.
    dx = size(x,2);
    for i = 1 :  nM
        maxes_ = squeeze(maxes(i, :))';
        locs_ = squeeze(locs(i, :, :));
        [meanVector, varVector, meangrad, vargrad] = mean_var(x, xx, ...
            yy, KernelMatrixInv{i}, l(i,:), sigma(i), sigma0(i));
        % Avoid numerical errors by enforcing variance to be positive.
        varVector(varVector<=sigma0(i)+eps) = sigma0(i)+eps;
        
        [innov, d_innov] = innovation(x, locs_, maxes_, xx, yy, sigma0(i), sigma(i), l(i,:), KernelMatrixInv{i});
        
        delta_mean = innov .* meanVector;
        delta_var = (-1) * innov .* innov;
        
        cond_mean = meanVector + delta_mean;

        % this should not be negative (asserted above)
        noiseless_var = varVector - sigma0(i);
        cond_var = max(noiseless_var + delta_var, eps);
        trunc_var = truncnormvar(maxes_, cond_mean, cond_var);
        
        % assert no numerical strange stuff goes on
        trunc_var = max(trunc_var, 0); 
        % Obtain the posterior standard deviation.
        varVector = max(varVector, sigma0(i));

        base_entropy = log(2 * pi * varVector);
        cond_entropy = log(2 * pi * (trunc_var + sigma0(i)));
        mean_cond_entropy = mean(cond_entropy, 2);

        base_grad = vargrad ./ varVector;
        vargrad = repmat(reshape(vargrad, 1, 1, dx), 1, nK, 1);
        cond_grad = (vargrad  - 2 * (d_innov .* innov)) ./ ...
        (trunc_var + sigma0(i));
        g = g + base_grad - reshape(mean(cond_grad, 2), dx, 1);

    end
    g = g / (nK * nM);
    f = f / (nK * nM);
else

    f = 0;
    sx = size(x,1);
    % Notice that if gradient is queried, sx == 1.
    dx = size(x,2);
    for i = 1 :  nM
        maxes_ = squeeze(maxes(i, :))';
        locs_ = squeeze(locs(i, :, :));
        
        % for loop is over hyperparameters - ignore
        [meanVector, varVector] = mean_var(x, xx, ...
            yy, KernelMatrixInv{i}, l(i,:), sigma(i), sigma0(i));
        % Avoid numerical errors by enforcing variance to be positive.
        varVector(varVector<=sigma0(i)+eps) = sigma0(i)+eps;

        innov = innovation(x, locs_, maxes_, xx, yy, sigma0(i), sigma(i), l(i,:), KernelMatrixInv{i});
        delta_mean = innov .* meanVector;
        delta_var = (-1) * innov .* innov;
        
        cond_mean = meanVector + delta_mean;

        % this should not be negative (asserted above)
        noiseless_var = varVector - sigma0(i);
        cond_var = max(noiseless_var + delta_var, eps);
        trunc_var = truncnormvar(maxes_, cond_mean, cond_var);
        % Obtain the posterior standard deviation.
        base_entropy = log(2 * pi * varVector);
        cond_entropy = log(2 * pi * (trunc_var + sigma0(i)));
        mean_cond_entropy = mean(cond_entropy, 2);
        f = f + base_entropy - mean_cond_entropy;
    end
    f = f / (nK * nM);
end