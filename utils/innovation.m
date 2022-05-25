function [innov, d_innov] = innovation(x, locs, maxes, xx, yy, sigma0, sigma, l, KernelMatrixInv)

l = l';
% k_t 5 x 10
k_t = computeKnm(x, locs, l, sigma);

% 10 * 10 --> 10 x 1
% noiseless = 1e-2 * the noise variance
kappa_t = diag(computeKnm(locs, locs, l, sigma)) + sigma0 * 1e-2;

% 3 x 5
k_0 = computeKnm(xx, x, l, sigma);
% size 3 * 10
eps_t = computeKnm(xx, locs, l, sigma);

innov_num = k_t - k_0.' * KernelMatrixInv * eps_t;

innov_denom = sqrt(kappa_t - diag(eps_t' * KernelMatrixInv * eps_t));
innov_denom = repmat(innov_denom, 1, size(x, 1))';
innov = innov_num ./ innov_denom ;  

if nargout == 2
    dk_t_T = compute_dKnm(x, locs, l, sigma);
    dk_0_T = compute_dKnm(x, xx, l, sigma);
    d_innov_num = dk_t_T - pagemtimes(dk_0_T, KernelMatrixInv * eps_t);
    d_innov = d_innov_num ./ innov_denom;
end

end