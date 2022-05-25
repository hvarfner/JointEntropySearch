% Copyright (c) 2014, J.M. Hernandez-Lobato, M.W. Hoffman, Z. Ghahramani
% This function is from the code for the paper
% Hernández-Lobato J. M., Hoffman M. W. and Ghahramani Z.
% Predictive Entropy Search for Efficient Global Optimization of Black-box
% Functions, In NIPS, 2014.
% https://bitbucket.org/jmh233/codepesnips2014
function [ d_kern ] = compute_dKnm(X, Xbar, l, sigma)

	n = size(X, 1);
	m = size(Xbar, 1);
	dim = size(X, 2);

	X_calc = X .* repmat(sqrt(l'), n, 1);
	Xbar_calc = Xbar .* repmat(sqrt(l'), m, 1);

	Q = repmat(sum(X_calc.^2, 2), 1, m);
	Qbar = repmat(sum(Xbar_calc.^2, 2)', n, 1);

	distance = Qbar + Q - 2 * X_calc * Xbar_calc';
	kern = sigma * exp(-0.5 * distance);

	X_d = X .* repmat(l', n, 1);
	Xbar_d = Xbar .* repmat(l', m, 1);

	X_d = reshape(X_d, n, 1, dim);
	X_d = repmat(X_d, 1, m, 1);
	
	Xbar_d = reshape(Xbar_d, 1, m, dim);
	Xbar_d = repmat(Xbar_d, n, 1, 1);
	
	X_diff = X_d - Xbar_d;
	
	% add dimension to kernel
	dim_kern = repmat(reshape(kern, n, m, 1), 1, 1, dim);
	d_kern = (-1) * X_diff .* dim_kern;
	% output gradient: x_len (query) x output_len (co) x dim
end