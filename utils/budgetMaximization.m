% Copyright (c) 2017 Zi Wang
% Copyright (c) 2014, J.M. Hernandez-Lobato, M.W. Hoffman, Z. Ghahramani
% This function is partially adapted from the code for the paper
% Hernández-Lobato J. M., Hoffman M. W. and Ghahramani Z.
% Predictive Entropy Search for Efficient Global Optimization of Black-box
% Functions, In NIPS, 2014.
function [ optimum, fval] = budgetMaximization(target, xmin, xmax, guesses, useGradient)
% Approximately globally maximize the function target.
% target is a function handle returning the function value (and gradient
% if useGradient is true).
% target can take input of dimension nSample*d if only outputing the values.
%try
% Sample a random set of inputs and find the best one to initialize
% fmincon for optimization
if nargin == 3
    guesses = [];
end
if nargin == 4
    useGradient = 1;
end

norm_points = 25;
dim = size(xmax, 1);
gridSize = round(20000 * sqrt(dim));
Xgrid = rand_sample_interval(xmin, xmax, gridSize);

guess_plus_noise = repmat(guesses, norm_points, 1) + 0.01 * randn(norm_points * size(guesses, 1), dim);
guess_plus_noise = min(max(guess_plus_noise, 0), 1);
Xgrid = [ Xgrid ; guess_plus_noise ];

y = target(Xgrid);

[maxVal, maxIdx] = max(y, [], 1);

optimum = Xgrid(maxIdx,:);
fval = maxVal;

