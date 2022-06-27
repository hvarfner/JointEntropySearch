% Copyright (c) 2017 Zi Wang
% Copyright (c) 2014, J.M. Hernandez-Lobato, M.W. Hoffman, Z. Ghahramani
% This function is partially adapted from the code for the paper
% Hernández-Lobato J. M., Hoffman M. W. and Ghahramani Z.
% Predictive Entropy Search for Efficient Global Optimization of Black-box
% Functions, In NIPS, 2014.
function [ optimum, fval] = globalMinimization(target, xmin, xmax, guesses)

if nargin == 3
    guesses = [];
end
if nargin == 4
    useGradient = 1;
end
dim = size(xmin, 1);
gridSize = round(20000 * sqrt(dim));
Xgrid = rand_sample_interval(xmin, xmax, gridSize);
Xgrid = [ Xgrid ; guesses ];

y = target(Xgrid);
[minValue, maxIdx] = min(y);

start = Xgrid(maxIdx,:);

[ optimum, fval] = fmincon(target, start, [], [], [], [], xmin, xmax, [], ...
    optimset('MaxFunEvals', 100, 'TolX', eps, 'Display', 'off', 'GradObj', 'off'));
if fval > minValue
    optimum = start;
    fval = minValue;
    disp('failed global opt seq')
end

end
