% Evaluate EP objective

function [ objective gradient ] = evaluateEPobjective(ret, Xstar)

	objective = zeros(size(Xstar, 1), 1);
	gradient = zeros(size(Xstar, 1), size(Xstar, 2));
	for i = 1 : size(ret, 2)
		[ objectiveLocal gradientLocal ] = evaluateObjectiveEPinternal(ret{ i }, Xstar);
		objective = objective - objectiveLocal;
		gradient = gradient - gradientLocal;
	end

	objective = objective / size(ret, 2);
	gradient = gradient / size(ret, 2);
