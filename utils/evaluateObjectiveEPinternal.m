% Function that evaluates the objective function and its gradient

function [ objective gradObjective ] = evaluateObjectiveEPinternal(ret, Xstar)

	% We compute the new columns of the covariance matrix

	nXstar = size(Xstar, 1);
	Kstar = computeNewColumnCovMatrix(ret.X, ret.n, ret.d, ret.m, ret.l, ret.sigma, ret.sigma0, Xstar, nXstar);

	% We compute the predictive mean under no constraints

	mPred = Kstar * ret.vAux * ret.mAux;
	vPred = ret.sigma - sum(Kstar .* (Kstar * ret.vAux), 2);

	% We compute the posterior covariances at the locations and at the minimum

	covPredMinimum = Kstar( 1 : nXstar, size(Kstar, 2) ) - Kstar * ret.vAux * ret.KstarMinimum';
	
	scale = (1 - 1e-4);
	while (any(vPred - 2 * scale * covPredMinimum + ret.vPredMinimum < 1e-10))
		scale = scale^2;
	end
	covPredMinimum = scale * covPredMinimum;

	% We compute the predictive distribution

	s = vPred - 2 * covPredMinimum + ret.vPredMinimum;
	alpha = (mPred - ret.mPredMinimum) ./ sqrt(s);
	beta = exp((-0.5 * log(2 * pi) - 0.5 * alpha.^2) - logphi(alpha));

	% We obtain the predictive variance under the constraints

	vPredConstrained = ret.sigma0 + vPred - beta ./ s .* ((beta + alpha) .* (vPred - covPredMinimum).^2);

	% We compute the predictive variance under no constraints

	if (ret.n > 0)
		kstarPosterior = computeKnm(Xstar, ret.X, ret.l, ret.sigma);
		vPredNoConstraints = (ret.sigma + ret.sigma0 + 1e-10 * ret.sigma) - sum((kstarPosterior * ret.KposteriorInv) .* kstarPosterior, 2);
	else
		vPredNoConstraints =  repmat(ret.sigma + ret.sigma0 + 1e-10 * ret.sigma, nXstar, 1);
	end

	objective = log(sqrt(2 * pi * exp(1) * vPredNoConstraints)) - log(sqrt(2 * pi * exp(1) * vPredConstrained));

	% We evaluate the gradient of the objective function

	gradObjective = zeros(size(Xstar, 1), ret.d);
	for i = 1 : ret.d

		KstarGradient = computeGradientNewColumnCovMatrix(ret.X, ret.n, ret.d, ret.m, ret.l, ret.sigma, ret.sigma0, Xstar, nXstar, i);
		vPredGradient = -2 * sum(Kstar .* (KstarGradient * ret.vAux), 2);

		if (ret.n > 0) 
			kstarPosteriorGradient = -ret.l( i ) * (repmat(Xstar(:, i), 1, ret.n) - repmat(ret.X(:, i)', nXstar, 1)) .* ...
				computeKnm(Xstar, ret.X, ret.l, ret.sigma);
			vPredNoConstraintsGradient = - 2 * sum((kstarPosterior * ret.KposteriorInv) .* kstarPosteriorGradient, 2);

		else
			vPredNoConstraintsGradient = zeros(nXstar, 1);
		end

		mPredGradient = KstarGradient * ret.vAux * ret.mAux;

		covPredMinimumGradient = scale * (KstarGradient( 1 : nXstar, size(Kstar, 2) ) - KstarGradient * ret.vAux * ret.KstarMinimum');
		sGradient = vPredGradient - 2 * covPredMinimumGradient;
		alphaGradient = (mPredGradient .* sqrt(s) - 0.5 ./ sqrt(s) .* sGradient .* (mPred - ret.mPredMinimum)) ./ s;
		betaGradient = -beta .* (alpha + beta) .* alphaGradient;

		B = beta ./ s;
		C = (beta + alpha);
		D = (vPred - covPredMinimum).^2;

		BGradient = betaGradient ./ s - sGradient ./ s.^2 .* beta;
		CGradient = betaGradient + alphaGradient;
		DGradient = 2 * vPred .* vPredGradient + 2 * covPredMinimum .* covPredMinimumGradient - ...
			2 * vPredGradient .* covPredMinimum - 2 * vPred .* covPredMinimumGradient;

		vPredConstrainedGradient = vPredGradient - BGradient .* C .* D - B .* CGradient .* D - B .* C .* DGradient;

		gradObjective(:, i) =  0.5 * vPredNoConstraintsGradient ./ vPredNoConstraints - 0.5 * vPredConstrainedGradient ./ vPredConstrained;
	end
