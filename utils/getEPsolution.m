function [ ret ] = getEPsolution(X, y, m, l, sigma, sigma0, hessian)

	n = size(X, 1);
	d = size(X, 2);

	sizeA = d + 1;
	sizeB = n + d + d * (d - 1) / 2;
	sizeTotal = sizeA + sizeB;

	% We obtain the covariance matrices

	K = computeCovMatrix(X, n, d, m, l, sigma, sigma0);

	Kab = K( (sizeB + 1) : (sizeB + sizeA), 1 : sizeB );
	Kb = K( 1 : sizeB, 1 : sizeB );
	Ka = K( (sizeB + 1) : sizeTotal, (sizeB + 1) : sizeTotal );
	b = [ y; zeros(sizeB - n - d * (d - 1) / 2, 1) ];
	b = [ b ; hessian ];

	% We obtain the means and covariances for the conditional distribution of a given b
    jitter = eye(size(Kb, 1)) * 1e-2;
	KbInverse = chol2invchol(Kb + jitter);
	magivenb = Kab * KbInverse * b;
	Kagivenb = Ka - Kab * KbInverse * Kab';

	% We run the EP method

	if size(X, 1) == 0
		miny = 0;	
	else
		miny = min(y);
	end

	retEP = runEPinternal(miny, d, Kagivenb, magivenb, sigma0);

	% We compute the matrices needed to make predictions

	vAux = chol2invchol(diag([ zeros(sizeB, 1) ; retEP.vHat.^-1 ]) + K);
	mAux = [ b ; retEP.mHat ./ retEP.vHat ];

	% We compute the predictive mean at the minimum ignoring additive noise

	KstarMinimum = computeNewColumnCovMatrix(X, n, d, m, l, sigma, sigma0, m, 1);
	
	mPredMinimum = KstarMinimum * vAux * mAux;
	vPredMinimum = sigma - KstarMinimum * vAux * KstarMinimum';

	% We compute the posterior covariance matrix

	if size(X, 1) == 0
		KposteriorInv = zeros(0, 0);
	else
		KposteriorInv = chol2invchol(computeKmm(X, l, sigma, sigma0));
	end

	ret = struct('X', X, 'y', y, 'n', n, 'd', d, 'l', l, 'sigma', sigma, 'sigma0', sigma0, ...
		'vAux', vAux, 'mAux', mAux, 'KstarMinimum', KstarMinimum, 'mPredMinimum', mPredMinimum, ...
		'vPredMinimum', vPredMinimum, 'sizeA', sizeA, 'sizeB', sizeB, 'sizeTotal', sizeTotal, 'm', m, ...
		'KposteriorInv', KposteriorInv);
