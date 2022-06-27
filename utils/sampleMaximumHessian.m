function [ samples hessians ] = sampleMaximumHessian(nSamples, Xsamples, Ysamples, sigma0, sigma, l, xmin, xmax, nFeatures)

	d = size(Xsamples, 2);

	samples = zeros(nSamples, d);
	hessians = {};

	for i = 1 : nSamples

		% We draw the random features

		W = randn(nFeatures, d) .* repmat(sqrt(l(i,:)), nFeatures, 1);
		b = 2 * pi * rand(nFeatures, 1);

		noise = randn(nFeatures, 1);
	
%		W = load('../RmoduleFinal/auxiliaryFiles/W.txt');
%		b = load('../RmoduleFinal/auxiliaryFiles/b.txt');
%		noise = load('../RmoduleFinal/auxiliaryFiles/noise.txt');

		% We sample from the posterior on the coefficients

		if (size(Xsamples, 1) > 0)

			tDesignMatrix = sqrt(2 * sigma(i) / nFeatures) * cos(W * Xsamples' + repmat(b, 1, size(Xsamples, 1)));

			if (size(Xsamples, 1) < nFeatures)

				woodbury = tDesignMatrix' * tDesignMatrix + sigma0(i) * eye(size(Xsamples, 1));
				inverseWoodbury = chol2invchol(woodbury);
				z = tDesignMatrix * Ysamples / sigma0(i);
				m = z - tDesignMatrix * (inverseWoodbury * (tDesignMatrix' * z));

				[ U D ] = eig(woodbury);
				D = diag(D);
				R = (sqrt(D) .* (sqrt(D) + sqrt(sigma0(i)))).^-1;

       				theta = noise - (tDesignMatrix * (U * (R .* (U' * (tDesignMatrix' * noise))))) + m;

			else

				Sigma = chol2invchol(cross(tDesignMatrix, tDesignMatrix') / sigma0(i) + eye(nFeatures));
				m = Sigma * tDesignMatrix * Ysamples / sigma0(i);
				theta = m + noise * chol(Sigma);

			end
		else

			theta = noise;

		end

		% We specify the objective function and its gradient

		targetVector = @(x) (theta' * sqrt(2 * sigma(i) / nFeatures) * cos(W * x' + repmat(b, 1, size(x, 1))))';
		targetVectorGradient = @(x) theta' * -sqrt(2 * sigma(i) / nFeatures) * (repmat(sin(W * x' + b), 1, d) .* W);

		% We do global optimization
		
		[ sample hessian ]= globalMaximizationHessian(targetVector, targetVectorGradient, xmin, xmax, zeros(0, d));

		samples(i, :) = sample;
		hessians{ i } = hessian;

%		fprintf(1, '%d\n', i);
	end
