function [ fHat ] = runEPinternal(ymin, d, Kagivenb, magivenb, sigma0)
	KagivenbInverse = chol2invchol(Kagivenb);
	a = struct('m', magivenb ./ diag(Kagivenb), 'v', diag(Kagivenb).^-1);
        
	noise = 1e-6;
	fHat = struct('mHat', zeros(d + 1, 1), 'vHat', zeros(d + 1, 1));

	% We repeat until the method converges

	damping = 1;
	convergence = false;

	while (~convergence)

		aOld = a;
	
		% We eliminate the contribution of the approximate factor from the posterior approximation

		vOld = (a.v - fHat.vHat).^-1;
		mOld = vOld .* (a.m - fHat.mHat);

		% We refine the constraint factors

		mOldAux = mOld;
		vOldAux = vOld;
		mOldAux( d + 1 ) = ymin - mOldAux( d + 1 );
		vOldAux( d + 1 ) = vOldAux( d + 1 ) + sigma0;

		alpha = mOldAux ./ sqrt(vOldAux);

		ratio = exp((-0.5 * log(2 * pi) - 0.5 * alpha.^2) - logphi(alpha));
		beta = ratio .* (alpha + ratio) ./ vOldAux;
		kappa = (mOldAux ./ sqrt(vOldAux) + ratio) ./ sqrt(vOldAux);
		kappa( d + 1 ) = -kappa( d + 1 );

		vHatNew = beta ./ (1 - beta .* vOld);
		vHatNew( find(abs(vHatNew) < 1e-300) ) = 1e-300;
		mHatNew = (mOld + 1 ./ kappa) .* vHatNew;

		vHatNew( find(vOld < 0) ) = fHat.vHat( find(vOld < 0) );
		mHatNew( find(vOld < 0) ) = fHat.mHat( find(vOld < 0) );

		failedEPupdate = true;
		while (failedEPupdate) 

			% We do damping

			mHatNew = mHatNew * damping + fHat.mHat * (1 - damping);
			vHatNew = vHatNew * damping + fHat.vHat * (1 - damping);

			% We verify that the posterior is well defined if not we increase the damping

			[ eigenvectors eigenvalues ] = eig(diag(vHatNew) + KagivenbInverse);

			if (any(1 / diag(eigenvalues) <= 1e-10))
				failedEPupdate = false;
			else
				damping = damping * 0.5;
			end
		end

		fHat.mHat = mHatNew;
		fHat.vHat = vHatNew;
		% We update the posterior marginals

		fHat.vHat = max(noise, fHat.vHat);
		Vnew = chol2invchol(diag(fHat.vHat) + KagivenbInverse);
		mNew = Vnew * (fHat.mHat + KagivenbInverse * magivenb);

		a.m = mNew ./ diag(Vnew);
		a.v = diag(Vnew).^-1;

		change = max(abs(a.m ./ a.v - aOld.m ./ aOld.v));
		change = max(change, max(abs(a.v.^-1 - aOld.v.^-1)));

		if (change < 1e-6)
			convergence = true;
		end
	
		damping = damping * 0.99;

%		fprintf(1, '%f\n', change);
	end
