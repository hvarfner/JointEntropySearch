% Initialize EP approximation

function [ret ] = initializeEPapproximation(X, y, m, l, sigma, sigma0, hessians)

	ret = {};
	for i = 1 : size(m, 1)
		ret{ i } = getEPsolution(X, y, m(i, :)', l(i, :)', sigma(i), sigma0(i), hessians{ i });
%		fprintf(1, '%d\n', i);
	end
