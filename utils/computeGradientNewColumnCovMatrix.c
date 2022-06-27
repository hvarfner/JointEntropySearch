#include <math.h>
#include "mex.h"

/* Function that evaluates cov[x_i,x_j] */

double cov(double *x_i, double *x_j, double sigma, double *l, int d) {

	int i;

	double squaredDistance = 0;
	for (i = 0 ; i < d ; i++)
		squaredDistance += (x_i[ i ] - x_j[ i ]) * (x_i[ i ] - x_j[ i ]) * l[ i ];

	return sigma * exp(-0.5 * squaredDistance);
}

/* Function that evaluates d cov[x_i,x_j] / [ dx_j[k] ] */

double dcovdxjk(double *x_i, double *x_j, double sigma, double *l, int d, int k) {

	return l[ k ] * (x_i[ k ] - x_j[ k ]) * cov(x_i, x_j, sigma, l, d);
}

/* Function that evaluates d cov[x_i,x_j] / [ dx_i[m] ] */

double dcovdxim(double *x_i, double *x_j, double sigma, double *l, int d, int m) {

	return -l[ m ] * (x_i[ m ] - x_j[ m ]) * cov(x_i, x_j, sigma, l, d);
}

/* Function that evaluates d^2 cov[x_i,x_j] / [ dx_j[k] dx_i[m] ] */

double d2covdxjkxim(double *x_i, double *x_j, double sigma, double *l, int d, int k, int m) {

	return (m == k ? l[ m ] : 0) * cov(x_i, x_j, sigma, l, d) +
		l[ k ] * (x_i[ k ] - x_j[ k ]) * dcovdxim(x_i, x_j, sigma, l, d, m);
}

/* Function that evaluates d^3 cov[x_i,x_j] / [ dx_j[k] dx_j[lIndex] dx_i[m] ] */

double d3covdxjkxjlxim(double *x_i, double *x_j, double sigma, double *l, int d, int k, int lIndex, int m) {

	return -(lIndex == k ? l[ k ] : 0) * dcovdxim(x_i, x_j, sigma, l, d, m) +
		(m == k ? l[ k ] : 0) * dcovdxjk(x_i, x_j, sigma, l, d, lIndex) +
		l[ k ] * (x_i[ k ] - x_j[ k ]) * d2covdxjkxim(x_i, x_j, sigma, l, d, lIndex, m);
}

/* Functions needed to compute the derivative of the new column of the GP covariance matrix for making predictions */

/* Function which fills in the gradient of the covariance entries between xstar and the n data points */

void computeGradientCovXstarDataPoints(double *ret, double *Xstar, int nXstar,
	double *X, int n, int nrow, int d, double *l, double sigma, double sigma0, int z) {

	int i, j, k;

	/* We allocate memory to store the data points */

	double *x1 = (double *) malloc(sizeof(double) * d);
	double *x2 = (double *) malloc(sizeof(double) * d);
	
	/* We compute the covariance function at the n available observations */
	
	for (i = 0 ; i <  nXstar ; i++) {
		for (j = 0 ; j < n ; j++) {

			/* We store the data points at which to evaluate the covariance function */

			for (k = 0 ; k < d ; k++) {
				x1[ k ] = Xstar[ i + nXstar * k ];
				x2[ k ] = X[ j + n * k ];
			}

			/* We evaluate the covariance function */

			ret[ i + j * nXstar ] = dcovdxim(x1, x2, sigma, l, d, z);
		}
	}

	free(x1); free(x2);
}

/* Function which fills in the gradient of the covariance entries between xstar and the gradient at the minimum */

void computeGradientCovXstarGradient(double *ret, double *Xstar, int nXstar, double *X, int n,
	int nrow, int d, double *l, double sigma, double *m, int z) {

	int i, j, k;

	/* We allocate memory to store the data points */

	double *x1 = (double *) malloc(sizeof(double) * d);

	/* We evaluate the covariance function at the n available observations and d gradient variables */

	for (i = 0 ; i < nXstar ; i++) {
		for (j = 0 ; j < d ; j++) {

			/* We store the data points at which to evaluate the covariance function */

			for (k = 0 ; k < d ; k++)
				x1[ k ] = Xstar[ i + nXstar * k ];

			/* We evaluate the covariance function */

			ret[ i + (n + j) * nXstar ] = d2covdxjkxim(x1, m, sigma, l, d, j, z);
		}
	}

	free(x1);
}

/* Function which fills in the gradient of the covariance entries between xstar and the non diagonal hessian elements */

void computeGradientCovXstarNonDiagHessian(double *ret, double *Xstar,
	int nXstar, double *X, int n, int nrow, int d, double *l, double sigma, double *m, int z) {

	int i, j, k, h, counter;

	/* We allocate memory to store the data points */

	double *x1 = (double *) malloc(sizeof(double) * d);

	for (i = 0 ; i < nXstar ; i++) {
		counter = 0;
		for (j = 0 ; j < d ; j++) {
			for (h = j + 1 ; h < d ; h++) {

				/* We store the data points at which to evaluate the covariance function */

				for (k = 0 ; k < d ; k++)
					x1[ k ] = Xstar[ i + nXstar * k ];

				/* We evaluate the covariance function */

				ret[ i + (n + d + counter) * nXstar ] = d3covdxjkxjlxim(x1, m, sigma, l, d, j, h, z);

				counter++;
			}
		}
	}

	free(x1);
}

/* Function which fills in the gradient of the covariance entries between xstar and the diagonal hessian elements */

void computeGradientCovXstarDiagHessian(double *ret, double *Xstar,
	int nXstar, double *X, int n, int nrow, int d, double *l, double sigma, double *m, int z) {

	int i, j, k;

	/* We allocate memory to store the data points */

	double *x1 = (double *) malloc(sizeof(double) * d);

	for (i = 0 ; i < nXstar ; i++) {
		for (j = 0 ; j < d ; j++) {

			/* We store the data points at which to evaluate the covariance function */

			for (k = 0 ; k < d ; k++)
				x1[ k ] = Xstar[ i + nXstar * k ];

			/* We evaluate the covariance function */

			ret[ i + (n + d + d * (d - 1) / 2 + j) * nXstar ] = d3covdxjkxjlxim(x1, m, sigma, l, d, j, j, z);
		}
	}

	free(x1);
}

/* Function which fills in the covariance entries between xstar and the minimum */

void computeGradientCovXstarMinimum(double *ret, double *Xstar,
	int nXstar, double *X, int n, int nrow, int d, double *l, double sigma, double *m, int z) {

	int i, j, k, h, counter;

	/* We allocate memory to store the data points */

	double *x1 = (double *) malloc(sizeof(double) * d);

	for (i = 0 ; i < nXstar ; i++) {

		/* We store the data points at which to evaluate the covariance function */

		for (k = 0 ; k < d ; k++)
			x1[ k ] = Xstar[ i + nXstar * k ];

		/* We evaluate the covariance function */

		ret[ i + (n + d + d * (d - 1) / 2 + d) * nXstar ] = dcovdxim(x1, m, sigma, l, d, z);
	}

	free(x1);
}

/* Function that computes the derivative of the new column of the GP covariance matrix for making predictions */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	#define X_matlab prhs[0]
	#define n_matlab prhs[1]
	#define d_matlab prhs[2]
	#define m_matlab prhs[3]
	#define l_matlab prhs[4]
	#define sigma_matlab prhs[5]
	#define sigma0_matlab prhs[6]
	#define Xstar_matlab prhs[7]
	#define nXstar_matlab prhs[8]
	#define z_matlab prhs[9]

	#define ret_matlab plhs[0]

	int i, j, nrow;

	/* We initialize the c pointers to the R variables */

	int n = *mxGetPr(n_matlab);
	int d = *mxGetPr(d_matlab);
	double *X = mxGetPr(X_matlab);
	double *Xstar = mxGetPr(Xstar_matlab);
	int nXstar = *mxGetPr(nXstar_matlab);
	double *m = mxGetPr(m_matlab);
	double *l = mxGetPr(l_matlab);
	double sigma = *mxGetPr(sigma_matlab);
	double sigma0 = *mxGetPr(sigma0_matlab);
	int z = *mxGetPr(z_matlab);

	/* We allocate memory for the result */

	nrow = (n + d + d * (d - 1) / 2 + d + 1);
	ret_matlab = mxCreateDoubleMatrix(nXstar, nrow, mxREAL);
	
	double *ret = mxGetPr(ret_matlab);

	z = z - 1;

	/********** WE COMPUTE THE GRADIENTS OF THE COVARIANCES BETWEEN Xstar AND EVERYTHING ELSE **********/

	/* We compute the gradient of the covariances for the n available observations */
	
	computeGradientCovXstarDataPoints(ret, Xstar, nXstar, X, n, nrow, d, l, sigma, sigma0, z);

	/* We compute the gradient of the covariances at the n available observations and the d gradient values at the minimum */

	computeGradientCovXstarGradient(ret, Xstar, nXstar, X, n, nrow, d, l, sigma, m, z);

	/* We compute the graident of the covariances at the n available observations and the d * (d - 1) / 2 non diagonal
	 * entries of the hessian at the minimum */

	computeGradientCovXstarNonDiagHessian(ret, Xstar, nXstar, X, n, nrow, d, l, sigma, m, z);

	/* We compute the gradient fo the covariances at the n available observations and the d diagonal entries of the
	 * hessian at the minimum */

	computeGradientCovXstarDiagHessian(ret, Xstar, nXstar, X, n, nrow, d, l, sigma, m, z);

	/* We compute the gradient of the covariances at the n available observations and the minimum */

	computeGradientCovXstarMinimum(ret, Xstar, nXstar, X, n, nrow, d, l, sigma, m, z);

	return;
}
