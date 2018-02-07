/*
 * MonteCarlo.c
 *
 *  Created on: 06/feb/2018
 *      Author: marco
 */

#include "MonteCarlo.h"

//////////////////////////////////////////////////////
//////////   FINANCE FUNCTIONS
//////////////////////////////////////////////////////

static double randMinMax(double min, double max){
    double x=(double)rand()/(double)(RAND_MAX);
    return max*x+(1.0f-x)*min;
}

//Generator of a normal pseudo-random number with mean mu and volatility sigma
static double gaussian( double mu, double sigma ){
    double x = randMinMax(0, 1);
    double y = randMinMax(0, 1);
    return mu + sigma*(sqrt( -2.0 * log(x) ) * cos( 2.0 * M_PI * y ));
}

//Simulation std, rho and covariance matrix
double* getRandomSigma( int n ){
    Matrix std;
    //init the vectors of std
    mat_init(&std, 1, n);
    //creating the vectors of stds
    int i;
    for(i=0;i<n;i++)
        std.data[i] = randMinMax(0, 1);
    //printf("Stampo per debug vettore sigma:\n");
    //mat_print(&std);
    return std.data;
}
double* getRandomRho( int n ){
    Matrix rho;
    //init the vectors of rhos
    mat_init(&rho, n, n);
    int i,j;
    //creating the vectors of rhos
    for(i=0;i<n;i++){
        for(j=i;j<n;j++){
            double r;
            if(i==j)
                r=1;
            else
                r=randMinMax(-1, 1);
            rho.data[j+i*n] = r;
            rho.data[i+j*n] = r;
        }
    }
    //printf("Stampo per debug matrice rho:\n");
    //mat_print(&rho);
    return rho.data;
}

double* getCovMat(double *std, double *rho, int n){
    Matrix sigma;
    int i,j;
    //init the variance-covariance matrix
    mat_init(&sigma, n, n);
    //creating variance-covariance matrix
    for(i=0;i<n;i++)
        for(j=i;j<n;j++){
            double val = std[i]*std[j]*rho[j+i*n];
            sigma.data[j+i*n] = val;
            sigma.data[i+j*n] = val;
        }
    return sigma.data;
}
//----------------------------------------------------
//Simulation of a Gaussian vector X~N(m,∑)
//Step 1: Compute the square root of the matrix ∑, generate lower triangular matrix A
//Step 2: Simulate n indipendent standard random variables G ~ N(0,1)
//Step 3: Return m + A*G
static void simGaussVect(double *drift, double *volatility, int n, double *result){
    int i;
    double *g=NULL;
    g=(double*)malloc(n*sizeof(double));
    //RNGs
    for(i=0;i<n;i++)
        g[i]=gaussian(0, 1);
    Matrix gauss, r, vol;
    gauss.rows = n;     r.rows=n;       vol.cols = n;
    gauss.cols = 1;     r.cols=1;       vol.rows = n;
    gauss.data = &g[0]; r.data=result;  vol.data = volatility;
    //A*G
    mat_prod(&vol,&gauss,&r);
    //X=m+A*G
    for(i=0;i<n;i++){
        r.data[i] += drift[i];
    }
    free(g);
}

static void multiStockValue(double *s, double *v, double *g, double t, double r, int n, double *values){
    int i;
    for(i=0;i<n;i++){
        double mu = (r - 0.5 * v[i] * v[i])*t;
        double si = v[i] * g[i] * sqrt(t);
        values[i] = s[i] * exp(mu+si);
    }
}

extern "C" void RandomBasketOpt(double *st, double *randRho, double *randV, double *wp, double *drift, int N){
	int i;
	srand((unsigned)time(NULL));
	st=(double*)malloc(N*sizeof(double));
	wp=(double*)malloc(N*sizeof(double));
	drift=(double*)malloc(N*sizeof(double));
	for(i=0;i<N;i++){
		st[i]=randMinMax(K-10, K+10);
        wp[i]=dw;
        drift[i]=0;
	}
	randRho = getRandomRho(N);
	randV = getRandomSigma(N);
}

extern "C" void FreeBasketOpt(double *st, double *randRho, double *randV, double *wp, double *drift){
	free(st);
	free(randV);
	free(randRho);
	free(wp);
	free(drift);
}

extern "C" OptionValue CPUBasketOptCall(MultiOptionData *option, int sim){
    int i,j, n = option->n;

    //Monte Carlo algorithm
    OptionValue callValue;
    double st_sum=0.0f, sum=0.0f, var_sum=0.0f, price, emp_stdev;
    double k = option->k;
    double t = option->t;
    double r = option->r;

    //vectors of brownian and ST
    double bt[N];
    double s[N];

    for(i=0; i<sim; i++){
        st_sum = 0;
        //Simulation of stock prices
        simGaussVect(option->d, option->p, n, bt);
        multiStockValue(option->s, option->v, bt, t, r, n, s);
        for(j=0;j<n;j++)
            st_sum += s[j]*option->w[j];
        price = st_sum - k;
        if(price<0)
            price = 0.0f;
        sum += price;
        var_sum += price*price;
    }
    emp_stdev = sqrt(((double)sim * var_sum - sum * sum)/((double)sim * (double)(sim - 1)));

    callValue.Expected = exp(-r*t) * (sum/(double)sim);
    callValue.Confidence = 1.96 * emp_stdev/sqrt(sim);
    return callValue;
}
