/*
 * MonteCarloHost.c
 *
 *  Created on: 06/feb/2018
 *  Author: marco
 */

#include "MonteCarlo.h"

void prodMat(
		double *first,
		double *second,
		double *result,
		int f_rows,
		int f_cols,
		int s_cols
){
    double somma;
    int i,j,k;
    for(i=0;i<f_rows;i++){
        for(j=0;j<s_cols;j++){
            somma = 0;
            for(k=0;k<f_cols;k++)
                //somma += first->data[i][k]*second->data[k][j];
                somma += first[k+i*f_cols] * second[j+k*s_cols];
            //result->data[i][j] = somma;
            result[j+i*s_cols] = somma;
        }
    }
}

//////////////////////////////////////////////////////
//////////   FINANCE FUNCTIONS
//////////////////////////////////////////////////////

static double randMinMax(double min, double max){
    double x=(double)rand()/(double)(RAND_MAX);
    return max*x+(1.0f-x)*min;
}

//Generator of a normal pseudo-random number with mean mu and volatility sigma with Box-Muller method
static double gaussian( double mu, double sigma ){
    double x = randMinMax(0, 1);
    double y = randMinMax(0, 1);
    return mu + sigma*(sqrt( -2.0 * log(x) ) * cos( 2.0 * M_PI * y ));
}

static double cnd(double d){
    const double       A1 = 0.31938153;
    const double       A2 = -0.356563782;
    const double       A3 = 1.781477937;
    const double       A4 = -1.821255978;
    const double       A5 = 1.330274429;
    const double ONEOVER2PI = 0.39894228040143267793994605993438;
    double K = 1.0 / (1.0 + 0.2316419 * fabs(d));
    double cnd = ONEOVER2PI * exp(- 0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    if (d > 0)
        cnd = 1.0 - cnd;
    return cnd;
}

double host_bsCall ( OptionData option ){
    double d1 = ( log(option.s / option.k) + (option.r + 0.5 * option.v * option.v) * option.t) / (option.v * sqrt(option.t));
    double d2 = d1 - option.v * sqrt(option.t);
    return option.s * cnd(d1) - option.k * exp(- option.r * option.t) * cnd(d2);
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
    prodMat(volatility, &g[0], result, n, n, 1);
    //X=m+A*G
    for(i=0;i<n;i++){
        result[i] += drift[i];
    }
    free(g);
}

// Call payoff
static double callPayoff( double s, double k, double t, double r, double v){
    double value = s * exp( (r - 0.5 * v * v) * t + gaussian(0,1) * sqrt(t) * v ) - k;
    return (value>0) ? (value):(0);
}

static void multiStockValue(double *s, double *v, double *g, double t, double r, int n, double *values){
    int i;
    for(i=0;i<n;i++){
        double mu = (r - 0.5 * v[i] * v[i])*t;
        double si = v[i] * g[i] * sqrt(t);
        values[i] = s[i] * exp(mu+si);
    }
}

// Monte Carlo simulation on the CPU
OptionValue host_vanillaOpt( OptionData option, int path){
    long double sum, var_sum, price, emp_stdev;
    OptionValue callValue;
    int i;
    sum = var_sum = 0.0f;
    srand((unsigned)time(NULL));

    for( i=0; i<path; i++){
        price = callPayoff(option.s,option.k,option.t,option.r,option.v);
        sum += price;
        var_sum += price * price;
    }

    price = exp(-option.r*option.t) * (sum/(double)path);
    emp_stdev = sqrt(
                     ((double)path * var_sum - sum * sum)
                     /
                     ((double)path * (double)(path - 1))
                     );

    callValue.Expected = price;
    callValue.Confidence = 1.96 * emp_stdev/sqrt(path);
    return callValue;
}

OptionValue host_basketOpt(MultiOptionData *option, int sim){
    int i,j;

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
        simGaussVect(option->d, &option->p[0][0], N, bt);
        multiStockValue(option->s, option->v, bt, t, r, N, s);
        for(j=0;j<N;j++)
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
