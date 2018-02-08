/*
 * MonteCarloHost.c
 *
 *  Created on: 06/feb/2018
 *  Author: marco
 */

#include "MonteCarlo.h"

void ProdMat(
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

//Generator of a normal pseudo-random number with mean mu and volatility sigma
static double gaussian( double mu, double sigma ){
    double x = randMinMax(0, 1);
    double y = randMinMax(0, 1);
    return mu + sigma*(sqrt( -2.0 * log(x) ) * cos( 2.0 * M_PI * y ));
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
    ProdMat(volatility, &g[0], result, n, n, 1);
    //X=m+A*G
    for(i=0;i<n;i++){
        result[i] += drift[i];
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

OptionValue CPUBasketOptCall(MultiOptionData *option, int sim){
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
