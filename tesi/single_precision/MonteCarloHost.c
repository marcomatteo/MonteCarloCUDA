/*
 * MonteCarloHost.c
 *
 *  Created on: 06/feb/2018
 *  Author: marco
 */

#include "MonteCarlo.h"

void MonteCarlo(MonteCarloData *data);

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

// Algoritmo Golub e Van Loan
void Chol( double c[N][N], double a[N][N] ){
    int i = 0, j = 0, k = 0;
    double v[N];
    for( j=0; j<N; j++){
        for( i=0; i<N; i++ ){
            a[i][j] = 0;
            if( i>=j){
                v[i]=c[i][j];
                for(k=0; k<=(j-1); k++)
                    v[i] -= a[j][k] * a[i][k];
                if(v[j]>0)
                    a[i][j] = v[i] / sqrt( v[j] );
            }
        }
    }
}

//////////////////////////////////////////////////////
//////////   FINANCE FUNCTIONS
//////////////////////////////////////////////////////

double randMinMax(double min, double max){
    double x=(double)rand()/(double)(RAND_MAX);
    return max*x+(1.0f-x)*min;
}

// Metodo di Box-Muller per generare una v.a. gaussiana con media mu e varianza sigma
static double gaussian( double mu, double sigma ){
    double x=(double)rand()/(double)(RAND_MAX);
    double x=(double)rand()/(double)(RAND_MAX);
    return mu + sigma*(sqrt( -2.0 * log(x) ) * cos( 2.0 * M_PI * y ));
}

// Approssimazione di Hastings della funzione cumulata di una v.a. gaussiana
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

// Prezzo di una opzione call secondo la formula di Black & Scholes
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
static void simGaussVect(double *drift, double *volatility, double *result){
    int i;
    double g[N];
    //RNGs
    for(i=0;i<N;i++)
        g[i]=gaussian(0, 1);
    prodMat(volatility, g, result, N, N, 1);
    //X=m+A*G
    for(i=0;i<N;i++){
        result[i] += drift[i];
    }
}

// Call payoff
static double callPayoff( OptionData option ){
    double value = option.s * exp(
                                  (option.r-0.5*option.v*option.v)*option.t+gaussian(0,1)*sqrt(option.t)*option.v)
    - option.k;
    return (value>0) ? (value):(0);
}

// Call payoff di un vettore di sottostanti
static void multiStockValue(double *s, double *v, double *g, double t, double r, int n, double *values){
    int i;
    for(i=0;i<n;i++){
        double mu = (r - 0.5 * v[i] * v[i])*t;
        double si = v[i] * g[i] * sqrt(t);
        values[i] = s[i] * exp(mu+si);
    }
}

void MonteCarlo(MonteCarloData *data){
    double sum, var_sum, emp_stdev, price;
    int i;
    sum = var_sum = 0.0f;
    srand((unsigned)time(NULL));
    if(data->numOpt == 1){
        OptionData opt;
        opt.s = data->option.s[0];
        opt.k = data->option.k;
        opt.v = data->option.v[0];
        opt.t = data->option.t;
        opt.r = data->option.r;
        for( i=0; i<data->path; i++){
            price = callPayoff(opt);
            sum += price;
            var_sum += price * price;
        }
    }
    else{
        //vectors of brownian and ST
        double bt[N], s[N];
        float st_sum;
        int j;
        for(i=0; i<data->path; i++){
            st_sum = 0;
            //Simulation of stock prices
            simGaussVect(data->option.d, &data->option.p[0][0], bt);
            multiStockValue(data->option.s, data->option.v, bt, data->option.t, data->option.r, N, s);
            for(j=0;j<N;j++)
                st_sum += s[j]*data->option.w[j];
            price = (double)st_sum - data->option.k;
            if(price<0)
                price = 0.0f;
            sum += price;
            var_sum += price*price;
        }
    }
    
    price = exp(-data->option.r*data->option.t) * (sum/(double)data->path);
    emp_stdev = sqrt(
                     ((double)data->path * var_sum - sum * sum)
                     /
                     ((double)data->path * (double)(data->path - 1))
                     );
    
    data->callValue.Confidence = 1.96 * emp_stdev/sqrt(data->path);
    data->callValue.Expected = price;
}

// Monte Carlo simulation on the CPU
OptionValue host_vanillaOpt( OptionData option, int path){
    MonteCarloData data;
    data.option.s[0] = option.s;
    data.option.v[0] = option.v;
    data.option.k = option.k;
    data.option.t = option.t;
    data.option.r = option.r;
    data.numOpt = 1;
    data.path = path;
    
    MonteCarlo(&data);
    return data.callValue;
}

OptionValue host_basketOpt(MultiOptionData *option, int sim){
    MonteCarloData data;
    data.option = *option;
    data.numOpt = N;
    data.path = sim;
    
    MonteCarlo(&data);
    return data.callValue;
}

void host_cvaEquityOption(CVA *cva, int numBlocks, int numThreads){
    int i;
    double dt = cva->opt.t / (double)cva->n;
    MonteCarloData data;
    // Option
    data.option.w[0] = 1;
    data.option.d[0] = 0;
    data.option.p[0][0] = 1;
    data.option.s[0] = cva->opt.s;
    data.option.v[0] = cva->opt.v;
    data.option.k = cva->opt.k;
    data.option.r = cva->opt.r;
    data.option.t = cva->opt.t;
    
    // Execution parameters
    data.numOpt = N;
    data.path = PATH;
    
    // Original option price
    MonteCarlo(&data);
    cva->ee[0] = data.callValue;
    
    // Expected Exposures (ee), Default probabilities (dp,fp)
    double sommaProdotto1=0,sommaProdotto2=0;
    for( i=1; i<(cva->n+1); i++){
        if((data.option.t -= (dt))<0){
            cva->ee[i].Confidence = 0;
            cva->ee[i].Expected = 0;
        }
        else{
            MonteCarlo(&data);
            cva->ee[i] = data.callValue;
        }
        cva->dp[i] = exp(-(dt)*(i-1) * cva->credit.creditspread / 100 / cva->credit.lgd)
        - exp(-(dt*i) * cva->credit.creditspread / 100 / cva->credit.lgd );
        cva->fp[i] = exp(-(dt)*(i-1) * cva->credit.fundingspread / 100 / cva->credit.lgd)
        - exp(-(dt*i) * cva->credit.fundingspread / 100 / cva->credit.lgd );
        sommaProdotto1 += cva->ee[i].Expected * cva->dp[i];
        sommaProdotto2 += cva->ee[i].Expected * cva->fp[i];
    }
    // CVA and FVA
    cva->cva = -sommaProdotto1*cva->credit.lgd/100;
    cva->fva = -sommaProdotto2*cva->credit.lgd/100;
}

///////////////////////////////////
//    PRINT FUNCTIONS
///////////////////////////////////
void printVect( double *mat, int c ){
    int i,j,r=1;
    for(i=0; i<r; i++){
        printf("\n!\t");
        for(j=0; j<c; j++){
            printf("\t%f\t",mat[j+i*c]);
        }
        printf("\t!");
    }
    printf("\n\n");
}
void printMat( double *mat, int r, int c ){
    int i,j;
    for(i=0; i<r; i++){
        printf("\n!\t");
        for(j=0; j<c; j++){
            printf("\t%f\t",mat[j+i*c]);
        }
        printf("\t!");
    }
    printf("\n\n");
}
void printOption( OptionData o){
    printf("\n-\tOption data\t-\n\n");
    printf("Underlying asset price:\t € %.2f\n", o.s);
    printf("Strike price:\t\t € %.2f\n", o.k);
    printf("Risk free interest rate: %.2f %%\n", o.r * 100);
    printf("Volatility:\t\t\t %.2f %%\n", o.v * 100);
    printf("Time to maturity:\t\t %.2f %s\n", o.t, (o.t>1)?("years"):("year"));
}

void printMultiOpt( MultiOptionData *o){
    printf("\n-\tBasket Option data\t-\n\n");
    printf("Number of assets: %d\n",N);
    printf("Underlying assets prices:\n");
    printVect(o->s, N);
    printf("Volatility:\n");
    printVect(o->v, N);
    printf("Weights:");
    printVect(o->w, N);
    printf("Correlation matrix:\n");
    printMat(&o->p[0][0], N, N);
    printf("Strike price:\t € %.2f\n", o->k);
    printf("Risk free interest rate: %.2f \n", o->r);
    printf("Time to maturity:\t %.2f %s\n", o->t, (o->t>1)?("years"):("year"));
}
