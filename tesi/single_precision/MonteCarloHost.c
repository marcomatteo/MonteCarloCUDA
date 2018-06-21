/*
 * MonteCarloHost.c
 *
 *  Created on: 06/feb/2018
 *  Author: marco
 */

#include "MonteCarlo.h"

void MonteCarlo(MonteCarloData *data);

void prodMat(
             float *first,
             float *second,
             float *result,
             int f_rows,
             int f_cols,
             int s_cols
             ){
    float somma;
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
void Chol( float c[N][N], float a[N][N] ){
    int i = 0, j = 0, k = 0;
    float v[N];
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

float randMinMax(float min, float max){
    float x=(float)rand()/(float)(RAND_MAX);
    return max*x+(1.0f-x)*min;
}

// Metodo di Box-Muller per generare una v.a. gaussiana con media mu e varianza sigma
static float gaussian( float mu, float sigma ){
    float x = randMinMax(0, 1);
    float y = randMinMax(0, 1);
    return mu + sigma*(sqrt( -2.0 * log(x) ) * cos( 2.0 * M_PI * y ));
}

// Approssimazione di Hastings della funzione cumulata di una v.a. gaussiana
static float cnd(float d){
    const float       A1 = 0.31938153;
    const float       A2 = -0.356563782;
    const float       A3 = 1.781477937;
    const float       A4 = -1.821255978;
    const float       A5 = 1.330274429;
    const float ONEOVER2PI = 0.39894228040143267793994605993438;
    float K = 1.0 / (1.0 + 0.2316419 * fabs(d));
    float cnd = ONEOVER2PI * exp(- 0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    if (d > 0)
        cnd = 1.0 - cnd;
    return cnd;
}

// Prezzo di una opzione call secondo la formula di Black & Scholes
float host_bsCall ( OptionData option ){
    float d1 = ( log(option.s / option.k) + (option.r + 0.5 * option.v * option.v) * option.t) / (option.v * sqrt(option.t));
    float d2 = d1 - option.v * sqrt(option.t);
    return option.s * cnd(d1) - option.k * exp(- option.r * option.t) * cnd(d2);
}

//----------------------------------------------------
//Simulation of a Gaussian vector X~N(m,∑)
//Step 1: Compute the square root of the matrix ∑, generate lower triangular matrix A
//Step 2: Simulate n indipendent standard random variables G ~ N(0,1)
//Step 3: Return m + A*G
static void simGaussVect(float *drift, float *volatility, float *result){
    int i;
    float g[N];
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
static float callPayoff( OptionData option ){
    float value = option.s * exp(
                                  (option.r-0.5*option.v*option.v)*option.t+gaussian(0,1)*sqrt(option.t)*option.v)
    - option.k;
    return (value>0) ? (value):(0);
}

// Call payoff di un vettore di sottostanti
static void multiStockValue(float *s, float *v, float *g, float t, float r, int n, float *values){
    int i;
    for(i=0;i<n;i++){
        float mu = (r - 0.5 * v[i] * v[i])*t;
        float si = v[i] * g[i] * sqrt(t);
        values[i] = s[i] * exp(mu+si);
    }
}

void MonteCarlo(MonteCarloData *data){
    float sum, var_sum, emp_stdev, price;
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
        float bt[N], s[N];
        float st_sum;
        int j;
        for(i=0; i<data->path; i++){
            st_sum = 0;
            //Simulation of stock prices
            simGaussVect(data->option.d, &data->option.p[0][0], bt);
            multiStockValue(data->option.s, data->option.v, bt, data->option.t, data->option.r, N, s);
            for(j=0;j<N;j++)
                st_sum += s[j]*data->option.w[j];
            price = (float)st_sum - data->option.k;
            if(price<0)
                price = 0.0f;
            sum += price;
            var_sum += price*price;
        }
    }
    
    price = exp(-data->option.r*data->option.t) * (sum/(float)data->path);
    emp_stdev = sqrt(
                     ((float)data->path * var_sum - sum * sum)
                     /
                     ((float)data->path * (float)(data->path - 1))
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

void host_cvaEquityOption(CVA *cva, int sims){
    int i;
    float dt = cva->opt.t / (float)cva->n;
    MonteCarloData data;
    // Option
    data.option = cva->opt;
    
    // Execution parameters
    data.numOpt = N;
    data.path = sims;
    
    // Original option price
    MonteCarlo(&data);
    cva->ee[0] = data.callValue;
    
    // Expected Exposures (ee), Default probabilities (dp,fp)
    float sommaProdotto1=0;
    //float sommaProdotto2=0;
    for( i=1; i<(cva->n+1); i++){
        if((data.option.t -= (dt))<0){
            cva->ee[i].Confidence = 0;
            cva->ee[i].Expected = 0;
        }
        else{
            MonteCarlo(&data);
            cva->ee[i] = data.callValue;
        }
        cva->dp[i] = exp(-(dt*i) * cva->defInt)) - exp(-(dt)*(i-1) * (cva->defInt));
        //cva->fp[i] = exp(-(dt)*(i-1) * cva->credit.fundingspread / 100 / cva->credit.lgd)- exp(-(dt*i) * cva->credit.fundingspread / 100 / cva->credit.lgd );
        sommaProdotto1 += cva->ee[i].Expected * cva->dp[i];
        //sommaProdotto2 += cva->ee[i].Expected * cva->fp[i];
    }
    // CVA and FVA
    cva->cva = -sommaProdotto1 * cva->lgd;
    //cva->fva = -sommaProdotto2*cva->credit.lgd/100;
}

///////////////////////////////////
//    PRINT FUNCTIONS
///////////////////////////////////
void printVect( float *mat, int c ){
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
void printMat( float *mat, int r, int c ){
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
