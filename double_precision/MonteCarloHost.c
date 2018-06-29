/*
 * MonteCarloHost.c
 *
 *  Created on: 06/feb/2018
 *  Author: marco
 */

#include "MonteCarlo.h"

void MonteCarlo(MonteCarloData *data);

/////////////////////////////////////////
////////    PRINT FUNCTIONS     /////////
/////////////////////////////////////////
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
    printf("Volatility:\t\t %.2f %%\n", o.v * 100);
    printf("Time to maturity:\t %.2f %s\n", o.t, (o.t>1)?("years"):("year"));
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
    printf("Strike price:\t\t € %.2f\n", o->k);
    printf("Risk free interest rate: %.2f \n", o->r);
    printf("Time to maturity:\t %.2f %s\n", o->t, (o->t>1)?("years"):("year"));
}

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
    double x = randMinMax(0, 1);
    double y = randMinMax(0, 1);
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

// Geometric Brownian path
static double geomBrownian(double s, double t, double r, double v){
    double x = (r - 0.5 * v * v) * t + gaussian(0,1) * sqrt(t) * v;
    return s * exp(x);
}

// Call payoff
static double callPayoff( OptionData option ){
    double value = option.s * exp((option.r-0.5*option.v*option.v) * option.t + gaussian(0,1) * sqrt(option.t) * option.v) - option.k;
    return (value>0) ? (value):(0);
}

// Call payoff di un vettore di sottostanti
static void multiStockValue(double *s, double *v, double *g, double t, double r, int n, double *values){
    int i;
    for(i=0;i<n;i++){
        double mu = (r - 0.5 * v[i] * v[i])*t;
        double si = g[i] * sqrt(t);
        values[i] = s[i] * exp(mu+si);
    }
}

void MonteCarlo(MonteCarloData *data){
    double sum, var_sum, emp_stdev, price;
    int i;
    sum = var_sum = 0.0f;
    srand((unsigned)time(NULL));
    double r, t;
    if(data->numOpt == 1){
        r = data->sopt.r;
        t = data->sopt.t;
        for( i=0; i<data->path; i++){
            price = callPayoff(data->sopt);
            sum += price;
            var_sum += price * price;
        }
    }
    else{
        //vectors of brownian and ST
        double bt[N], s[N];
        double st_sum;
        int j;
        r = data->mopt.r;
        t = data->mopt.t;
        for(i=0; i<data->path; i++){
            st_sum = 0;
            //Simulation of stock prices
            simGaussVect(data->mopt.d, &data->mopt.p[0][0], bt);
            multiStockValue(data->mopt.s, data->mopt.v, bt, data->mopt.t, data->mopt.r, N, s);
            for(j=0;j<N;j++)
                st_sum += s[j]*data->mopt.w[j];
            price = (double)st_sum - data->mopt.k;
            if(price<0)
                price = 0.0f;
            sum += price;
            var_sum += price*price;
        }
    }
    
    price = exp(-r*t) * (sum/(double)data->path);
    emp_stdev = sqrt(
                     ((double)data->path * var_sum - sum * sum)
                     /
                     ((double)data->path * (double)(data->path - 1))
                     );
    
    data->callValue.Confidence = 1.96 * emp_stdev/sqrt(data->path);
    data->callValue.Expected = price;
}

void cvaMonteCarlo(MonteCarloData *data, double intdef, double lgd, int n_grid){
    double sum, var_sum, dt;
    int i, j, sims = data->path;
    OptionData option;
    
    sum = var_sum = 0.0f;
    srand((unsigned)time(NULL));
    
    option.s = data->sopt.s;
    option.t = data->sopt.t;
    option.r = data->sopt.r;
    option.v = data->sopt.v;
    option.k = data->sopt.k;
    
    dt = data->sopt.t / n_grid;
   
    for(i=0; i < sims; i++){
        double ee, mean_price, s;
        mean_price = 0;
        option.s = data->sopt.s;
        option.t = data->sopt.t;
        for(j=1; j <= n_grid; j++){
            double dp = exp(-(dt*(j-1))*intdef)-exp(-(dt*j)*intdef);
            s = geomBrownian(option.s,dt,option.r,option.v);
            option.t -= dt;
            if( option.t < 0)
                ee = 0;
            else
                ee = host_bsCall(option);
            /*  debug
            if(i==100){
                printf("Path: %d \nTime %.2f\n\n",j,option.t);
                printf("ee %f\n",ee);
                printf("dp %f\n",dp);
            }
             */
            mean_price += dp * ee;
            option.s = s;
        }
        mean_price *= lgd;
        sum += mean_price;
        var_sum += mean_price * mean_price;
    }
    
    // Closing Monte Carlo
    
    double emp_stdev, price;
    price = sum/(double)sims;
    emp_stdev = sqrt(((double)sims * var_sum - sum * sum)/((double)sims * (double)(sims - 1)));
    
    data->callValue.Confidence = 1.96 * emp_stdev/sqrt(sims);
    data->callValue.Expected = price;
}

/////////////////////////////////////////////
/////////////// WRAPPERS    /////////////////
/////////////////////////////////////////////

OptionValue host_vanillaOpt( OptionData option, int path){
    MonteCarloData data;
    data.sopt = option;
    data.numOpt = 1;
    data.path = path;
    
    MonteCarlo(&data);
    return data.callValue;
}

OptionValue host_basketOpt(MultiOptionData *option, int path){
    MonteCarloData data;
    data.mopt = *option;
    data.numOpt = N;
    data.path = path;
    
    MonteCarlo(&data);
    return data.callValue;
}

OptionValue host_cvaEquityOption(CVA *cva, int path){
    MonteCarloData data;
    // Option
    data.sopt = cva->option;
    data.numOpt = 1;
    data.path = path;
    
    cvaMonteCarlo(&data, cva->defInt, cva->lgd, cva->n);
    return data.callValue;
}


