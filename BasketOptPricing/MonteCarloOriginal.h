//
//  MonteCarlo.h
//  
//
//  Created by Marco Matteo Buzzulini on 07/12/17.
//

#ifndef MonteCarlo_h
#define MonteCarlo_h

#include "book.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 3
#define MAX_BLOCKS 64
#define MAX_THREADS 512
#define SIM 4000

//Switch to 1 for testing with random numbers
#define RAND 0


// Matrix structure
typedef struct{
    // the matrix is converted to a 1-dimensional vector
    // m[i][j] = m[j+i*cols] for i<rows and j<cols
    double *data;
    int rows;
    int cols;
} Matrix;

typedef struct {
    float s;    // stock price
    float k;    // strike price
    float r;    // risk-free rate
    float v;    // the volatility
    float t;    // time to maturity
} OptionData;

typedef struct{
    double *s;  //Stock vector
    double *v;  //Volatility vector
    double *p;  //Correlation matrix
    double *d;  //Drift vector
    double *w;  //Weight vector
    double k;
    double t;
    double r;
    int n;
} MultiOptionData;

typedef struct {
    double Expected;   // the simulated price
    double Confidence;    // confidence intervall
} OptionValue;

/////////////////////////////////////////////////////////////
// Matrix Functions
/////////////////////////////////////////////////////////////
void printVect( double *vet, int l );
void printMat( double *mat, int r, int c );
void mat_print( Matrix *mat );

void mat_init( Matrix *mat, int rows, int cols);
void mat_free( Matrix *mat );
void mat_setData( double *data, Matrix *mat );

void mat_prod( Matrix *first, Matrix *second, Matrix *result);
void ProdMat( double *first, double *second, int rows, int n, int cols, double *result);

double* Chol( Matrix *c );

// set Random Data
double* getRandomSigma(int n);
double* getRandomRho( int n );
double* getCovMat(double *std, double *rho, int n);
double randMinMax(double min, double max);
double gaussian(double mu, double sigma);

/////////////////////////////////////////////////////////////
// Option Functions
/////////////////////////////////////////////////////////////
void printOption ( OptionData o );
void printMultiOpt(MultiOptionData *o);
//simulation of stock values from a dependent random number g with the Geometric Brownian Motion
void multiStockValue(double *s, double *v, double *g, double t, double r, int n, double *values);
// simulation of a Gaussian Vector with drift vectors and volatility variance-covariance matrix reduced with cholevski algorithm
void simGaussVect(double *drift, double *volatility, int n, double *result);
// Monte Carlo functions
OptionValue CPUBasketOptCall(MultiOptionData *option, int sim);
void GPUBasketOpt(MultiOptionData *option, OptionValue *callValue);

#endif /* MonteCarlo_h */
