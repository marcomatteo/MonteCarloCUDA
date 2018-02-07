/*
 * MonteCarlo.h
 *
 *  Created on: 06/feb/2018
 *      Author: marco
 */

#ifndef MONTECARLO_H_
#define MONTECARLO_H_

#include "Matrix.h"

#include <math.h>
#include <time.h>
#include <stdlib.h>

#ifndef N
#define N 3
#endif	/* N */
#ifndef RAND
#define RAND 0
#endif	/* RAND	*/
#ifndef MAX_BLOCKS
#define MAX_BLOCKS 1000
#endif	/* MAX_BLOCKS	*/
#ifndef	MAX_THREADS
#define MAX_THREADS 1024
#endif	/*	MAX_THREADS	*/
#ifndef	SIM
#define SIM 10000
#endif	/*	SIM	*/

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#ifndef CudaCheck
#define CudaCheck(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
#endif

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
// Option Functions
/////////////////////////////////////////////////////////////
/*
void printOption ( OptionData o );
void printMultiOpt(MultiOptionData *o);

// set Random Data
double* getRandomSigma(int n);
double* getRandomRho( int n );
double* getCovMat(double *std, double *rho, int n);
double randMinMax(double min, double max);
double gaussian(double mu, double sigma);

//simulation of stock values from a dependent random number g with the Geometric Brownian Motion
void multiStockValue(double *s, double *v, double *g, double t, double r, int n, double *values);

// simulation of a Gaussian Vector with drift vectors and volatility variance-covariance matrix reduced with cholevski algorithm
void simGaussVect(double *drift, double *volatility, int n, double *result);
*/

// Monte Carlo functions
extern "C" OptionValue CPUBasketOptCall(MultiOptionData *option, int sim);
extern "C" void GPUBasketOpt(MultiOptionData *option, OptionValue *callValue);

#endif /* MONTECARLO_H_ */
