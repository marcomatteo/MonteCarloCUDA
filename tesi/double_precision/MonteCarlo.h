/*
 * MonteCarlo.h
 *
 *  Created on: 06/feb/2018
 *      Author: marco
 */

#ifndef MONTECARLO_H_
#define MONTECARLO_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define N 1
#define THREADS 2
#define NTHREADS 128
#define BLOCKS 256
#define PATH 40
#define SIMPB 10240

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
    double s;    // stock price
    double k;    // strike price
    double r;    // risk-free rate
    double v;    // the volatility
    double t;    // time to maturity
} OptionData;

// Static MultiOptionData
typedef struct{
    double s[N];  	//Stock vector
    double v[N];  	//Volatility vector
    double p[N][N]; //Correlation matrix
    double d[N];  	//Drift vector
    double w[N];  	//Weight vector
    double k;
    double t;
    double r;
} MultiOptionData;

typedef struct {
    double Expected;   	// the simulated price
    double Confidence;    // confidence intervall
} OptionValue;

/* typedef struct {
 float creditspread;    // credit spread
 float fundingspread;   // funding spread
 float lgd;                // loss given default
 } CreditData; */

typedef struct{
    // Expected Exposures
    OptionValue *ee;
    // Default probabilities
    double *dp, defInt, lgd;
    // double *fp; // Founding probabilities
    // Option data
    MultiOptionData opt;
    // CVA
    double cva;
    // FVA
    //double fva;
    // Num of simulations
    int n;
}CVA;

// Struct for Monte Carlo methods
typedef struct{
	OptionValue callValue;
	MultiOptionData option;
    int numOpt, path;
} MonteCarloData;

#endif /* MONTECARLO_H_ */
