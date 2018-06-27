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

#define N 30

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

/*
typedef struct {
    double creditspread;    // credit spread
    double fundingspread;   // funding spread
    double lgd;    			// loss given default
} CreditData;
*/

// MultiOptionData
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

typedef struct{
    // Expected Exposures
    OptionValue *ee;
    // Default probabilities
    double defInt, lgd, *dp;
    // Option data
    int ns; 
    MultiOptionData opt;
    OptionData option;
    // CVA
    double cva;
    // Num of simulations
    int n;
}CVA;

// Struct for Monte Carlo methods
typedef struct{
	OptionValue callValue;
	MultiOptionData mopt;
    OptionData sopt;
    int numOpt, path;
} MonteCarloData;

#endif /* MONTECARLO_H_ */
