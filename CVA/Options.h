/*
 * Options.h
 *
 *  Created on: 06/feb/2018
 *      Author: marco
 */

#ifndef OPTIONS_H_
#define OPTIONS_H_

#include "Matrix.h"

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
void printOption ( OptionData o );
void printMultiOpt(MultiOptionData *o);

#endif /* OPTIONS_H_ */
