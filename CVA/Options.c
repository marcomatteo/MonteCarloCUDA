/*
 * Options.c
 *
 *  Created on: 06/feb/2018
 *      Author: marco
 */

#include "Options.h"

void printOption( OptionData o){
    printf("\n-\tOption data\t-\n\n");
    printf("Underlying asset price:\t € %.2f\n", o.s);
    printf("Strike price:\t\t € %.2f\n", o.k);
    printf("Risk free interest rate: %.2f %%\n", o.r * 100);
    printf("Volatility:\t\t\t %.2f %%\n", o.v * 100);
    printf("Time to maturity:\t\t %.2f %s\n", o.t, (o.t>1)?("years"):("year"));
}

void printMultiOpt( MultiOptionData *o){
    int n=o->n;
    printf("\n-\tBasket Option data\t-\n\n");
    printf("Number of assets: %d\n",n);
    printf("Underlying assets prices:\n");
    printVect(o->s, n);
    printf("Volatility:\n");
    printVect(o->v, n);
    printf("Weights:");
    printVect(o->w, n);
    printf("Correlation matrix:\n");
    printMat(o->p, n, n);
    printf("Strike price:\t € %.2f\n", o->k);
    printf("Risk free interest rate %.2f \n", o->r);
    printf("Time to maturity:\t %.2f %s\n", o->t, (o->t>1)?("years"):("year"));
}

