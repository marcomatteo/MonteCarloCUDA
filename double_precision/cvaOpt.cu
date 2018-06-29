//
//  MonteCarlo.cu
//  tesi
//
//  Created by Marco Matteo Buzzulini on 27/11/17.
//  Copyright © 2017 Marco Matteo Buzzulini. All rights reserved.
//

#include "MonteCarlo.h"

#define THREADS 256
#define BLOCKS 512
#define PATH 40
#define SIMPB 131072

extern "C" double host_bsCall ( OptionData );
extern "C" void host_cvaEquityOption(CVA *, int);
extern "C" OptionValue dev_cvaEquityOption(CVA *, int, int, int);
extern "C" void printOption( OptionData o);
extern "C" void Chol( double c[N][N], double a[N][N] );
extern "C" void printMultiOpt( MultiOptionData *o);
extern "C" double randMinMax(double min, double max);

void getRandomSigma( double* std );
void getRandomRho( double* rho );
void pushVett( double* vet, double x );

const double defInt = 0.03;
const double recoveryRate = 0.4;
const double S = 100;
const double K = 100;
const double R = 0.05;
const double V = 0.2;
const double T = 1.f;

int main(int argc, const char * argv[]) {
    /*--------------------------- DATA INSTRUCTION -----------------------------------*/
    CVA cva;
    cva.defInt = defInt;
    cva.lgd = (1 - recoveryRate);
    cva.n = PATH;
    
    // n+1 because it starts from 1
    double *bs_price = (double*)malloc(sizeof(double)*(cva.n+1));

    OptionValue result;
    printf("\nCVA of an European call Option\nIntensita di default %.2f, LGD %.2f\n",cva.defInt,cva.lgd);
    cva.option.v = V;
    cva.option.s = S;
    cva.option.t = T;
    cva.option.r = R;
    cva.option.k = K;
    cva.ns = 1;
    
    cudaEvent_t d_start, d_stop;
    int i, j, SIMS;
    double dt, cholRho[N][N];
    float GPU_timeSpent=0;
    
	//	CUDA Parameters optimized
    printf("Inserisci il numero di simulazioni Monte Carlo(x131.072): ");
    scanf("%d",&SIMS);
    SIMS *= SIMPB;
    printf("\nScenari di Monte Carlo: %d\n",SIMS);
    
    printOption(cva.option);
    bs_price[0] = host_bsCall(cva.option);
    int n = cva.option.t;
    dt = cva.option.t/(double)cva.n;
    for(i=1;i<cva.n+1;i++){
        if((cva.option.t -= dt)<0)
            bs_price[i] = 0;
        else
            bs_price[i] = host_bsCall(cva.option);
    }
    cva.option.t = n;
    

	// Timer init
    CudaCheck( cudaEventCreate( &d_start ));
    CudaCheck( cudaEventCreate( &d_stop ));
    
    // CPU Monte Carlo
    /*
    printf("\nCVA execution on CPU:\n");
    CudaCheck( cudaEventRecord( d_start, 0 ));
    host_cvaEquityOption(&cva, SIMS);
    CudaCheck( cudaEventRecord( d_stop, 0));
    CudaCheck( cudaEventSynchronize( d_stop ));
    CudaCheck( cudaEventElapsedTime( &CPU_timeSpent, d_start, d_stop ));
    CPU_timeSpent /= 1000;
    printf("\nPrezzi Simulati:\n");
    
        printf("|\ti\t\t|\tPrezzi BS\t| Differenza Prezzi\t|\tPrezzi\t\t|\tDefault Prob\t|\n");
        for(i=0;i<cva.n+1;i++){
            difference = abs(cva.ee[i].Expected - bs_price[i]);
            printf("|\t%f\t|\t%f\t|\t%f\t|\t%f\t|\t%f\t|\n",dt*i,bs_price[i],difference,cva.ee[i].Expected,cva.dp[i]);
        }
    
    printf("\nCVA: %f\n\n",cva.cva);
    printf("\nTotal execution time: %f s\n\n", CPU_timeSpent);
    printf("--------------------------------------------------\n");
     */
    // GPU Monte Carlo
    printf("\nCVA execution on GPU:\n");
    CudaCheck( cudaEventRecord( d_start, 0 ));
    result = dev_cvaEquityOption(&cva, BLOCKS, THREADS, SIMS);
    CudaCheck( cudaEventRecord( d_stop, 0));
    CudaCheck( cudaEventSynchronize( d_stop ));
    CudaCheck( cudaEventElapsedTime( &GPU_timeSpent, d_start, d_stop ));
    //GPU_timeSpent /= 1000;

    printf("\nTotal execution time: %f ms\n\n", GPU_timeSpent);
    printf("\nCVA: %f\n\n",result.Expected);
    //printf("Speed up: %f\n\n",CPU_timeSpent/GPU_timeSpent);
   	free(bs_price);
    return 0;
}

