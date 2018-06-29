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

extern "C" OptionValue host_cvaEquityOption(CVA *, int);
extern "C" OptionValue dev_cvaEquityOption(CVA *, int, int, int);

extern "C" void printOption(OptionData o);

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
    
    printf("\nCVA of an European call Option\nIntensita di default %.2f, LGD %.2f\n",cva.defInt,cva.lgd);
    cva.option.v = V;
    cva.option.s = S;
    cva.option.t = T;
    cva.option.r = R;
    cva.option.k = K;
    cva.ns = 1;
    
    cudaEvent_t d_start, d_stop;
    int SIMS;
    float GPU_timeSpent=0, CPU_timeSpent=0;
    OptionValue dev_result, host_result;
    
	//	CUDA Parameters optimized
    printf("Inserisci il numero di simulazioni Monte Carlo(x131.072): ");
    scanf("%d",&SIMS);
    SIMS *= SIMPB;
    printf("\nScenari di Monte Carlo: %d\n",SIMS);
    
    printOption(cva.option);

	// Timer init
    CudaCheck( cudaEventCreate( &d_start ));
    CudaCheck( cudaEventCreate( &d_stop ));
    
    // CPU Monte Carlo
    
    printf("\nCVA execution on CPU...\n");
    CudaCheck( cudaEventRecord( d_start, 0 ));
    host_result = host_cvaEquityOption(&cva, SIMS);
    CudaCheck( cudaEventRecord( d_stop, 0));
    CudaCheck( cudaEventSynchronize( d_stop ));
    CudaCheck( cudaEventElapsedTime( &CPU_timeSpent, d_start, d_stop ));
    
    printf("\nCVA: %f\nConfidence Interval: %f\n\n",host_result.Expected, host_result.Confidence);
    printf("\nTotal execution time: %f s\n\n", CPU_timeSpent);
    printf("--------------------------------------------------\n");
    
    // GPU Monte Carlo
    printf("\nCVA execution on GPU...\n");
    CudaCheck( cudaEventRecord( d_start, 0 ));
    dev_result = dev_cvaEquityOption(&cva, BLOCKS, THREADS, SIMS);
    CudaCheck( cudaEventRecord( d_stop, 0));
    CudaCheck( cudaEventSynchronize( d_stop ));
    CudaCheck( cudaEventElapsedTime( &GPU_timeSpent, d_start, d_stop ));

    printf("\nTotal execution time: %f ms\n\n", GPU_timeSpent);
    printf("\nCVA: %f\nConfidence Interval: %f\n\n",dev_result.Expected, dev_result.Confidence);
    printf("Speed up: %f\n\n",CPU_timeSpent/GPU_timeSpent);
    return 0;
}

