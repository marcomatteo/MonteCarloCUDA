//
//  MonteCarlo.cu
//  tesi
//
//  Created by Marco Matteo Buzzulini on 27/11/17.
//  Copyright © 2017 Marco Matteo Buzzulini. All rights reserved.
//

#include "MonteCarlo.h"

#define THREADS 128
#define BLOCKS 512
#define PATH 50
#define SIMPB 131072

extern "C" OptionValue host_cvaEquityOption(CVA *, int);
extern "C" OptionValue dev_cvaEquityOption(CVA *, int, int, int);

extern "C" void printOption( OptionData o);

const float defInt = 0.03;
const float recoveryRate = 0.4;
const float S = 100;
const float K = 100;
const float R = 0.05;
const float V = 0.2;
const float T = 1.f;

int main(int argc, const char * argv[]) {
    /*--------------------------- DATA INSTRUCTION -----------------------------------*/
    CVA cva;
    cva.defInt = defInt;
    cva.lgd = (1 - recoveryRate);
    cva.n = PATH;
    
    printf("\nCVA of an European call Option\nDefault intensity %.2f, LGD %.2f\n",cva.defInt,cva.lgd);
    cva.option.v = V;
    cva.option.s = S;
    cva.option.t = T;
    cva.option.r = R;
    cva.option.k = K;
    cva.ns = 1;
    
    cudaEvent_t d_start, d_stop;
    int SIMS;
    float GPU_timeSpent=0, CPU_timeSpent=0;
    OptionValue dev_result = {0,0}, host_result = {0,0};
    
    // Option
    printOption(cva.option);
    
    //    CUDA Parameters optimized
    printf("Monte Carlo simulations (x131.072): ");
    scanf("%d",&SIMS);
    SIMS *= SIMPB;
    
    // Display
    printf("\nMonte Carlo simulations: %d\n",SIMS);
    printf("CVA simulation path: %d\n",PATH);
    printf("Loop interactions: %d\n",PATH*SIMS);
    
    
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
    
    printf("\nCVA: \n%f \nConfidence Interval: \n%f \n\n",host_result.Expected, host_result.Confidence);
    printf("\nTotal execution time: (ms)\n%f\n\n", CPU_timeSpent);
    printf("--------------------------------------------------\n");
    
    // GPU Monte Carlo
    printf("\nCVA execution on GPU...\n");
    int i;
    for(i=7; i<11; i++){
        int j, th=2;
        for(j=0;j<i;j++)
            th *= 2;
        CudaCheck( cudaEventRecord( d_start, 0 ));
        dev_result = dev_cvaEquityOption(&cva, BLOCKS, th, SIMS);
        CudaCheck( cudaEventRecord( d_stop, 0));
        CudaCheck( cudaEventSynchronize( d_stop ));
        CudaCheck( cudaEventElapsedTime( &GPU_timeSpent, d_start, d_stop ));
        
        printf("\nTotal execution time: (ms) \n%f \n\n", GPU_timeSpent);
        printf("\nCVA: %f\nConfidence Interval: \n%f \n\n",dev_result.Expected, dev_result.Confidence);
        printf("GPU speedup: \n%f \n\n",CPU_timeSpent/GPU_timeSpent);
    }
    return 0;
}

