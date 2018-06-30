/*
 *  cvaOpt.cu
 *  Monte Carlo methods in CUDA
 *  Dissertation project
 *  Created on: 06/feb/2018
 *  Author: Marco Matteo Buzzulini
 *  Copyright © 2018 Marco Matteo Buzzulini. All rights reserved.
 */

#include "MonteCarlo.h"

#define THREADS 128
#define BLOCKS 512
#define PATH 250
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
    /*
     printf("\nMonte Carlo multiplier (x131.072): ");
     scanf("%d",&SIMS);
     SIMS *= SIMPB;
     */
    // Display
    /*
     printf("\n \t Now running:\n");
     printf("\nMonte Carlo simulations: %d\n",SIMS);
     printf("CVA simulation path: %d\n",PATH);
     printf("Loop interactions: %d\n",PATH*SIMS);
     */
    // Timer init
    CudaCheck( cudaEventCreate( &d_start ));
    CudaCheck( cudaEventCreate( &d_stop ));
    int i,k;
    int paths[5] = { 25, 50, 75, 250, 500};
    for(k=0;k<5;k++){
        SIMS = paths[k];
        for(i=0; i<4; i++){
            // Display
            printf("\n \t Now running:\n");
            printf("\nMonte Carlo simulations: %d\n",SIMPB);
            printf("CVA simulation path: %d\n",SIMS);
            printf("Loop interactions: %d\n",SIMS*SIMPB);
            cva.n = SIMS;
            /*
             printf("\nCVA execution on CPU...\n");
             CudaCheck( cudaEventRecord( d_start, 0 ));
             host_result = host_cvaEquityOption(&cva, SIMPB);
             CudaCheck( cudaEventRecord( d_stop, 0));
             CudaCheck( cudaEventSynchronize( d_stop ));
             CudaCheck( cudaEventElapsedTime( &CPU_timeSpent, d_start, d_stop ));
             
             printf("\nCVA: \n%f \n",host_result.Expected);
             printf("\nTotal execution time: (ms) \n%f \n\n", CPU_timeSpent);
             printf("--------------------------------------------------\n");
             */
            // GPU Monte Carlo
            printf("\nCVA execution on GPU...\n");
            int j, th;
            th = 2;
            for(j=0;j<(i+6);j++)
                th *= 2;
            printf("\n %d BLOCKS / %d THREADS / %d SIMS\n",BLOCKS, th, SIMPB);
            CudaCheck( cudaEventRecord( d_start, 0 ));
            dev_result = dev_cvaEquityOption(&cva, BLOCKS, th, SIMPB);
            CudaCheck( cudaEventRecord( d_stop, 0));
            CudaCheck( cudaEventSynchronize( d_stop ));
            CudaCheck( cudaEventElapsedTime( &GPU_timeSpent, d_start, d_stop ));
            
            printf("\nTotal execution time: (ms) \n%f \n", GPU_timeSpent);
            printf("\nGPU speedup: \n%f \n",CPU_timeSpent/GPU_timeSpent);
            printf("\nCVA: \n%f \n\n",dev_result.Expected);
        }
    }
    return 0;
}

