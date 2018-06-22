//
//  MonteCarlo.cu
//  tesi
//
//  Created by Marco Matteo Buzzulini on 27/11/17.
//  Copyright © 2017 Marco Matteo Buzzulini. All rights reserved.
//

#include "MonteCarlo.h"

extern "C" float host_bsCall ( OptionData );
extern "C" OptionValue host_vanillaOpt(OptionData, int);
extern "C" OptionValue dev_vanillaOpt(OptionData *, int, int);
extern "C" void printOption( OptionData o);

void Parameters(int *numBlocks, int *numThreads);
void memAdjust(cudaDeviceProp *deviceProp, int *numThreads);
void sizeAdjust(cudaDeviceProp *deviceProp, int *numBlocks, int *numThreads);

const int threadsPerBlock[NTHREADS] = {128, 256, 512, 1024};

////////////////////////////////////////////////////////////////////////////////////////
//                                      MAIN
////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[]) {
    /*--------------------------- VARIABLES -----------------------------------*/
	// Option Data
	OptionData option;
	option.v = 0.2;
	option.s = 100;
	option.k= 100.f;
	option.r= 0.048790164;
	option.t= 1.f;
	// Simulation
	int i, SIMS;
	OptionValue CPU_sim, GPU_sim[NTHREADS];
	float CPU_timeSpent=0, GPU_timeSpent[NTHREADS], speedup[NTHREADS];
	float bs_price, difference[NTHREADS];
	cudaEvent_t d_start, d_stop;

    /*--------------------------- START PROGRAM -----------------------------------*/
	printf("Vanilla Option Pricing\n");
	// CUDA parameters for parallel execution
    printf("Inserisci il numero di simulazioni (x131.072): ");
    scanf("%d",&SIMS);
    SIMS *= 131072;
	//printf("\nScenari di Monte Carlo: %d\n",SIMS);
	//	Print Option details
	printOption(option);
	// Time instructions
    CudaCheck( cudaEventCreate( &d_start ));
    CudaCheck( cudaEventCreate( &d_stop ));
    //	Black & Scholes price
    bs_price = host_bsCall(option);
    printf("\nPrezzo Black & Scholes: %f\n",bs_price);

    // CPU Monte Carlo
    printf("\nMonte Carlo execution on CPU:\n");
    printf("N^ simulations: %d\n",SIMS);
    CudaCheck( cudaEventRecord( d_start, 0 ));
    CPU_sim=host_vanillaOpt(option, SIMS);
    CudaCheck( cudaEventRecord( d_stop, 0));
    CudaCheck( cudaEventSynchronize( d_stop ));
    CudaCheck( cudaEventElapsedTime( &CPU_timeSpent, d_start, d_stop ));
    CPU_timeSpent /= 1000;

    // GPU Monte Carlo
    printf("\nMonte Carlo execution on GPU:\n");
    printf("(NumBlocks, NumSimulations): ( %d ; %d )\n",BLOCKS,SIMS/BLOCKS);
    for(i=0; i<NTHREADS; i++){
    	CudaCheck( cudaEventRecord( d_start, 0 ));
    	GPU_sim[i] = dev_vanillaOpt(&option, threadsPerBlock[i], SIMS);
        CudaCheck( cudaEventRecord( d_stop, 0));
   	    CudaCheck( cudaEventSynchronize( d_stop ));
   	    CudaCheck( cudaEventElapsedTime( &GPU_timeSpent[i], d_start, d_stop ));
   	    GPU_timeSpent[i] /= 1000;
   	    difference[i] = abs(GPU_sim[i].Expected - bs_price);
   	    speedup[i] = abs(CPU_timeSpent / GPU_timeSpent[i]);
    }

    // Comparing time spent with the two methods
    printf( "\n-\tResults:\t-\n");
    printf("Simulated price for the option with CPU: Expected price, I.C., time\n%f \n%f \n%f \n",  CPU_sim.Expected, CPU_sim.Confidence, CPU_timeSpent);
    printf("Simulated price for the option with GPU:\n");
    printf("  : NumThreads : Price : Confidence Interval : Difference from BS price :  Time : Speedup :");
    printf("\n");
    for(i=0; i<NTHREADS; i++){
    	printf("%d \n",threadsPerBlock[i]);
    	printf("%f \n",GPU_sim[i].Expected);
    	printf("%f \n",GPU_sim[i].Confidence);
    	printf("%f \n",difference[i]);
    	printf("%f \n",GPU_timeSpent[i]);
    	printf("%.2f \n",speedup[i]);
    	printf("---\n");
    }
    
    CudaCheck( cudaEventDestroy( d_start ));
    CudaCheck( cudaEventDestroy( d_stop ));
    return 0;
}
