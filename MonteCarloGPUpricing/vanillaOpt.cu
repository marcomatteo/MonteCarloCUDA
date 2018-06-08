//
//  MonteCarlo.cu
//  tesi
//
//  Created by Marco Matteo Buzzulini on 27/11/17.
//  Copyright © 2017 Marco Matteo Buzzulini. All rights reserved.
//

#include "MonteCarlo.h"
#include <cuda_runtime.h>

// includes, project
#include <helper_functions.h> // Helper functions (utilities, parsing, timing)
#include <helper_cuda.h>      // helper functions (cuda error checking and initialization)
#include <multithreading.h>

extern "C" double host_bsCall ( OptionData );
extern "C" OptionValue host_vanillaOpt(OptionData, int);
extern "C" OptionValue dev_vanillaOpt(OptionData *, int, int);
extern "C" void printOption( OptionData o);
extern "C" void Parameters(int *numBlocks, int *numThreads);

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
	int numBlocks, numThreads[THREADS], i;
	int SIMS;
	OptionValue CPU_sim, GPU_sim[THREADS];
	float d_CPU_timeSpent=0, GPU_timeSpent[THREADS], speedup[THREADS];
	double price, bs_price, difference[THREADS];
	cudaEvent_t d_start, d_stop;

    /*--------------------------- START PROGRAM -----------------------------------*/
	printf("Vanilla Option Pricing\n");
	// CUDA parameters for parallel execution
	Parameters(&numBlocks, numThreads);
	SIMS = numBlocks*PATH;
	printf("\nScenari di Monte Carlo: %d\n",SIMS);
	//	Print Option details
	printOption(option);
	// Time instructions
    CudaCheck( cudaEventCreate( &d_start ));
    CudaCheck( cudaEventCreate( &d_stop ));
    //	Black & Scholes price
    bs_price = host_bsCall(option);
    printf("\nPrezzo Black & Scholes: %f\n",bs_price);

    // CPU Monte Carlo
    printf("\nMonte Carlo execution on CPU:\nN^ simulations: %d\n",SIMS);
    CudaCheck( cudaEventRecord( d_start, 0 ));
    CPU_sim=host_vanillaOpt(option, SIMS);
    CudaCheck( cudaEventRecord( d_stop, 0));
    CudaCheck( cudaEventSynchronize( d_stop ));
    CudaCheck( cudaEventElapsedTime( &d_CPU_timeSpent, d_start, d_stop ));
    d_CPU_timeSpent /= 1000;
    price = CPU_sim.Expected;

    // GPU Monte Carlo
    printf("\nMonte Carlo execution on GPU:\nN^ simulations: %d\n",SIMS);
    for(i=0; i<THREADS; i++){
    	CudaCheck( cudaEventRecord( d_start, 0 ));
    	GPU_sim[i] = dev_vanillaOpt(&option, numBlocks, numThreads[i]);
        CudaCheck( cudaEventRecord( d_stop, 0));
   	    CudaCheck( cudaEventSynchronize( d_stop ));
   	    CudaCheck( cudaEventElapsedTime( &GPU_timeSpent[i], d_start, d_stop ));
   	    GPU_timeSpent[i] /= 1000;
   	    difference[i] = abs(GPU_sim[i].Expected - bs_price);
   	    speedup[i] = abs(d_CPU_timeSpent / GPU_timeSpent[i]);
    }

    // Comparing time spent with the two methods
    printf( "\n-\tResults:\t-\n");
    printf("Simulated price for the option with CPU: € %f with I.C. %f\n", price, CPU_sim.Confidence);
    printf("Total execution time CPU: %f s with device function\n\n", d_CPU_timeSpent);
    printf("Simulated price for the option with GPU:\n");
    printf("  : NumThreads : Price : Confidence Interval : Difference from BS price :  Time : Speedup :");
    printf("\n");
    for(i=0; i<THREADS; i++){
    	printf(": \t %d ",numThreads[i]);
    	printf(" \t %f ",GPU_sim[i].Expected);
    	printf(" \t %f  ",GPU_sim[i].Confidence);
    	printf(" \t %f \t",difference[i]);
    	printf(" \t %f ",GPU_timeSpent[i]);
    	printf(" \t %.2f \t",speedup[i]);
    	printf(":\n");
    }
    
    CudaCheck( cudaEventDestroy( d_start ));
    CudaCheck( cudaEventDestroy( d_stop ));
    return 0;
}
