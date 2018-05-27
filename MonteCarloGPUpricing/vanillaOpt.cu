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

#define RISULTATI 4

//	Host Black & Scholes
extern "C" double host_bsCall ( OptionData );

//	Host MonteCarlo
extern "C" OptionValue host_vanillaOpt(OptionData, int);

//	Device MonteCarlo
extern "C" OptionValue dev_vanillaOpt(OptionData *, int, int);

///////////////////////////////////
//	PRINT FUNCTIONS
///////////////////////////////////

void printOption( OptionData o){
    printf("\n-\tOption data\t-\n\n");
    printf("Underlying asset price:\t € %.2f\n", o.s);
    printf("Strike price:\t\t € %.2f\n", o.k);
    printf("Risk free interest rate: %.2f\n", o.r);
    printf("Volatility:\t\t %.2f\n", o.v);
    printf("Time to maturity:\t %.2f %s\n", o.t, (o.t>1)?("years"):("year"));
}


///////////////////////////////////
//	ADJUST FUNCTIONS
///////////////////////////////////

void sizeAdjust(cudaDeviceProp *deviceProp, int *numBlocks, int *numThreads){
	int maxGridSize = deviceProp->maxGridSize[0];
	int maxBlockSize = deviceProp->maxThreadsPerBlock;
	//	Replacing in case of wrong size
	if(*numBlocks > maxGridSize){
		*numBlocks = maxGridSize;
		printf("Warning: maximum size of Grid is %d",*numBlocks);
	}
	if(*numThreads > maxBlockSize){
		*numThreads = maxBlockSize;
		printf("Warning: maximum size of Blocks is %d",*numThreads);
	}
}

void memAdjust(cudaDeviceProp *deviceProp, int *numThreads){
		size_t maxShared = deviceProp->sharedMemPerBlock;
		size_t maxConstant = deviceProp->totalConstMem;
		int sizeDouble = sizeof(double);
		int numShared = sizeDouble * *numThreads * 2;
		if(sizeof(MultiOptionData) > maxConstant){
			printf("\nWarning: Excess use of constant memory: %zu\n",maxConstant);
			printf("A double variable size is: %d\n",sizeDouble);
			printf("In a MultiOptionData struct there's a consumption of %zu constant memory\n",sizeof(MultiOptionData));
			printf("In this Basket Option there's %d stocks\n",N);
			int maxDim = (int)maxConstant/(sizeDouble*5);
			printf("The optimal number of dims should be: %d stocks\n",maxDim);
		}
		if(numShared > maxShared){
			printf("\nWarning: Excess use of shared memory: %zu\n",maxShared);
			printf("A double variable size is: %d\n",sizeDouble);
			int maxThreads = (int)maxShared / (2*sizeDouble);
			printf("The optimal number of thread should be: %d\n",maxThreads);
		}
		printf("\n");
}

void optimalAdjust(cudaDeviceProp *deviceProp, int *numBlocks, int *numThreads){
	int multiProcessors = deviceProp->multiProcessorCount;
	int cudaCoresPM = _ConvertSMVer2Cores(deviceProp->major, deviceProp->minor);
	*numBlocks = multiProcessors * 40;
	*numThreads = pow(2,(int)(log(cudaCoresPM)/log(2)))*2;
	sizeAdjust(deviceProp,numBlocks, numThreads);
}

void choseParameters(int *numBlocks, int *numThreads){
		cudaDeviceProp deviceProp;
		CudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
		char risp;
		printf("\nParametri CUDA:\n");
		printf("Scegli il numero di Blocchi: ");
		scanf("%d",numBlocks);
		printf("Scegli il numero di Threads per blocco: ");
		scanf("%d",numThreads);
		printf("Vuoi ottimizzare i parametri inseriti? (Y/N) ");
		scanf("%s",&risp);
		if((risp=='Y')||(risp=='y'))
			optimalAdjust(&deviceProp,numBlocks, numThreads);
		else
			sizeAdjust(&deviceProp,numBlocks, numThreads);
		memAdjust(&deviceProp,numThreads);
}

void Parameters(int *numBlocks, int *numThreads){
		cudaDeviceProp deviceProp;
		CudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
		numThreads[0] = 128;
		numThreads[1] = 256;
		numThreads[2] = 512;
		numThreads[3] = 1024;
		printf("\nParametri CUDA:\n");
		printf("Scegli il numero di Blocchi: ");
		scanf("%d",numBlocks);
}

////////////////////////////////////////////////////////////////////////////////////////
//                                      MAIN
////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[]) {
    /*--------------------------- Variabili -----------------------------------*/
	OptionData option;
	option.v = 0.2;
	option.s = 100;
	option.k= 100.f;
	option.r= 0.048790164;
	option.t= 1.f;

	int numBlocks, numThreads[RISULTATI], i;
	int SIMS;

	OptionValue CPU_sim, GPU_sim[RISULTATI];

	float d_CPU_timeSpent=0, GPU_timeSpent[RISULTATI], speedup[RISULTATI];
	double price, bs_price, difference[RISULTATI];
	cudaEvent_t d_start, d_stop;
	clock_t h_start, h_stop;

    /*--------------------------- Programma -----------------------------------*/

	printf("Vanilla Option Pricing\n");

	//	Definizione e stampa dei parametri CUDA per l'esecuzione in parallelo
	Parameters(&numBlocks, numThreads);
	SIMS = numBlocks*PATH;

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
    for(i=0; i<RISULTATI; i++){
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
    for(i=0; i<RISULTATI; i++){
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
