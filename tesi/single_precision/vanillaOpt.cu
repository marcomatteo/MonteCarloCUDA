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
extern "C" OptionValue dev_vanillaOpt(OptionData *, int, int, int);
extern "C" void printOption( OptionData o);

void Parameters(int *numBlocks, int *numThreads);
void memAdjust(cudaDeviceProp *deviceProp, int *numThreads);
void sizeAdjust(cudaDeviceProp *deviceProp, int *numBlocks, int *numThreads);

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
	float price, bs_price, difference[THREADS];
	cudaEvent_t d_start, d_stop;

    /*--------------------------- START PROGRAM -----------------------------------*/
	printf("Vanilla Option Pricing\n");
	// CUDA parameters for parallel execution
	Parameters(&numBlocks, numThreads);
    printf("Inserisci il numero di simulazioni (x100.000): ");
    scanf("%d",&SIMS);
    SIMS *= 100000;
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
    CudaCheck( cudaEventElapsedTime( &d_CPU_timeSpent, d_start, d_stop ));
    d_CPU_timeSpent /= 1000;
    price = CPU_sim.Expected;

    // GPU Monte Carlo
    printf("\nMonte Carlo execution on GPU:\n");
    printf("(NumBlocks, NumSimulations): ( %d ; %d )\n",BLOCKS,SIMS/BLOCKS);
    for(i=0; i<THREADS; i++){
    	CudaCheck( cudaEventRecord( d_start, 0 ));
    	GPU_sim[i] = dev_vanillaOpt(&option, numBlocks, numThreads[i],SIMS);
        CudaCheck( cudaEventRecord( d_stop, 0));
   	    CudaCheck( cudaEventSynchronize( d_stop ));
   	    CudaCheck( cudaEventElapsedTime( &GPU_timeSpent[i], d_start, d_stop ));
   	    GPU_timeSpent[i] /= 1000;
   	    difference[i] = abs(GPU_sim[i].Expected - bs_price);
   	    speedup[i] = abs(d_CPU_timeSpent / GPU_timeSpent[i]);
    }

    // Comparing time spent with the two methods
    printf( "\n-\tResults:\t-\n");
    printf("Simulated price for the option with CPU: Expected price, I.C.\n %f \nI.C. %f\n", price, CPU_sim.Confidence);
    printf("Total execution time CPU: %f\n\n", d_CPU_timeSpent);
    printf("Simulated price for the option with GPU:\n");
    for(i=0; i<THREADS; i++){
        printf("  : NumThreads : Price : Confidence Interval : Difference from BS price :  Time : Speedup :");
        printf("\n");
    	printf("%d \n",numThreads[i]);
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
///////////////////////////////////
//    ADJUST FUNCTIONS
///////////////////////////////////

void sizeAdjust(cudaDeviceProp *deviceProp, int *numBlocks, int *numThreads){
    int maxGridSize = deviceProp->maxGridSize[0];
    int maxBlockSize = deviceProp->maxThreadsPerBlock;
    //    Replacing in case of wrong size
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
    int sizeDouble = sizeof(float);
    int numShared = sizeDouble * *numThreads * 2;
    if(sizeof(MultiOptionData) > maxConstant){
        printf("\nWarning: Excess use of constant memory: %zu\n",maxConstant);
        printf("A float variable size is: %d\n",sizeDouble);
        printf("In a MultiOptionData struct there's a consumption of %zu constant memory\n",sizeof(MultiOptionData));
        printf("In this Basket Option there's %d stocks\n",N);
        int maxDim = (int)maxConstant/(sizeDouble*5);
        printf("The optimal number of dims should be: %d stocks\n",maxDim);
    }
    if(numShared > maxShared){
        printf("\nWarning: Excess use of shared memory: %zu\n",maxShared);
        printf("A float variable size is: %d\n",sizeDouble);
        int maxThreads = (int)maxShared / (2*sizeDouble);
        printf("The optimal number of thread should be: %d\n",maxThreads);
    }
}

void Parameters(int *numBlocks, int *numThreads){
    cudaDeviceProp deviceProp;
    int i = 0;
    CudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    numThreads[0] = 128;
    numThreads[1] = 256;
    numThreads[2] = 512;
    numThreads[3] = 1024;
    *numBlocks = BLOCKS;
    //printf("\nParametri CUDA:\n");
    for (i=0; i<THREADS; i++) {
        sizeAdjust(&deviceProp,numBlocks, &numThreads[i]);
        memAdjust(&deviceProp, &numThreads[i]);
    }
}
