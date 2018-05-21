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

//	Host Black & Scholes
extern "C" double host_bsCall ( OptionData );

//	Host MonteCarlo
extern "C" OptionValue host_vanillaOpt(OptionData, int);

//	Device MonteCarlo
extern "C" OptionValue dev_vanillaOpt(OptionData *, int, int);

//	CVA: per ora è in test la simulazione delle Expected Exposures
extern "C" void dev_cvaEquityOption(CVA *cva, int numBlocks, int numThreads);


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
	*numThreads = pow(2,(int)(log(cudaCoresPM)/log(2)));
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

////////////////////////////////////////////////////////////////////////////////////////
//                                      MAIN
////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[]) {
    /*--------------------------- DATA INSTRUCTION -----------------------------------*/
	OptionData option;
	option.v = 0.25;
	option.s = 100;
	option.k= 100.f;
	option.r= 0.05;
	option.t= 1.f;
	int i;

	printf("Expected Exposures of an Equity Option\n");

	//	Definizione dei parametri CUDA per l'esecuzione in parallelo
	int numBlocks, numThreads;
	choseParameters(&numBlocks, &numThreads);

	printf("Simulazione di ( %d ; %d )\n",numBlocks, numThreads);
	int SIMS = numBlocks*PATH;

	//	Print Option details
	printOption(option);

	CVA cva;
	cva.n = 40;
	cva.credit.creditspread=150;
	cva.credit.fundingspread=75;
	cva.credit.lgd=60;
	cva.opt = option;
	cva.dp = (double*)malloc((cva.n+1)*sizeof(double));
	cva.fp = (double*)malloc((cva.n+1)*sizeof(double));
	// Puntatore al vettore di prezzi simulati, n+1 perché il primo prezzo è quello originale
	cva.ee = (OptionValue *)malloc(sizeof(OptionValue)*(cva.n+1));

    //float CPU_timeSpent=0, speedup;
    float GPU_timeSpent=0;
    double *price = (double*)malloc(sizeof(double)*(cva.n+1));
    double *bs_price = (double*)malloc(sizeof(double)*(cva.n+1));
    double difference;

    cudaEvent_t d_start, d_stop;
    CudaCheck( cudaEventCreate( &d_start ));
    CudaCheck( cudaEventCreate( &d_stop ));

    //	Black & Scholes price
    double dt = option.t/(double)cva.n;
    bs_price[0] = host_bsCall(option);
    for(i=1;i<cva.n+1;i++){
    	if((option.t -= dt)<0)
    		bs_price[i] = 0;
    	else
    		bs_price[i] = host_bsCall(option);
    }

    //	Ripristino valore originale del Time to mat
    option.t= 1.f;

    // GPU Monte Carlo
    printf("\nMonte Carlo execution on GPU:\nN^ simulations: %d\n",SIMS);
    CudaCheck( cudaEventRecord( d_start, 0 ));
    dev_cvaEquityOption(&cva, numBlocks, numThreads);
    CudaCheck( cudaEventRecord( d_stop, 0));
    CudaCheck( cudaEventSynchronize( d_stop ));
    CudaCheck( cudaEventElapsedTime( &GPU_timeSpent, d_start, d_stop ));
    GPU_timeSpent /= 1000;

    printf("\nTotal execution time: %f s\n\n", GPU_timeSpent);

    printf("\nPrezzi Simulati:\n");
   	printf("|\ti\t\t|\tPrezzi BS\t| Differenza Prezzi\t|\tPrezzi\t\t|\tDefault Prob\t|\n");
   	for(i=0;i<cva.n+1;i++){
   		difference = abs(cva.ee[i].Expected - bs_price[i]);
   		printf("|\t%f\t|\t%f\t|\t%f\t|\t%f\t|\t%f\t|\n",dt*i,bs_price[i],difference,cva.ee[i].Expected,cva.dp[i]);
   	}
   	printf("\nCVA: %f\nFVA: %f\nTotal: %f\n\n",cva.cva,cva.fva,(cva.cva+cva.fva));

   	free(cva.dp);
   	free(cva.fp);
   	free(cva.ee);
   	free(price);
   	free(bs_price);
    return 0;
}
