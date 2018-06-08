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
extern "C" void host_cvaEquityOption(CVA *cva, int numBlocks, int numThreads);
extern "C" void dev_cvaEquityOption(CVA *cva, int numBlocks, int numThreads);
extern "C" void printOption( OptionData o);
extern "C" void Parameters(int *numBlocks, int *numThreads);

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
	int numBlocks, numThreads, i, SIMS;
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
    double difference, dt,
    *price = (double*)malloc(sizeof(double)*(cva.n+1)),
    *bs_price = (double*)malloc(sizeof(double)*(cva.n+1));
    cudaEvent_t d_start, d_stop;

    printf("Expected Exposures of an Equity Option\n");
	//	Definizione dei parametri CUDA per l'esecuzione in parallelo
	Parameters(&numBlocks, &numThreads);
	printf("Simulazione di ( %d ; %d )\n",numBlocks, numThreads);
	SIMS = numBlocks*PATH;

	//	Print Option details
	printOption(option);

	// Timer init
    CudaCheck( cudaEventCreate( &d_start ));
    CudaCheck( cudaEventCreate( &d_stop ));

    //	Black & Scholes price
    dt = option.t/(double)cva.n;
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
    printf("\nCVA execution on GPU:\nN^ simulations per time interval: %d * %d\n",SIMS,cva.n);
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
