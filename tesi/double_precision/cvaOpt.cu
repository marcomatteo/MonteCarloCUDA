//
//  MonteCarlo.cu
//  tesi
//
//  Created by Marco Matteo Buzzulini on 27/11/17.
//  Copyright © 2017 Marco Matteo Buzzulini. All rights reserved.
//

#include "MonteCarlo.h"

extern "C" double host_bsCall ( OptionData );
extern "C" void host_cvaEquityOption(CVA *cva, int numBlocks, int numThreads);
extern "C" void dev_cvaEquityOption(CVA *cva, int numBlocks, int numThreads);
extern "C" void printOption( OptionData o);

void Parameters(int *numBlocks, int *numThreads);
void memAdjust(cudaDeviceProp *deviceProp, int *numThreads);
void sizeAdjust(cudaDeviceProp *deviceProp, int *numBlocks, int *numThreads);

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
    float GPU_timeSpent=0, CPU_timeSpent=0;
    double difference, dt,
    *bs_price = (double*)malloc(sizeof(double)*(cva.n+1));
    cudaEvent_t d_start, d_stop;

    printf("Expected Exposures of an Equity Option\n");
	//	Definizione dei parametri CUDA per l'esecuzione in parallelo
    Parameters(&numBlocks, &numThreads);
    printf("Inserisci il numero di simulazioni (x100.000): ");
    scanf("%d",&SIMS);
    SIMS *= 100000;
    printf("\nScenari di Monte Carlo: %d\n",SIMS);

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

    // CPU Monte Carlo
    printf("\nCVA execution on CPU:\n");
    CudaCheck( cudaEventRecord( d_start, 0 ));
    host_cvaEquityOption(&cva, numBlocks, numThreads,SIMS);
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
    printf("\nCVA: %f\nFVA: %f\nTotal: %f\n\n",cva.cva,cva.fva,(cva.cva+cva.fva));
    printf("\nTotal execution time: %f s\n\n", CPU_timeSpent);

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
   	free(bs_price);
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

void Parameters(int *numBlocks, int *numThreads){
    cudaDeviceProp deviceProp;
    int i = 0;
    CudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    *numThreads = NTHREADS;
    *numBlocks = BLOCKS;
    for (i=0; i<THREADS; i++) {
        sizeAdjust(&deviceProp,numBlocks, &numThreads[i]);
        memAdjust(&deviceProp, &numThreads[i]);
    }
}
