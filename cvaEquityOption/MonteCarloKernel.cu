/*
 * MonteCarloKernel.cu
 *
 *  Created on: 06/feb/2018
 *  Author: marco
 */

//#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "MonteCarlo.h"

/*
 * Error handling from Cuda programming - shane cook
 */
void cuda_error_check(const char * prefix, const char * postfix){
	if (cudaPeekAtLastError() != cudaSuccess){
		printf("\n%s%s%s", prefix, cudaGetErrorString(cudaGetLastError()), postfix);
		cudaDeviceReset();
		//wait_exit();
		exit(1);
	}
}

typedef struct{
	OptionValue *h_CallValue, *d_CallValue;
	OptionValue callValue;
    curandState *RNG;
    int numBlocks, numThreads;
    MultiOptionData option;
} MonteCarloData;

// Inizializzazione per Monte Carlo da fare una volta sola
void MonteCarlo_init(OptionValue *h_CallValue, OptionValue *d_CallValue, curandState *RNG, int numBlocks, int numThreads);
// Liberazione della memoria da fare una volta sola
void MonteCarlo_free(OptionValue *h_CallValue, OptionValue *d_CallValue, curandState *RNG);
// Metodo Monte Carlo che si può richiamare quante volte si vuole
void MonteCarlo(MultiOptionData option, OptionValue *h_CallValue, OptionValue *d_CallValue, curandState *RNG, int numBlocks, int numThreads);


__device__ __constant__ MultiOptionData OPTION;
__device__ __constant__ int N_OPTION;

__device__ void brownianVect(double *bt, curandState threadState){
	int i,j;
	double g[N];
	for(i=0;i<N_OPTION;i++)
		g[i]=curand_normal(&threadState);
	for(i=0;i<N_OPTION;i++){
		double somma = 0;
		for(j=0;j<N_OPTION;j++)
	 		//somma += first->data[i][k]*second->data[k][j];
			somma += OPTION.p[i][j] * g[j];
	     	//result->data[i][j] = somma;
		bt[i] = somma;
	}
	for(i=0;i<N_OPTION;i++)
		bt[i] += OPTION.d[i];
}

__device__ double blackScholes(double *bt){
	int j;
	double s[N], st_sum=0, price;
	for(j=0;j<N_OPTION;j++)
	     s[j] = OPTION.s[j] * exp((OPTION.r - 0.5 * OPTION.v[j] * OPTION.v[j])*OPTION.t+OPTION.v[j] * bt[j] * sqrt(OPTION.t));
	// Third step: Mean price
	for(j=0;j<N_OPTION;j++)
		st_sum += s[j] * OPTION.w[j];
	// Fourth step: Option payoff
	price = st_sum - OPTION.k;
	if(price<0)
		price = 0.0f;
	return price;
}


__global__ void MultiMCBasketOptKernel(curandState * randseed, OptionValue *d_CallValue){
    int i;
    // Parameters for shared memory
    int sumIndex = threadIdx.x;
    int sum2Index = sumIndex + blockDim.x;
    // Parameter for reduction
    int blockIndex = blockIdx.x;

    /*------------------ SHARED MEMORY DICH ----------------*/
    extern __shared__ double s_Sum[];

    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Copy random number state to local memory
    curandState threadState = randseed[tid];

    OptionValue sum = {0, 0};

    for( i=sumIndex; i<PATH; i+=blockDim.x){
    	//vectors of brownian and ST
    	double price=0.0f, g[N], s[N], bt[N], st_sum=0.0f;
    	int j,k;

    	brownianVect(bt,threadState);
        price=blackScholes(bt);


        sum.Expected += price;
        sum.Confidence += price*price;
    }
    //Copy to the shared memory
    s_Sum[sumIndex] = sum.Expected;
    s_Sum[sum2Index] = sum.Confidence;
    __syncthreads();
    //Reduce shared memory accumulators and write final result to global memory
    int halfblock = blockDim.x/2;
    do{
        if ( sumIndex < halfblock ){
            s_Sum[sumIndex] += s_Sum[sumIndex+halfblock];
            s_Sum[sum2Index] += s_Sum[sum2Index+halfblock];
            __syncthreads();
        }
        halfblock /= 2;
    }while ( halfblock != 0 );
    __syncthreads();
    //Keeping the first element for each block using one thread
    if (sumIndex == 0){
    	d_CallValue[blockIndex].Expected = s_Sum[sumIndex];
    	d_CallValue[blockIndex].Confidence = s_Sum[sum2Index];
    }
}

__global__ void randomSetup( curandState *randSeed ){
    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Each threadblock gets different seed, threads within a threadblock get different sequence numbers
    curand_init(blockIdx.x + gridDim.x, threadIdx.x, 0, &randSeed[tid]);
}

void MonteCarlo_init(MonteCarloData *data){
	cudaEvent_t start, stop;
	CudaCheck( cudaEventCreate( &start ));
    CudaCheck( cudaEventCreate( &stop ));
    float time;

    int n_option = N;

    /*--------------- CONSTANT MEMORY ----------------*/
    CudaCheck(cudaMemcpyToSymbol(N_OPTION,&n_option,sizeof(int)));

	// RANDOM NUMBER GENERATION KERNEL
	//Allocate states for pseudo random number generators
	CudaCheck(cudaMalloc((void **) &data->RNG, data->numBlocks * data->numThreads * sizeof(curandState)));
	//Setup for the random number sequence
    CudaCheck( cudaEventRecord( start, 0 ));
    randomSetup<<<data->numBlocks, data->numThreads>>>(data->RNG);
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "RNG done in %f milliseconds\n", time);

    //	Host Memory Allocation
    CudaCheck(cudaMallocHost(&data->h_CallValue, sizeof(OptionValue)*(data->numBlocks)));
    //	Device Memory Allocation
    CudaCheck(cudaMalloc(&data->d_CallValue, sizeof(OptionValue)*(data->numBlocks)));

    CudaCheck( cudaEventDestroy( start ));
    CudaCheck( cudaEventDestroy( stop ));
}

void MonteCarlo_free(MonteCarloData *data){
	//Free memory space
	CudaCheck(cudaFree(data->RNG));
    CudaCheck(cudaFreeHost(data->h_CallValue));
    CudaCheck(cudaFree(data->d_CallValue));
}

void MonteCarlo(MonteCarloData *data){
	/*--------------- CONSTANT MEMORY ----------------*/
	CudaCheck(cudaMemcpyToSymbol(OPTION,&data->option,sizeof(MultiOptionData)));

	/*----------------- SHARED MEMORY -------------------*/
	int i, numShared = sizeof(double) * data->numThreads * 2;

	MultiMCBasketOptKernel<<<data->numBlocks, data->numThreads, numShared>>>(data->RNG,(OptionValue *)(data->d_CallValue));
	cuda_error_check("\Errore nel lancio MultiMCBasketOptKernel: ","\n");

	//MEMORY CPY: prices per block
	CudaCheck(cudaMemcpy(data->h_CallValue, data->d_CallValue, data->numBlocks * sizeof(OptionValue), cudaMemcpyDeviceToHost));

	// Closing Monte Carlo
	long double sum=0, sum2=0, price, empstd;
    long int nSim = data->numBlocks * PATH;
    for ( i = 0; i < data->numBlocks; i++ ){
    	sum += data->h_CallValue[i].Expected;
	    sum2 += data->h_CallValue[i].Confidence;
	}
	price = exp(-(data->option.r*data->option.t)) * (sum/(double)nSim);
    empstd = sqrt((double)((double)nSim * sum2 - sum * sum)/((double)nSim * (double)(nSim - 1)));
    data->callValue.Confidence = 1.96 * empstd / (double)sqrt((double)nSim);
    data->callValue.Expected = price;
}

extern "C" OptionValue dev_basketOpt(MultiOptionData *option, int numBlocks, int numThreads){
		int i;
	    OptionValue callValue;
	    /*----------------- HOST MEMORY -------------------*/
	    OptionValue *h_CallValue;
	    //Allocation pinned host memory for prices
	    CudaCheck(cudaHostAlloc(&h_CallValue, sizeof(OptionValue)*(numBlocks),cudaHostAllocDefault));

	    /*--------------- CONSTANT MEMORY ----------------*/
	    CudaCheck(cudaMemcpyToSymbol(OPTION,&option,sizeof(MultiOptionData)));

	    /*----------------- DEVICE MEMORY -------------------*/
	    OptionValue *d_CallValue;
	    CudaCheck(cudaMalloc(&d_CallValue, sizeof(OptionValue)*(numBlocks)));

	    /*----------------- SHARED MEMORY -------------------*/
	    int numShared = sizeof(double) * numThreads * 2;

	    /*------------ RNGs and TIME VARIABLES --------------*/
	    curandState *RNG;
	    cudaEvent_t start, stop;
	    CudaCheck( cudaEventCreate( &start ));
	    CudaCheck( cudaEventCreate( &stop ));
	    float time;

	    // RANDOM NUMBER GENERATION KERNEL
	    //Allocate states for pseudo random number generators
	    CudaCheck(cudaMalloc((void **) &RNG, numBlocks * numThreads * sizeof(curandState)));
	    //Setup for the random number sequence
	    CudaCheck( cudaEventRecord( start, 0 ));
	    randomSetup<<<numBlocks, numThreads>>>(RNG);
	    CudaCheck( cudaEventRecord( stop, 0));
	    CudaCheck( cudaEventSynchronize( stop ));
	    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
	    printf( "RNG done in %f milliseconds\n", time);


	    //MONTE CARLO KERNEL
	    CudaCheck( cudaEventRecord( start, 0 ));
	    MultiMCBasketOptKernel<<<numBlocks, numThreads, numShared>>>(RNG,(OptionValue *)(d_CallValue));
	    CudaCheck( cudaEventRecord( stop, 0));
	    CudaCheck( cudaEventSynchronize( stop ));
	    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
	    printf( "Monte Carlo simulations done in %f milliseconds\n", time);
	    CudaCheck( cudaEventDestroy( start ));
	    CudaCheck( cudaEventDestroy( stop ));

	    //MEMORY CPY: prices per block
	    CudaCheck(cudaMemcpy(h_CallValue, d_CallValue, numBlocks * sizeof(OptionValue), cudaMemcpyDeviceToHost));

	    // Closing Monte Carlo
	    long double sum=0, sum2=0, price, empstd;
	    long int nSim = numBlocks * PATH;
	    for ( i = 0; i < numBlocks; i++ ){
	        sum += h_CallValue[i].Expected;
	        sum2 += h_CallValue[i].Confidence;
	    }
	    price = exp(-(option->r*option->t)) * (sum/(double)nSim);
	    empstd = sqrt((double)((double)nSim * sum2 - sum * sum)
	                         /((double)nSim * (double)(nSim - 1)));
	    callValue.Confidence = 1.96 * empstd / (double)sqrt((double)nSim);
	    callValue.Expected = price;

	    //Free memory space
	    CudaCheck(cudaFree(RNG));
	    CudaCheck(cudaFreeHost(h_CallValue));
	    CudaCheck(cudaFree(d_CallValue));

	    return callValue;
}

extern "C" OptionValue dev_vanillaOpt(OptionData *opt, int numBlocks, int numThreads){

	MultiOptionData option;
	option.w[0] = 1;
	option.d[0] = 0;
	option.p[0][0] = 1;
	option.s[0] = opt->s;
	option.v[0] = opt->v;
	option.k = opt->k;
	option.r = opt->r;
	option.t = opt->t;

    MonteCarloData data;
    data.option = option;
    data.numBlocks = numBlocks;
    data.numThreads = numThreads;

    MonteCarlo_init(&data);
    MonteCarlo(&data);
    MonteCarlo_free(&data);

    return data.callValue;
}

extern "C" void dev_cvaEquityOption(OptionValue *callValue, OptionData opt, CreditData credit, int n, int numBlocks, int numThreads){
    int i;
    double dt = opt.t / (double)n;

    /*--------------- CONSTANT MEMORY ----------------*/
    MultiOptionData option;
    option.w[0] = 1;
    option.d[0] = 0;
    option.p[0][0] = 1;
    option.s[0] = opt.s;
    option.v[0] = opt.v;
    option.k = opt.k;
    option.r = opt.r;
    option.t = opt.t;


/*	//-------------	STREAMS -----------------
    cudaStream_t stream0, stream1;
    CudaCheck(cudaStreamCreate(&stream0));
    CudaCheck(cudaStreamCreate(&stream1));
*/
    MonteCarloData data;
    data.option = option;
    data.numBlocks = numBlocks;
    data.numThreads = numThreads;

    MonteCarlo_init(&data);
    MonteCarlo(&data);
    callValue[0] = data.callValue;

	for( i=1; i<(n+1); i++){
		if((data.option.t -= dt)<0)
			callValue[i] = 0;
		else{
			MonteCarlo(&data);
			callValue[i] = data.callValue;
		}
	}

	MonteCarlo_free(&data);

}
