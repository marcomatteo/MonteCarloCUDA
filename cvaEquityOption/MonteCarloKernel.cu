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
	double *g=(double*)malloc(N_OPTION*sizeof(double));
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
	free(g);
}

__device__ void blackScholes(double *price, double *bt){
	int i;
	double *s=(double*)malloc(N_OPTION*sizeof(double)), mean;
	for(i=0;i<N_OPTION;i++)
        s[i] = OPTION.s[i] * exp((OPTION.r - 0.5 * OPTION.v[i] * OPTION.v[i])*OPTION.t+OPTION.v[i] * bt[i] * sqrt(OPTION.t));
	for(i=0;i<N_OPTION;i++)
		mean += s[i] * OPTION.w[i];
	*price = mean - OPTION.k;
	if(*price<0)
		*price = 0.0f;
	free(s);
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
    	double *bt=(double*)malloc(N_OPTION*sizeof(double)), price=0.0f;

        /* RNGs
        for(j=0;j<N_OPTION;j++)
        	g[j]=curand_normal(&threadState);
        */
        /* A*G
        double somma;
        int j,k;
        for(j=0;j<N_OPTION;j++){
        	somma = 0;
         	for(k=0;k<N_OPTION;k++)
         		//somma += first->data[i][k]*second->data[k][j];
                somma += OPTION.p[j][k] * g[k];
         	//result->data[i][j] = somma;
            bt[j] = somma;
        }
        X=m+A*G
        for(j=0;j<N_OPTION;j++)
            bt[j] += OPTION.d[j];
        */
    	brownianVect(bt,threadState);

        /*
         * Second step: Price simulation
        for(j=0;j<N_OPTION;j++)
                s[j] = OPTION.s[j] * exp((OPTION.r - 0.5 * OPTION.v[j] * OPTION.v[j])*OPTION.t+OPTION.v[j] * bt[j] * sqrt(OPTION.t));
         * Third step: Mean price
        for(j=0;j<N_OPTION;j++)
            st_sum += s[j] * OPTION.w[j];
         * Fourth step: Option payoff
        price = st_sum - OPTION.k;
        if(price<0)
            price = 0.0f;
        */
        blackScholes(&price,bt);

        //	Fifth step:	Monte Carlo price sum
        sum.Expected += price;
        sum.Confidence += price*price;
        free(bt);
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
/*
void MonteCarlo_init(OptionValue **h_CallValue, OptionValue **d_CallValue, curandState *RNG, int numBlocks, int numThreads){
	cudaEvent_t start, stop;
	CudaCheck( cudaEventCreate( &start ));
    CudaCheck( cudaEventCreate( &stop ));
    float time;

    int n_option = N;

    //--------------- CONSTANT MEMORY ----------------
    CudaCheck(cudaMemcpyToSymbol(N_OPTION,&n_option,sizeof(int)));

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

    //	Host Memory Allocation
    CudaCheck(cudaMallocHost(h_CallValue, sizeof(OptionValue)*(numBlocks)));
    //	Device Memory Allocation
    CudaCheck(cudaMalloc(d_CallValue, sizeof(OptionValue)*(numBlocks)));

    CudaCheck( cudaEventDestroy( start ));
    CudaCheck( cudaEventDestroy( stop ));
}

void MonteCarlo_free(OptionValue *h_CallValue, OptionValue *d_CallValue, curandState *RNG){
	//Free memory space
	CudaCheck(cudaFree(RNG));
    CudaCheck(cudaFreeHost(h_CallValue));
    CudaCheck(cudaFree(d_CallValue));
}

OptionValue MonteCarlo(MultiOptionData option, OptionValue *h_CallValue, OptionValue *d_CallValue, curandState *RNG, int numBlocks, int numThreads){
	OptionValue callValue;
	//--------------- CONSTANT MEMORY ----------------
	CudaCheck(cudaMemcpyToSymbol(OPTION,&option,sizeof(MultiOptionData)));

	//----------------- SHARED MEMORY -------------------
	int i, numShared = sizeof(double) * numThreads * 2;

	MultiMCBasketOptKernel<<<numBlocks, numThreads, numShared>>>(RNG,(OptionValue *)(d_CallValue));
	cuda_error_check("\nLancio Kernel Monte Carlo "," fallito \n");

	//MEMORY CPY: prices per block
	CudaCheck(cudaMemcpy(h_CallValue, d_CallValue, numBlocks * sizeof(OptionValue), cudaMemcpyDeviceToHost));

	// Closing Monte Carlo
	long double sum=0, sum2=0, price, empstd;
    long int nSim = numBlocks * PATH;
    for ( i = 0; i < numBlocks; i++ ){
    	sum += h_CallValue[i].Expected;
	    sum2 += h_CallValue[i].Confidence;
	}
	price = exp(-(option.r*option.t)) * (sum/(double)nSim);
    empstd = sqrt((double)((double)nSim * sum2 - sum * sum)/((double)nSim * (double)(nSim - 1)));
    callValue.Confidence = 1.96 * empstd / (double)sqrt((double)nSim);
    callValue.Expected = price;

    return callValue;
}
*/
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
    /*----------------- HOST MEMORY -------------------*/
    OptionValue *h_CallValue0, *h_CallValue1;
    //Allocation pinned host memory for prices
    CudaCheck(cudaHostAlloc(&h_CallValue0, sizeof(OptionValue)*(numBlocks),cudaHostAllocDefault));
    CudaCheck(cudaHostAlloc(&h_CallValue1, sizeof(OptionValue)*(numBlocks),cudaHostAllocDefault));

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
    CudaCheck(cudaMemcpyToSymbol(OPTION,&option,sizeof(MultiOptionData)));

    /*-------------	STREAMS -----------------*/
    cudaStream_t stream0, stream1;
    CudaCheck(cudaStreamCreate(&stream0));
    CudaCheck(cudaStreamCreate(&stream1));

    /*----------------- DEVICE MEMORY -------------------*/
    OptionValue *d_CallValue0,*d_CallValue1;
    CudaCheck(cudaMalloc(&d_CallValue0, sizeof(OptionValue)*(numBlocks)));
    CudaCheck(cudaMalloc(&d_CallValue1, sizeof(OptionValue)*(numBlocks)));

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
    /*
    CudaCheck( cudaEventRecord( start, 0 ));

    	MultiMCBasketOptKernel<<<numBlocks, numThreads, numShared, stream0>>>(RNG,(OptionValue *)(d_CallValue0),((double)i*dt));

    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Monte Carlo simulations done in %f milliseconds\n", time);
    CudaCheck( cudaEventDestroy( start ));
    CudaCheck( cudaEventDestroy( stop ));
	*/

	for( i=0; i<(n+1); i+=2){
    	MultiMCBasketOptKernel<<<numBlocks, numThreads, numShared, stream0>>>(RNG,(OptionValue *)(d_CallValue0));
    	cuda_error_check("Primo kernel","");
    	MultiMCBasketOptKernel<<<numBlocks, numThreads, numShared, stream1>>>(RNG,(OptionValue *)(d_CallValue1));
    	cuda_error_check("Secondo kernel","");
    	//MEMORY CPY: prices per block
    	CudaCheck(cudaMemcpyAsync(h_CallValue0, d_CallValue0, numBlocks * sizeof(OptionValue), cudaMemcpyDeviceToHost,stream0));
    	CudaCheck(cudaMemcpyAsync(h_CallValue1, d_CallValue1, numBlocks * sizeof(OptionValue), cudaMemcpyDeviceToHost,stream1));
    	// Closing Monte Carlo
    	long double sum1=0, sum2=0, sum3=0, sum4=0, price, empstd;
        int nSim = numBlocks * PATH;
   	    for ( i = 0; i < numBlocks; i++ ){
   	        sum1 += h_CallValue0[i].Expected;
   	        sum2 += h_CallValue0[i].Confidence;
   	        sum3 += h_CallValue1[i].Expected;
   	       	sum4 += h_CallValue1[i].Confidence;
   	    }
   	    price = exp(-(option.r*option.t)) * (sum1/(double)nSim);
        empstd = sqrt((double)((double)nSim * sum2 - sum1 * sum1)/((double)nSim * (double)(nSim - 1)));
        callValue[i].Confidence = 1.96 * empstd / (double)sqrt((double)nSim);
    	callValue[i].Expected = price;
    	price = exp(-(option.r*option.t)) * (sum3/(double)nSim);
    	empstd = sqrt((double)((double)nSim * sum4 - sum3 * sum3)/((double)nSim * (double)(nSim - 1)));
        callValue[i+1].Confidence = 1.96 * empstd / (double)sqrt((double)nSim);
        callValue[i+1].Expected = price;
        CudaCheck(cudaStreamSynchronize(stream0));
        CudaCheck(cudaStreamSynchronize(stream1));
	}

    //Free memory space
    CudaCheck(cudaFree(RNG));
    CudaCheck(cudaFreeHost(h_CallValue0));
    CudaCheck(cudaFree(d_CallValue0));
    CudaCheck(cudaFreeHost(h_CallValue1));
    CudaCheck(cudaFree(d_CallValue1));
    CudaCheck(cudaStreamDestroy(stream0));
    CudaCheck(cudaStreamDestroy(stream1));
}
