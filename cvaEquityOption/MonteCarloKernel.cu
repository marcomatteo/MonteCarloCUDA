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

__device__ __constant__ MultiOptionData OPTION;

__global__ void MultiMCBasketOptKernel(curandState * randseed, OptionValue *d_CallValue, double z){
    int i,j;
    // Parameters for shared memory
    int sumIndex = threadIdx.x;
    int sum2Index = sumIndex + blockDim.x;
    // Parameter for reduction
    int blockIndex = blockIdx.x;

    //	Parametro opzionale z, che se >1 riduce il valore del Time to maturity dell'opzione
    double t = OPTION.t - z;

    /*------------------ SHARED MEMORY DICH ----------------*/
    extern __shared__ double s_Sum[];

    //Monte Carlo variables
    double st_sum=0.0f, price;

    //vectors of brownian and ST
    double bt[N];
    double s[N];

    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Copy random number state to local memory
    curandState threadState = randseed[tid];

    OptionValue sum = {0, 0};

    for( i=sumIndex; i<PATH; i+=blockDim.x){
        st_sum = 0;
        // First step: Brownian motion
        double g[N];
        // RNGs
        for(j=0;j<N;j++)
        	g[j]=curand_normal(&threadState);
        //A*G
        double somma;
        int j,k;
        for(j=0;j<N;j++){
        	somma = 0;
         	for(k=0;k<N;k++)
         		//somma += first->data[i][k]*second->data[k][j];
                somma += OPTION.p[j][k] * g[k];
         	//result->data[i][j] = somma;
            bt[j] = somma;
        }
        //X=m+A*G
        for(j=0;j<N;j++)
            bt[j] += OPTION.d[j];

        //	Second step: Price simulation
        for(j=0;j<N;j++)
                s[j] = OPTION.s[j] * exp((OPTION.r - 0.5 * OPTION.v[j] * OPTION.v[j])*t+OPTION.v[j] * bt[j] * sqrt(t));


        // Third step: Mean price
        for(j=0;j<N;j++)
            st_sum += s[j] * OPTION.w[j];

        //	Fourth step: Option payoff
        price = st_sum - OPTION.k;
        if(price<0)
            price = 0.0f;

        //	Fifth step:	Monte Carlo price sum
        sum.Expected += price;
        sum.Confidence += price*price;
    }
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

extern "C" OptionValue dev_basketOpt(MultiOptionData *option, int numBlocks, int numThreads){
    int i;
    OptionValue callValue;
    /*----------------- HOST MEMORY -------------------*/
    OptionValue *h_CallValue;
    //Allocation pinned host memory for prices
    CudaCheck(cudaHostAlloc(&h_CallValue, sizeof(OptionValue)*(numBlocks),cudaHostAllocDefault));

    /*--------------- CONSTANT MEMORY ----------------*/
    CudaCheck(cudaMemcpyToSymbol(OPTION,option,sizeof(MultiOptionData)));

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
    MultiMCBasketOptKernel<<<numBlocks, numThreads, numShared>>>(RNG,(OptionValue *)(d_CallValue),0);
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
    int i;
    OptionValue callValue;
    /*----------------- HOST MEMORY -------------------*/
    OptionValue *h_CallValue;
    //Allocation pinned host memory for prices
    CudaCheck(cudaHostAlloc(&h_CallValue, sizeof(OptionValue)*(numBlocks),cudaHostAllocDefault));

    /*--------------- CONSTANT MEMORY ----------------*/
    MultiOptionData option;
    option.w[0] = 1;
    option.d[0] = 0;
    option.p[0][0] = 1;
    option.s[0] = opt->s;
    option.v[0] = opt->v;
    option.k = opt->k;
    option.r = opt->r;
    option.t = opt->t;

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
    MultiMCBasketOptKernel<<<numBlocks, numThreads, numShared>>>(RNG,(OptionValue *)(d_CallValue),0);
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
    price = exp(-(option.r*option.t)) * (sum/(double)nSim);
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

extern "C" void dev_cvaEquityOption(OptionValue *callValue, OptionData opt, CreditData credit, int n, int numBlocks, int numThreads){
    int i;
    double dt = opt.t / (double)n;
    /*----------------- HOST MEMORY -------------------*/
    OptionValue *h_CallValue;
    //Allocation pinned host memory for prices
    CudaCheck(cudaHostAlloc(&h_CallValue, sizeof(OptionValue)*(numBlocks),cudaHostAllocDefault));

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
    /*
    CudaCheck( cudaEventRecord( start, 0 ));

    // Qui ci andrebbe Monte Carlo

    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Monte Carlo simulations done in %f milliseconds\n", time);
    CudaCheck( cudaEventDestroy( start ));
    CudaCheck( cudaEventDestroy( stop ));
	*/

	for( i=0; i<n+1; i++){
    	MultiMCBasketOptKernel<<<numBlocks, numThreads, numShared>>>(RNG,(OptionValue *)(d_CallValue),((double)i*dt));
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
        callValue[i].Confidence = 1.96 * empstd / (double)sqrt((double)nSim);
    	callValue[i].Expected = price;
    	printf("\nPrezzo simulato[%d]: %f\n",i, price);
	}

    //Free memory space
    CudaCheck(cudaFree(RNG));
    CudaCheck(cudaFreeHost(h_CallValue));
    CudaCheck(cudaFree(d_CallValue));
}
