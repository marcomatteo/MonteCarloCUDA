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

__global__ void MultiMCBasketOptKernel(curandState * randseed, OptionValue *d_CallValue){
    int i,j;
    int cacheIndex = threadIdx.x;
    int blockIndex = blockIdx.x;

    /*------------------ SHARED MEMORY DICH ----------------*/
    __shared__ double s_Sum[MAX_THREADS];
    __shared__ double s_Sum2[MAX_THREADS];

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

    for( i=cacheIndex; i<PATH; i+=blockDim.x){
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
        for(j=0;j<N;j++){
                s[j] = OPTION.s[j] * exp((OPTION.r - 0.5 * OPTION.v[j] * OPTION.v[j])*OPTION.t+OPTION.v[j] * bt[j] * sqrt(OPTION.t));
        }

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
    s_Sum[cacheIndex] = sum.Expected;
    s_Sum2[cacheIndex] = sum.Confidence;
    __syncthreads();
    //Reduce shared memory accumulators and write final result to global memory
    int halfblock = blockDim.x/2;
    do{
        if ( cacheIndex < halfblock ){
            s_Sum[cacheIndex] += s_Sum[cacheIndex+halfblock];
            s_Sum2[cacheIndex] += s_Sum2[cacheIndex+halfblock];
            __syncthreads();
        }
        halfblock /= 2;
    }while ( halfblock != 0 );
    __syncthreads();
    //Keeping the first element for each block using one thread
    if (threadIdx.x == 0){
    	d_CallValue[blockIndex].Expected = s_Sum[0];
    	d_CallValue[blockIndex].Confidence = s_Sum2[0];
    }
}

__global__ void randomSetup( curandState *randSeed ){
    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Each threadblock gets different seed, threads within a threadblock get different sequence numbers
    curand_init(blockIdx.x + gridDim.x, threadIdx.x, 0, &randSeed[tid]);
}

extern "C" OptionValue dev_basketOpt(MultiOptionData *option){
    int i;
    OptionValue callValue;
    /*----------------- HOST MEMORY -------------------*/
    OptionValue *h_CallValue;
    //Allocation pinned host memory for prices
    CudaCheck(cudaHostAlloc(&h_CallValue, sizeof(OptionValue)*(MAX_BLOCKS),cudaHostAllocDefault));

    /*--------------- CONSTANT MEMORY ----------------*/
    CudaCheck(cudaMemcpyToSymbol(OPTION,option,sizeof(MultiOptionData)));
    /*----------------- DEVICE MEMORY -------------------*/
    OptionValue *d_CallValue;
    CudaCheck(cudaMalloc(&d_CallValue, sizeof(OptionValue)*(MAX_BLOCKS)));

    /*------------ RNGs and TIME VARIABLES --------------*/
    curandState *RNG;
    cudaEvent_t start, stop;
    CudaCheck( cudaEventCreate( &start ));
    CudaCheck( cudaEventCreate( &stop ));
    float time;

    // RANDOM NUMBER GENERATION KERNEL
    //Allocate states for pseudo random number generators
    CudaCheck(cudaMalloc((void **) &RNG, MAX_BLOCKS * MAX_THREADS * sizeof(curandState)));
    //Setup for the random number sequence
    randomSetup<<<MAX_BLOCKS, MAX_THREADS>>>(RNG);

    //MONTE CARLO KERNEL
    CudaCheck( cudaEventRecord( start, 0 ));
    MultiMCBasketOptKernel<<<MAX_BLOCKS, MAX_THREADS>>>(RNG,(OptionValue *)(d_CallValue));
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "\nMonte Carlo simulations done in %f milliseconds\n", time);
    CudaCheck( cudaEventDestroy( start ));
    CudaCheck( cudaEventDestroy( stop ));

    //MEMORY CPY: prices per block
    CudaCheck(cudaMemcpy(h_CallValue, d_CallValue, MAX_BLOCKS * sizeof(OptionValue), cudaMemcpyDeviceToHost));

    // Closing Monte Carlo
    long double sum=0, sum2=0, price, empstd;
    long int nSim = MAX_BLOCKS * PATH;
    for ( i = 0; i < MAX_BLOCKS; i++ ){
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
