/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

//#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "MonteCarlo.h"

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#ifndef CudaCheck
#define CudaCheck(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
#endif

__device__ __constant__ double D_DRIFTVECT[N], D_CHOLMAT[N][N], D_S[N], D_V[N], D_W[N], D_K, D_T, D_R;

__device__ void prodConstMat(Matrix *second, Matrix *result){
    if(N != second->rows){
        printf("Non si può effettuare la moltiplicazione\n");
        return;
    }
    double somma;
    int i,j,k;
    result->rows = N;
    result->cols = second->cols;
    for(i=0;i<result->rows;i++){
        for(j=0;j<result->cols;j++){
            somma = 0;
            for(k=0;k<N;k++)
                //somma += first->data[i][k]*second->data[k][j];
                somma += D_CHOLMAT[i][k] * second->data[j+k*second->cols];
            //result->data[i][j] = somma;
            result->data[j+i*result->cols] = somma;
        }
    }
}

__device__ void devGaussVect(curandState *threadState, double *result, const int n){
    int i;
    // Random number vector
    double g[N];
    // RNGs
    for(i=0;i<n;i++)
        g[i]=curand_normal(threadState);
    Matrix gauss, r;
    gauss.rows = n;     r.rows=n;
    gauss.cols = 1;     r.cols=1;
    gauss.data = &g[0]; r.data=result;
    //A*G
    prodConstMat(&gauss,&r);
    //X=m+A*G
    for(i=0;i<n;i++){
        r.data[i] += D_DRIFTVECT[i];
    }
}

__device__ void devMultiStVal(double *values, double *g, double t, double r, int n){
    int i;
    for(i=0;i<n;i++){
        double mu = (r - 0.5 * D_V[i] * D_V[i])*t;
        double si = D_V[i] * g[i] * sqrt(t);
        values[i] = D_S[i] * exp(mu+si);
    }
}

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

    for( i=cacheIndex; i<SIM; i+=blockDim.x){
        st_sum = 0;
        //Simulation of stock prices
        devGaussVect(&threadState,bt,N);
        devMultiStVal(s, bt, D_T, D_R, N);
        for(j=0;j<N;j++)
            st_sum += s[j] * D_W[j];
        //Option payoff
        price = st_sum - D_K;
        if(price<0)
            price = 0.0f;
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

void GPUBasketOpt(MultiOptionData *option, OptionValue *callValue ){
    int i;
    /*----------------- HOST MEMORY -------------------*/
    OptionValue *h_CallValue;
    //Allocation pinned host memory for prices
    CudaCheck(cudaHostAlloc(&h_CallValue, sizeof(OptionValue)*(MAX_BLOCKS),cudaHostAllocDefault));

    /*--------------- CONSTANT MEMORY ----------------*/

    CudaCheck(cudaMemcpyToSymbol(D_DRIFTVECT,option->d,N*sizeof(double)));
    CudaCheck(cudaMemcpyToSymbol(D_CHOLMAT,option->p,N*N*sizeof(double)));
    CudaCheck(cudaMemcpyToSymbol(D_S,option->s,N*sizeof(double)));
    CudaCheck(cudaMemcpyToSymbol(D_V,option->v,N*sizeof(double)));
    CudaCheck(cudaMemcpyToSymbol(D_W,option->w,N*sizeof(double)));
    CudaCheck(cudaMemcpyToSymbol(D_K,&option->k,sizeof(double)));
    CudaCheck(cudaMemcpyToSymbol(D_T,&option->t,sizeof(double)));
    CudaCheck(cudaMemcpyToSymbol(D_R,&option->r,sizeof(double)));

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
    long int nSim = MAX_BLOCKS * SIM;
    for ( i = 0; i < MAX_BLOCKS; i++ ){
        sum += h_CallValue[i].Expected;
        sum2 += h_CallValue[i].Confidence;
    }
    price = exp(-(option->r*option->t)) * (sum/(double)nSim);
    empstd = sqrt((double)((double)nSim * sum2 - sum * sum)
                         /((double)nSim * (double)(nSim - 1)));
    callValue->Confidence = 1.96 * empstd / (double)sqrt((double)nSim);
    callValue->Expected = price;

    //Free memory space
    CudaCheck(cudaFree(RNG));
    CudaCheck(cudaFreeHost(h_CallValue));
    CudaCheck(cudaFree(d_CallValue));
}
