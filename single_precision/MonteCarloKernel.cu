/*
 *  MonteCarloKernel.cu
 *  Monte Carlo methods in CUDA
 *  Dissertation project
 *  Created on: 06/feb/2018
 *  Author: Marco Matteo Buzzulini
 */

#include <curand.h>
#include <curand_kernel.h>
#include "MonteCarlo.h"
#define max(a,b) \
({ __typeof__ (a) _a = (a); \
__typeof__ (b) _b = (b); \
_a > _b ? _a : _b; })

// Struct for Monte Carlo methods
typedef struct{
	OptionValue *h_CallValue, *d_CallValue;
	OptionValue callValue;
    OptionData sopt;
	MultiOptionData mopt;
    curandState *RNG;
    int numBlocks, numThreads, numOpt, path;
} dev_MonteCarloData;

// Memory initialization for MC
void MonteCarlo_init(dev_MonteCarloData *data);
// Freeing memory after MC
void MonteCarlo_closing(dev_MonteCarloData *data);
// Monte Carlo method for Option Pricing
void MonteCarlo(dev_MonteCarloData *data);
// Monte Carlo method for CVA - 1 black-scholes option
void cvaMonteCarlo(dev_MonteCarloData *data, float intdef, float lgd, int n_grid);

////////////////////////////////////////////////////////////////
////////////////    CONSTANT MEMORY     ////////////////////////
////////////////////////////////////////////////////////////////

// Basket Option
__device__ __constant__ MultiOptionData MOPTION;
// Vanilla Call Option
__device__ __constant__ OptionData OPTION;
// Number of underlyings, num simulations per block and the sims for CVA
__device__ __constant__ int N_OPTION, N_PATH, N_GRID;
// Financial parameters for CVA: Default intensity and Loss given default
__device__ __constant__ float INTDEF, LGD;

////////////////////////////////////////////////////////////////
////////////////    KERNEL FUNCTIONS    ////////////////////////
////////////////////////////////////////////////////////////////

/*  *   *   *   *   ONLY DEVICE   *   *   *   *   */
// Call Option payoff
__device__ float callPayoff(curandState *threadState){
    float z = curand_normal(threadState);
    float s = OPTION.s * expf((OPTION.r - 0.5 * OPTION.v * OPTION.v) * OPTION.t + OPTION.v * sqrtf(OPTION.t) * z);
    return max(s - OPTION.k,0);
}

// Basket option random number
__device__ void brownianVect(float *bt, curandState *threadState){
	int i,j;
	float g[N];
	for(i=0;i<N_OPTION;i++)
		g[i]=curand_normal(threadState);
	for(i=0;i<N_OPTION;i++){
		float somma = 0;
		for(j=0;j<N_OPTION;j++)
			somma += MOPTION.p[i][j] * g[j];
		bt[i] = somma;
	}
	for(i=0;i<N_OPTION;i++)
		bt[i] += MOPTION.d[i];
}
// Basket option payoff
__device__ float basketPayoff(float *bt){
	int j;
	float s[N], st_sum=0, price;
    for(j=0;j<N_OPTION;j++)
        s[j] = MOPTION.s[j] * expf((MOPTION.r - 0.5 * MOPTION.v[j] * MOPTION.v[j])*MOPTION.t+MOPTION.v[j] * bt[j] * sqrtf(MOPTION.t));
	// Third step: Mean price
	for(j=0;j<N_OPTION;j++)
		st_sum += s[j] * MOPTION.w[j];
	// Fourth step: Option payoff
	price = st_sum - MOPTION.k;

    return max(price,0);
}

// Simulating Geometric Brownian path
__device__ float geomBrownian( float s, float t, float z ){
    float x = (OPTION.r - 0.5 * OPTION.v * OPTION.v) * t + OPTION.v * sqrtf(t) * z;
    return s * expf(x);
}

// Hastings approximation of cumulative normal distribution
__device__ float cnd(float d){
    const float       A1 = 0.31938153;
    const float       A2 = -0.356563782;
    const float       A3 = 1.781477937;
    const float       A4 = -1.821255978;
    const float       A5 = 1.330274429;
    const float ONEOVER2PI = 0.39894228040143267793994605993438;
    float K = 1.0 / (1.0 + 0.2316419 * fabs(d));
    float cnd = ONEOVER2PI * expf(- 0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    if (d > 0)
        return 1.0 - cnd;
    else
        return cnd;
}
// Black & Scholes price formula for vanilla options
__device__ float device_bsCall ( float s, float t){
    float d1 = ( logf(s / OPTION.k) + (OPTION.r + 0.5 * OPTION.v * OPTION.v) * t) / (OPTION.v * sqrtf(t));
    float d2 = d1 - OPTION.v * sqrtf(t);
    return s * cnd(d1) - OPTION.k * expf(- OPTION.r * t) * cnd(d2);
}

/*  *   *   *   *   GLOBAL  *   *   *   *   */
// Basket Option Kernel
__global__ void basketOptMonteCarlo(curandState * randseed, OptionValue *d_CallValue){
    // Parameters for shared memory
    int sumIndex = threadIdx.x;
    int sum2Index = sumIndex + blockDim.x;
    /*  - SHARED MEMORY -  */
    extern __shared__ float s_Sum[];

    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Copy random number state to local memory
    curandState threadState = randseed[tid];

    OptionValue sum = {0, 0};
    int i;
    for( i=sumIndex; i<N_PATH; i+=blockDim.x){
    	float price=0.0f, bt[N];
    	// Random Number Generation
   		brownianVect(bt,&threadState);
   		// Price simulation with the basket call option payoff function
        price=basketPayoff(bt);
        // Mean sum
        sum.Expected += price;
        sum.Confidence += price*price;
    }
    // Copy to the shared memory
    s_Sum[sumIndex] = sum.Expected;
    s_Sum[sum2Index] = sum.Confidence;
    __syncthreads();
    // Reduce shared memory accumulators and write final result to global memory
    int halfblock = blockDim.x/2;
    // Reduction in log2(threadBlocks) steps, so threadBlock must be power of 2
    do{
        if ( sumIndex < halfblock ){
            s_Sum[sumIndex] += s_Sum[sumIndex+halfblock];
            s_Sum[sum2Index] += s_Sum[sum2Index+halfblock];
        }
        __syncthreads();
        halfblock /= 2;
    }while ( halfblock != 0 );
    // Copy to the global memory
    if (sumIndex == 0){
    		d_CallValue[blockIdx.x].Expected = s_Sum[sumIndex];
    		d_CallValue[blockIdx.x].Confidence = s_Sum[sum2Index];
    }
}
// Vanilla Option call Kernel
__global__ void vanillaOptMonteCarlo(curandState * randseed, OptionValue *d_CallValue){
    // Parameters for shared memory
    int sumIndex = threadIdx.x;
    int sum2Index = sumIndex + blockDim.x;
    
    /*------------------ SHARED MEMORY DICH ----------------*/
    extern __shared__ float s_Sum[];
    
    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Copy random number state to local memory
    curandState threadState = randseed[tid];
    
    OptionValue sum = {0, 0};
    int i;
    for( i=sumIndex; i<N_PATH; i+=blockDim.x){
        float price=0.0f;
        // Price simulation with the vanilla call option payoff function
        price = callPayoff(&threadState);
        sum.Expected += price;
        sum.Confidence += price*price;
    }
    // Copy to the shared memory
    s_Sum[sumIndex] = sum.Expected;
    s_Sum[sum2Index] = sum.Confidence;
    __syncthreads();
    // Reduce shared memory accumulators and write final result to global memory
    int halfblock = blockDim.x/2;
    // Reduction in log2(threadBlocks) steps, so threadBlock must be power of 2
    do{
        if ( sumIndex < halfblock ){
            s_Sum[sumIndex] += s_Sum[sumIndex+halfblock];
            s_Sum[sum2Index] += s_Sum[sum2Index+halfblock];
        }
        __syncthreads();
        halfblock /= 2;
    }while ( halfblock != 0 );
    // Copy to the global memory
    if (sumIndex == 0){
        d_CallValue[blockIdx.x].Expected = s_Sum[sumIndex];
        d_CallValue[blockIdx.x].Confidence = s_Sum[sum2Index];
    }
}

__global__ void cvaCallOptMC(curandState * randseed, OptionValue *d_CallValue){
    // Parameters for shared memory
    int sumIndex = threadIdx.x;
    int sum2Index = sumIndex + blockDim.x;
    /*  - SHARED MEMORY -  */
    extern __shared__ float s_Sum[];
    
    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Copy random number state to local memory
    curandState threadState = randseed[tid];
    
    float dt = OPTION.t / N_GRID;
    // Calcolo di un CVA
    // Step 1: simulare traiettoria sottostante, ad ogni istante dt calcolare prezzo opzione attualizzato con B&S
    // Step 2: calcolo CVA per ogni traiettoria e sommarlo alla variabile mean_price
    // Step 3: salvare nella memoria condivisa i CVA calcolati
    OptionValue sum = {0, 0};
    int i,j;
    for( i=sumIndex; i<N_PATH; i+=blockDim.x){
        float s, ee, t;
        float mean_price = 0;
        s = OPTION.s;
        t = OPTION.t;
        ee = device_bsCall(s,t);
        for(j=1; j <= N_GRID; j++){
            float dp = exp(-(dt*(j-1)) * INTDEF) - exp(-(dt*j) * INTDEF);
            if( (t -= dt)>=0 ){
                float z = curand_normal(&threadState);
                s = geomBrownian(s, dt, z);
                ee = device_bsCall(s,t);
            }
            else{
                ee = 0;
            }
            mean_price += dp * ee;
        }
        mean_price *= LGD;
        sum.Expected += mean_price;
        sum.Confidence += mean_price * mean_price;
    }
    // Copy to the shared memory
    s_Sum[sumIndex] = sum.Expected;
    s_Sum[sum2Index] = sum.Confidence;
    __syncthreads();
    // Reduce shared memory accumulators and write final result to global memory
    int halfblock = blockDim.x/2;
    // Reduction in log2(threadBlocks) steps, so threadBlock must be power of 2
    do{
        if ( sumIndex < halfblock ){
            s_Sum[sumIndex] += s_Sum[sumIndex+halfblock];
            s_Sum[sum2Index] += s_Sum[sum2Index+halfblock];
        }
        __syncthreads();
        halfblock /= 2;
    }while ( halfblock != 0 );
    // Copy to the global memory
    if (sumIndex == 0){
        d_CallValue[blockIdx.x].Expected = s_Sum[sumIndex];
        d_CallValue[blockIdx.x].Confidence = s_Sum[sum2Index];
    }
}

__global__ void randomSetup( curandState *randSeed ){
    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Each thread block gets different seed, threads within a thread block get different sequence numbers
    curand_init(blockIdx.x + gridDim.x, threadIdx.x, 0, &randSeed[tid]);
}

////////////////////////////////////////////////////////////////
////////////////    HOST FUNCTIONS  ////////////////////////////
////////////////////////////////////////////////////////////////

void MonteCarlo_init(dev_MonteCarloData *data){
	cudaEvent_t start, stop;
	CudaCheck( cudaEventCreate( &start ));
    CudaCheck( cudaEventCreate( &stop ));
    float time;

    /*--------------- CONSTANT MEMORY ----------------*/
    if( data->numOpt > 1){
        int n_option = data->numOpt;
        CudaCheck(cudaMemcpyToSymbol(N_OPTION,&n_option,sizeof(int)));
    }
    
    int n_path = data->path;
    printf("Numero di simulazioni per blocco:\t %d\n",n_path);
    printf("Numero di simulazioni per processo:\t %d\n",n_path/data->numThreads);
    CudaCheck(cudaMemcpyToSymbol(N_PATH,&n_path,sizeof(int)));

	// RANDOM NUMBER GENERATION KERNEL
	//Allocate states for pseudo random number generators
	CudaCheck(cudaMalloc((void **) &data->RNG, data->numBlocks * data->numThreads * sizeof(curandState)));
	//Setup for the random number sequence
    CudaCheck( cudaEventRecord( start, 0 ));
    randomSetup<<<data->numBlocks, data->numThreads>>>(data->RNG);
    cuda_error_check("\Errore nel lancio randomSetup: ","\n");
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "RNG done in ms \t %f\n", time);

    //	Host Memory Allocation
    CudaCheck( cudaEventRecord( start, 0 ));
    CudaCheck(cudaMallocHost(&data->h_CallValue, sizeof(OptionValue)*data->numBlocks));
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Host memory allocation done in ms \t %f\n", time);
    //	Device Memory Allocation
    CudaCheck( cudaEventRecord( start, 0 ));
    CudaCheck(cudaMalloc(&data->d_CallValue, sizeof(OptionValue)*data->numBlocks));
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Device memory allocation done in ms \t %f\n", time);

    CudaCheck( cudaEventDestroy( start ));
    CudaCheck( cudaEventDestroy( stop ));
}

void MonteCarlo_closing(dev_MonteCarloData *data){
    cudaEvent_t start, stop;
    CudaCheck( cudaEventCreate( &start ));
    CudaCheck( cudaEventCreate( &stop ));
    float time;
    
    CudaCheck( cudaEventRecord( start, 0 ));
    //Free memory space
	CudaCheck(cudaFree(data->RNG));
    CudaCheck(cudaFreeHost(data->h_CallValue));
    CudaCheck(cudaFree(data->d_CallValue));
    
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Free memory done in ms \t %f\n", time);
    
    CudaCheck( cudaEventDestroy( start ));
    CudaCheck( cudaEventDestroy( stop ));
}

void MonteCarlo(dev_MonteCarloData *data){
    cudaEvent_t start, stop;
    CudaCheck( cudaEventCreate( &start ));
    CudaCheck( cudaEventCreate( &stop ));
    float time, r,t;
    
	/*----------------- SHARED MEMORY -------------------*/
	int i, numShared = sizeof(float) * data->numThreads * 2;
    
    /*--------------- CONSTANT MEMORY ----------------*/
    if( data->numOpt == 1){
        r = data->sopt.r;
        t = data->sopt.t;
        CudaCheck(cudaMemcpyToSymbol(OPTION,&data->sopt,sizeof(OptionData)));
        // Time
        CudaCheck( cudaEventRecord( start, 0 ));
        vanillaOptMonteCarlo<<<data->numBlocks, data->numThreads, numShared>>>(data->RNG,(OptionValue *)(data->d_CallValue));
        cuda_error_check("\Errore nel lancio vanillaOptMonteCarlo: ","\n");
        CudaCheck( cudaEventRecord( stop, 0));
        CudaCheck( cudaEventSynchronize( stop ));
        CudaCheck( cudaEventElapsedTime( &time, start, stop ));
        printf( "Kernel done in ms \t %f\n", time);
    }
    else{
        r = data->mopt.r;
        t = data->mopt.t;
        CudaCheck(cudaMemcpyToSymbol(MOPTION,&data->mopt,sizeof(MultiOptionData)));
        // Time
        CudaCheck( cudaEventRecord( start, 0 ));
        basketOptMonteCarlo<<<data->numBlocks, data->numThreads, numShared>>>(data->RNG,(OptionValue *)(data->d_CallValue));
        cuda_error_check("\Errore nel lancio basketOptMonteCarlo: ","\n");
        CudaCheck( cudaEventRecord( stop, 0));
        CudaCheck( cudaEventSynchronize( stop ));
        CudaCheck( cudaEventElapsedTime( &time, start, stop ));
        printf( "Kernel done in ms \t %f\n", time);
    }

	//MEMORY CPY: prices per block
    // Time
    CudaCheck( cudaEventRecord( start, 0 ));
	CudaCheck(cudaMemcpy(data->h_CallValue, data->d_CallValue, data->numBlocks * sizeof(OptionValue), cudaMemcpyDeviceToHost));
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Copy from device-to-host done in ms \t %f\n", time);
    
	// Closing Monte Carlo
    float sum=0, sum2=0, price, empstd;
    long int nSim = data->numBlocks * data->path;
    // Time
    CudaCheck( cudaEventRecord( start, 0 ));
    for ( i = 0; i < data->numBlocks; i++ ){
    	sum += data->h_CallValue[i].Expected;
	    sum2 += data->h_CallValue[i].Confidence;
	}
	price = expf(-r*t) * (sum/(float)nSim);
    empstd = sqrtf((float)((float)nSim * sum2 - sum * sum)/((float)nSim * (float)(nSim - 1)));
    data->callValue.Confidence = 1.96 * empstd / (float)sqrtf((float)nSim);
    data->callValue.Expected = price;
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Call price done in ms \t %f\n", time);
    
    CudaCheck( cudaEventDestroy( start ));
    CudaCheck( cudaEventDestroy( stop ));
}

void cvaMonteCarlo(dev_MonteCarloData *data, float intdef, float lgd, int n_grid){
    cudaEvent_t start, stop;
    CudaCheck( cudaEventCreate( &start ));
    CudaCheck( cudaEventCreate( &stop ));
    float time;
    
    /*----------------- SHARED MEMORY -------------------*/
    int i, numShared = sizeof(float) * data->numThreads * 2;
    /*--------------- CONSTANT MEMORY ----------------*/
    CudaCheck(cudaMemcpyToSymbol(INTDEF, &intdef, sizeof(float)));
    CudaCheck(cudaMemcpyToSymbol(LGD, &lgd, sizeof(float)));
    CudaCheck(cudaMemcpyToSymbol(N_GRID, &n_grid, sizeof(int)));
    CudaCheck(cudaMemcpyToSymbol(OPTION, &data->sopt, sizeof(OptionData)));
    //Time
    CudaCheck( cudaEventRecord( start, 0 ));
    cvaCallOptMC<<<data->numBlocks, data->numThreads, numShared>>>(data->RNG,(OptionValue *)(data->d_CallValue));
    cuda_error_check("\Errore nel lancio cvaCallOptMC: ","\n");
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Kernel done in ms \t %f\n", time);
    
    //MEMORY CPY: prices per block
    CudaCheck(cudaMemcpy(data->h_CallValue, data->d_CallValue, data->numBlocks * sizeof(OptionValue), cudaMemcpyDeviceToHost));
    
    // Closing Monte Carlo
    float sum=0, sum2=0, price, empstd;
    long int nSim = data->numBlocks * data->path;
    CudaCheck( cudaEventRecord( start, 0 ));
    for ( i = 0; i < data->numBlocks; i++ ){
        sum += data->h_CallValue[i].Expected;
        sum2 += data->h_CallValue[i].Confidence;
    }
    price = sum/(float)nSim;
    empstd = sqrtf((float)((float)nSim * sum2 - sum * sum)/((float)nSim * (float)(nSim - 1)));
    data->callValue.Confidence = 1.96 * empstd / (float)sqrtf((float)nSim);
    data->callValue.Expected = price;
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "CVA price done in ms \t %f\n", time);
    
    CudaCheck( cudaEventDestroy( start ));
    CudaCheck( cudaEventDestroy( stop ));
}

////////////////////////////////////////////////
////////////////    WRAPPERS    ////////////////
////////////////////////////////////////////////

extern "C" OptionValue dev_basketOpt(MultiOptionData *option, int numBlocks, int numThreads, int sims){
	dev_MonteCarloData data;
    // Option
    data.mopt = *option;
    // Kernel parameters
    data.numBlocks = numBlocks;
    data.numThreads = numThreads;
    data.numOpt = N;
    data.path = sims / numBlocks;
    // Core
    MonteCarlo_init(&data);
    MonteCarlo(&data);
    MonteCarlo_closing(&data);
    
    return data.callValue;
}

extern "C" OptionValue dev_vanillaOpt(OptionData *opt, int numBlocks, int numThreads, int sims){
    dev_MonteCarloData data;
    // Option
    data.sopt = *opt;
    // Kernel parameters
    data.numBlocks = numBlocks;
    data.numThreads = numThreads;
    data.numOpt = 1;
    data.path = sims / numBlocks;
    // Core
    MonteCarlo_init(&data);
    MonteCarlo(&data);
    MonteCarlo_closing(&data);

    return data.callValue;
}

extern "C" OptionValue dev_cvaEquityOption(CVA *cva, int numBlocks, int numThreads, int sims){
    dev_MonteCarloData data;
    // Option
    data.sopt = cva->option;
    // Kernel parameters
    data.numBlocks = numBlocks;
    data.numThreads = numThreads;
    data.numOpt = 1;
    data.path = sims / numBlocks;
    // Core
    MonteCarlo_init(&data);
    cvaMonteCarlo(&data, (float)cva->defInt, (float)cva->lgd, cva->n);
    // Closing
    MonteCarlo_closing(&data);
    
    return data.callValue;
}


