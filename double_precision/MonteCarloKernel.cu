/*
 * MonteCarloKernel.cu
 *
 *  Created on: 06/feb/2018
 *  Author: marco
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

// Inizializzazione per Monte Carlo da svolgere una volta sola
void MonteCarlo_init(dev_MonteCarloData *data);
// Liberazione della memoria da svolgere una volta sola
void MonteCarlo_closing(dev_MonteCarloData *data);
// Metodo Monte Carlo
void MonteCarlo(dev_MonteCarloData *data);
// Metodo Monte Carlo per il calcolo di un CVA
void cvaMonteCarlo(dev_MonteCarloData *data, double intdef, double lgd, int n_grid);

////////////////////////////////////////////////////////////////
////////////////    CONSTANT MEMORY     ////////////////////////
////////////////////////////////////////////////////////////////

__device__ __constant__ MultiOptionData MOPTION;
__device__ __constant__ OptionData OPTION;
__device__ __constant__ int N_OPTION, N_PATH, N_GRID;
__device__ __constant__ double INTDEF, LGD;

////////////////////////////////////////////////////////////////
////////////////    KERNEL FUNCTIONS    ////////////////////////
////////////////////////////////////////////////////////////////

__device__ void brownianVect(double *bt, curandState *threadState){
    int i,j;
    double g[N];
    for(i=0;i<N_OPTION;i++)
        g[i]=curand_normal(threadState);
    for(i=0;i<N_OPTION;i++){
        double somma = 0;
        for(j=0;j<N_OPTION;j++)
            somma += MOPTION.p[i][j] * g[j];
        bt[i] = somma;
    }
    for(i=0;i<N_OPTION;i++)
        bt[i] += MOPTION.d[i];
}

__device__ double basketPayoff(double *bt){
    int j;
    double s[N], st_sum=0, price;
    for(j=0;j<N_OPTION;j++)
        s[j] = MOPTION.s[j] * exp((MOPTION.r - 0.5 * MOPTION.v[j] * MOPTION.v[j])*MOPTION.t+MOPTION.v[j] * bt[j] * sqrt(MOPTION.t));
    // Third step: Mean price
    for(j=0;j<N_OPTION;j++)
        st_sum += s[j] * MOPTION.w[j];
    // Fourth step: Option payoff
    price = st_sum - MOPTION.k;
    
    return max(price,0);
}

__device__ double geomBrownian( double s, double t, double z ){
    double x = (OPTION.r - 0.5 * OPTION.v * OPTION.v) * t + OPTION.v * sqrt(t) * z;
    return s * exp(x);
}

__device__ double callPayoff(curandState *threadState){
    double z = curand_normal(threadState);
    double s = OPTION.s * exp((OPTION.r - 0.5 * OPTION.v * OPTION.v) * OPTION.t + OPTION.v * sqrt(OPTION.t) * z);
    return max(s - OPTION.k,0);
}

// Approssimazione di Hastings della funzione cumulata di una v.a. gaussiana
__device__ double cnd(double d){
    const double       A1 = 0.31938153;
    const double       A2 = -0.356563782;
    const double       A3 = 1.781477937;
    const double       A4 = -1.821255978;
    const double       A5 = 1.330274429;
    const double ONEOVER2PI = 0.39894228040143267793994605993438;
    double K = 1.0 / (1.0 + 0.2316419 * fabs(d));
    double cnd = ONEOVER2PI * exp(- 0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    if (d > 0)
        return 1.0 - cnd;
    else
        return cnd;
}

// Prezzo di una opzione call secondo la formula di Black & Scholes
__device__ double device_bsCall ( double s, double t){
    double d1 = ( log(s / OPTION.k) + (OPTION.r + 0.5 * OPTION.v * OPTION.v) * t) / (OPTION.v * sqrt(t));
    double d2 = d1 - OPTION.v * sqrt(t);
    return s * cnd(d1) - OPTION.k * exp(- OPTION.r * t) * cnd(d2);
}

__global__ void basketOptMonteCarlo(curandState * randseed, OptionValue *d_CallValue){
    int i;
    // Parameters for shared memory
    int sumIndex = threadIdx.x;
    int sum2Index = sumIndex + blockDim.x;
    
    /*------------------ SHARED MEMORY DICH ----------------*/
    extern __shared__ double s_Sum[];
    
    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Copy random number state to local memory
    curandState threadState = randseed[tid];
    
    OptionValue sum = {0, 0};
    
    for( i=sumIndex; i<N_PATH; i+=blockDim.x){
        double price=0.0f, bt[N];
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
    if (sumIndex == 0){
        d_CallValue[blockIdx.x].Expected = s_Sum[sumIndex];
        d_CallValue[blockIdx.x].Confidence = s_Sum[sum2Index];
    }
}

__global__ void vanillaOptMonteCarlo(curandState * randseed, OptionValue *d_CallValue){
    int i;
    // Parameters for shared memory
    int sumIndex = threadIdx.x;
    int sum2Index = sumIndex + blockDim.x;
    
    /*------------------ SHARED MEMORY DICH ----------------*/
    extern __shared__ double s_Sum[];
    
    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Copy random number state to local memory
    curandState threadState = randseed[tid];
    
    OptionValue sum = {0, 0};
    
    for( i=sumIndex; i<N_PATH; i+=blockDim.x){
        double price=0.0f;
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
    if (sumIndex == 0){
        d_CallValue[blockIdx.x].Expected = s_Sum[sumIndex];
        d_CallValue[blockIdx.x].Confidence = s_Sum[sum2Index];
    }
}

__global__ void cvaCallOptMC(curandState * randseed, OptionValue *d_CallValue){
    int i,j;
    // Parameters for shared memory
    int sumIndex = threadIdx.x;
    int sum2Index = sumIndex + blockDim.x;
    
    /*------------------ SHARED MEMORY DICH ----------------*/
    extern __shared__ double s_Sum[];
    
    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Copy random number state to local memory
    curandState threadState = randseed[tid];
    
    double dt = OPTION.t / N_GRID;
    // Idea originaria: ogni blocco calcola un CVA
    // Invece, il problema dev'essere suddiviso in:
    // Step 1: simulare traiettoria sottostante, ad ogni istante dt calcolare prezzo opzione attualizzato con B&S
    // Step 2: calcolo CVA per ogni traiettoria e sommarlo alla variabile mean_price
    // Step 3: salvare nella memoria condivisa i CVA calcolati
    OptionValue sum = {0, 0};
    
    for( i=sumIndex; i<N_PATH; i+=blockDim.x){
        double s[2], call;
        double mean_price = 0;
        s[0] = OPTION.s;
        call = device_bsCall(s[0],OPTION.t);
        for(j=1; j <= N_GRID; j++){
            double z = curand_normal(&threadState);
            double dp = exp(-(dt*(j-1)) * INTDEF) - exp(-(dt*j) * INTDEF);
            s[1] = geomBrownian(s[0], dt, z);
            call = device_bsCall(s[1],(OPTION.t - (j*dt)));
            mean_price += dp * call;
            s[0] = s[1];
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
    printf("Numero di simulazioni per blocco: %d\n",n_path);
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
    printf( "RNG done in ms %f\n", time);
    
    //    Host Memory Allocation
    CudaCheck( cudaEventRecord( start, 0 ));
    CudaCheck(cudaMallocHost(&data->h_CallValue, sizeof(OptionValue)*data->numBlocks));
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Host memory allocation done in ms %f\n", time);
    //    Device Memory Allocation
    CudaCheck( cudaEventRecord( start, 0 ));
    CudaCheck(cudaMalloc(&data->d_CallValue, sizeof(OptionValue)*data->numBlocks));
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Device memory allocation done in ms %f\n", time);
    
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
    printf( "Free memory done in ms %f\n", time);
    
    CudaCheck( cudaEventDestroy( start ));
    CudaCheck( cudaEventDestroy( stop ));
}

void MonteCarlo(dev_MonteCarloData *data){
    cudaEvent_t start, stop;
    CudaCheck( cudaEventCreate( &start ));
    CudaCheck( cudaEventCreate( &stop ));
    float time, r,t;
    
    /*----------------- SHARED MEMORY -------------------*/
    int i, numShared = sizeof(double) * data->numThreads * 2;
    
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
        printf( "Kernel done in ms %f\n", time);
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
        printf( "Kernel done in ms %f\n", time);
    }
    
    //MEMORY CPY: prices per block
    // Time
    CudaCheck( cudaEventRecord( start, 0 ));
    CudaCheck(cudaMemcpy(data->h_CallValue, data->d_CallValue, data->numBlocks * sizeof(OptionValue), cudaMemcpyDeviceToHost));
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Copy from device-to-host done in ms %f\n", time);
    
    // Closing Monte Carlo
    double sum=0, sum2=0, price, empstd;
    long int nSim = data->numBlocks * data->path;
    // Time
    CudaCheck( cudaEventRecord( start, 0 ));
    for ( i = 0; i < data->numBlocks; i++ ){
        sum += data->h_CallValue[i].Expected;
        sum2 += data->h_CallValue[i].Confidence;
    }
    price = exp(-r*t) * (sum/(double)nSim);
    empstd = sqrt((double)((double)nSim * sum2 - sum * sum)/((double)nSim * (double)(nSim - 1)));
    data->callValue.Confidence = 1.96 * empstd / (double)sqrt((double)nSim);
    data->callValue.Expected = price;
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Call price done in ms %f\n", time);
    
    CudaCheck( cudaEventDestroy( start ));
    CudaCheck( cudaEventDestroy( stop ));
}

void cvaMonteCarlo(dev_MonteCarloData *data, double intdef, double lgd, int n_grid){
    cudaEvent_t start, stop;
    CudaCheck( cudaEventCreate( &start ));
    CudaCheck( cudaEventCreate( &stop ));
    float time;
    
    /*----------------- SHARED MEMORY -------------------*/
    int i, numShared = sizeof(double) * data->numThreads * 2;
    /*--------------- CONSTANT MEMORY ----------------*/
    CudaCheck(cudaMemcpyToSymbol(INTDEF, &intdef, sizeof(double)));
    CudaCheck(cudaMemcpyToSymbol(LGD, &lgd, sizeof(double)));
    CudaCheck(cudaMemcpyToSymbol(N_GRID, &n_grid, sizeof(int)));
    CudaCheck(cudaMemcpyToSymbol(OPTION, &data->sopt, sizeof(OptionData)));
    //Time
    CudaCheck( cudaEventRecord( start, 0 ));
    cvaCallOptMC<<<data->numBlocks, data->numThreads, numShared>>>(data->RNG,(OptionValue *)(data->d_CallValue));
    cuda_error_check("\Errore nel lancio cvaCallOptMC: ","\n");
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Kernel done in ms %f\n", time);
    
    //MEMORY CPY: prices per block
    CudaCheck(cudaMemcpy(data->h_CallValue, data->d_CallValue, data->numBlocks * sizeof(OptionValue), cudaMemcpyDeviceToHost));
    
    // Closing Monte Carlo
    double sum=0, sum2=0, price, empstd;
    long int nSim = data->numBlocks * data->path;
    CudaCheck( cudaEventRecord( start, 0 ));
    for ( i = 0; i < data->numBlocks; i++ ){
        sum += data->h_CallValue[i].Expected;
        sum2 += data->h_CallValue[i].Confidence;
    }
    price = sum/(double)nSim;
    empstd = sqrt((double)((double)nSim * sum2 - sum * sum)/((double)nSim * (double)(nSim - 1)));
    data->callValue.Confidence = 1.96 * empstd / (double)sqrt((double)nSim);
    data->callValue.Expected = price;
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "CVA price done in ms %f\n", time);
    
    CudaCheck( cudaEventDestroy( start ));
    CudaCheck( cudaEventDestroy( stop ));
}

////////////////////////////////////////////////
////////////////    WRAPPERS    ////////////////
////////////////////////////////////////////////

extern "C" OptionValue dev_basketOpt(MultiOptionData *option, int numBlocks, int numThreads, int sims){
    dev_MonteCarloData data;
    data.mopt = *option;
    data.numBlocks = numBlocks;
    data.numThreads = numThreads;
    data.numOpt = N;
    data.path = sims / numBlocks;
    
    MonteCarlo_init(&data);
    MonteCarlo(&data);
    MonteCarlo_closing(&data);
    
    return data.callValue;
}

extern "C" OptionValue dev_vanillaOpt(OptionData *opt, int numBlocks, int numThreads, int sims){
    dev_MonteCarloData data;
    data.sopt = *opt;
    data.numBlocks = numBlocks;
    data.numThreads = numThreads;
    data.numOpt = 1;
    data.path = sims / numBlocks;
    
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
    
    MonteCarlo_init(&data);
    cvaMonteCarlo(&data, (double)cva->defInt, (double)cva->lgd, cva->n);
    
    // Closing
    MonteCarlo_closing(&data);
    
    return data.callValue;
}


