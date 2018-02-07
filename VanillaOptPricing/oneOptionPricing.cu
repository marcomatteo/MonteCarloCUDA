/*              Testing GPU functions for Monte Carlo Simulation             */
/*                   Author: Buzzulini Marco Matteo                          */

// "CUDA by EXAMPLE" library for errors
#include "book.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define MAX_THREADS 1024
#define MAX_BLOCKS 65535
#define MAX_OP 45000

// Default Option data
typedef struct {
    float S;   // Underlying asset price
    float K;   // Strike-price
    float T;   // Time to maturity
    float R;   // Risk-neutral rate
    float V;   // Volatility
} OptionData;

// Default Option Values
typedef struct {
    float Expected;
    float Confidence;
} OptionValue;

// Preprocessed input Option Data
typedef struct {
    double S;
    double K;
    double MuPerT;
    double VPerSqrtT;
} ModOptionData;

// For Monte Carlo sample prices sum
typedef struct {
    double Expected;
    double Confidence;
} ModOptionValue;

void oneOptMonteCarloGPU( const OptionData *, OptionValue *, const int, int, const int);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////               GPU functions               /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* ------ Device functions ------*/
__device__ double Payoff(double S, double K, double MuPerT, double VPerSqrtT, double g) {
    double callValue = S * exp(MuPerT + VPerSqrtT * g) - K;
    return (callValue > 0.0) ? callValue : 0.0;
}

__device__ void SumReductionKernel( double *s1, double *s2 ){
    int halfblock = blockDim.x/2;
    int cacheIndex = threadIdx.x;
    
    while ( halfblock != 0 ){
        if ( cacheIndex < halfblock ){
            s1[cacheIndex] += s1[cacheIndex+halfblock];
            s2[cacheIndex] += s2[cacheIndex+halfblock];
            __syncthreads();
        }
        halfblock /= 2;
    }
}

/* ------ Host to Device functions ------*/
__global__ void MonteCarloKernel( curandState * randseed,
                                 const ModOptionData * d_OptionData,
                                 ModOptionValue * d_CallValue,
                                 const int N ){
    const int N_OPT = gridDim.x;
    const int THREAD_N = blockDim.x;
    __shared__ double s_Sum[MAX_THREADS];
    __shared__ double s_Sum2[MAX_THREADS];
    int optionIndex;
    
    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Copy random number state to local memory
    curandState threadState = randseed[tid];
    
    for( optionIndex = blockIdx.x; optionIndex < N_OPT; optionIndex += gridDim.x)
    {
        const double        S = d_OptionData[optionIndex].S;
        const double        K = d_OptionData[optionIndex].K;
        const double    MuPerT = d_OptionData[optionIndex].MuPerT;
        const double VPerSqrtT = d_OptionData[optionIndex].VPerSqrtT;
        int idxSum;
        
        /*  Monte Carlo core computation */
        for ( idxSum = threadIdx.x; idxSum < MAX_THREADS; idxSum += blockDim.x)
        {
            ModOptionValue sum = {0, 0};
            int i;
            for (i = idxSum; i < N; i += THREAD_N) {
                double              g = curand_normal(&threadState);
                double              sample = Payoff(S, K, MuPerT, VPerSqrtT, g);
                sum.Expected    += sample;
                sum.Confidence  += sample * sample;
            }
            s_Sum[idxSum]  = sum.Expected;
            s_Sum2[idxSum] = sum.Confidence;
        }
        __syncthreads();
        //Reduce shared memory accumulators and write final result to global memory
        //SumReductionKernel( s_Sum, s_Sum2 );
        int halfblock = blockDim.x/2;
        int cacheIndex = threadIdx.x;
        do{
            if ( cacheIndex < halfblock ){
                s_Sum[cacheIndex] += s_Sum[cacheIndex+halfblock];
                s_Sum2[cacheIndex] += s_Sum2[cacheIndex+halfblock];
                __syncthreads();
            }
            halfblock /= 2;
        }while ( halfblock != 0 );
        __syncthreads();
        
        // Keeping the first element for each block using one thread
        if (cacheIndex == 0){
            d_CallValue[optionIndex].Expected = s_Sum[0];
            d_CallValue[optionIndex].Confidence = s_Sum2[0];
        }
    }
}

__global__ void randomSetup( curandState *randSeed ){
    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Each threadblock gets different seed, threads within a threadblock get different sequence numbers
    curand_init(blockIdx.x + gridDim.x, threadIdx.x, 0, &randSeed[tid]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////               CPU functions               /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void oneOptMonteCarloGPU( const OptionData h_CallData, OptionValue *h_CallValue, const int N_OPT, const int N_THREADS, const int N ){
    ModOptionData   *h_CallBufferData, *d_CallData;
    ModOptionValue  *h_CallBufferValue, *d_CallValue;
    curandState *RNG;
    cudaEvent_t start, stop;
    
    HANDLE_ERROR( cudaEventCreate( &start ));
    HANDLE_ERROR( cudaEventCreate( &stop ));
    float time;
    
    // Allocate device memory
    HANDLE_ERROR(cudaMalloc(&d_CallData, sizeof(ModOptionData)*(N_OPT)));
    HANDLE_ERROR(cudaMalloc(&d_CallValue, sizeof(ModOptionValue)*(N_OPT)));
    // Allocate pinned-host memory for faster and asynchronous copies
    HANDLE_ERROR(cudaHostAlloc(&h_CallBufferData, sizeof(ModOptionData)*(N_OPT),cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc(&h_CallBufferValue, sizeof(ModOptionValue)*(N_OPT),cudaHostAllocDefault));
    
    // Allocate states for pseudo random number generators
    HANDLE_ERROR(cudaMalloc((void **) &RNG, N_OPT * N_THREADS * sizeof(curandState)));
    //  Setup for the random number sequence
    randomSetup<<<N_OPT, N_THREADS>>>(RNG);
    
    // Core function
    int i;
    for (i = 0; i < N_OPT; i++){
        h_CallBufferData[i].S        = h_CallData.S;
        h_CallBufferData[i].K        = h_CallData.K;
        h_CallBufferData[i].MuPerT   = (h_CallData.R - h_CallData.V * h_CallData.V * 0.5) * h_CallData.T;
        h_CallBufferData[i].VPerSqrtT= h_CallData.V * sqrt(h_CallData.T);
    }
    HANDLE_ERROR(cudaMemcpy(d_CallData, h_CallBufferData, N_OPT * sizeof(ModOptionData), cudaMemcpyHostToDevice));
    
    HANDLE_ERROR( cudaEventRecord( start, 0 ));
    MonteCarloKernel<<<N_OPT, N_THREADS>>>(RNG,(ModOptionData *)(d_CallData),(ModOptionValue *)(d_CallValue), N );
    HANDLE_ERROR( cudaEventRecord( stop, 0));
    HANDLE_ERROR( cudaEventSynchronize( stop ));
    HANDLE_ERROR( cudaEventElapsedTime( &time, start, stop ));
    //printf( "Computations done in %.5f milliseconds\n", time);
    HANDLE_ERROR( cudaEventDestroy( start ));
    HANDLE_ERROR( cudaEventDestroy( stop ));
    HANDLE_ERROR(cudaMemcpy(h_CallBufferValue, d_CallValue, N_OPT * sizeof(ModOptionValue), cudaMemcpyDeviceToHost));
    
    // Closing MonteCarlo
    int path = N * N_OPT;
    long double sum=0, sum2=0, emp_stDev;
    const double    RT = h_CallData.R * h_CallData.T;
    for ( i = 0; i < N_OPT; i++ ){
        sum += h_CallBufferValue[i].Expected;
        sum2 += h_CallBufferValue[i].Confidence;
    }
    h_CallValue->Expected = exp(-RT) * (sum/(float)path);
    emp_stDev = sqrt(
                     ((double)path * sum2 - sum * sum)
                     /
                     ((double)path * (double)(path - 1))
                    );
    h_CallValue->Confidence = 1.96 * (float)emp_stDev / (float)sqrt(path);
    
    HANDLE_ERROR(cudaFree(RNG));
    HANDLE_ERROR(cudaFreeHost(h_CallBufferValue));
    HANDLE_ERROR(cudaFreeHost(h_CallBufferData));
    HANDLE_ERROR(cudaFree(d_CallValue));
    HANDLE_ERROR(cudaFree(d_CallData));
}

void adjustSize( int *blocks, int *threads, int *op ){
    cudaDeviceProp deviceProp;
    HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
    int maxThreads = deviceProp.maxThreadsPerBlock;
    *blocks = (( *blocks> MAX_BLOCKS) ? MAX_BLOCKS : *blocks );
    *threads = (( *threads> maxThreads) ? maxThreads : *threads );
    *op = (( *op> MAX_OP) ? MAX_OP : *op );
}

// Cumulative Normal Distribution Function
static double CND(double d){
    const double       A1 = 0.31938153;
    const double       A2 = -0.356563782;
    const double       A3 = 1.781477937;
    const double       A4 = -1.821255978;
    const double       A5 = 1.330274429;
    const double ONEOVER2PI = 0.39894228040143267793994605993438;
    double K = 1.0 / (1.0 + 0.2316419 * fabs(d));
    double cnd = ONEOVER2PI * exp(- 0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    
    if (d > 0)
        cnd = 1.0 - cnd;
    return cnd;
}

// Generator of a normal pseudo-random number
static float gaussian(){
    float
    x = (float)rand()/(float)RAND_MAX,
    y = (float)rand()/(float)RAND_MAX;
    return ( sqrt( -2.0*log(x) ) * cos( 2*M_PI*y ) );
}

// Black-Scholes pricing formula for call option
void BlackScholesCall ( double *value, OptionData option){
    double s = option.S;
    double k = option.K;
    double t = option.T;
    double r = option.R;
    double v = option.V;
    
    double    d1 = ( log(s / k) + (r + 0.5 * v * v) * t) / (v * sqrt(t));
    double    d2 = d1 - v * sqrt(t);
    double cndD1 = CND(d1);
    double cndD2 = CND(d2);
    double expRT = exp(- r * t);
    
    *value = s * cndD1 - k * expRT * cndD2;
}

// Call payoff
static double callPayoff( double S, double K, double T, double R, double V){
    double value = S * exp( (R - 0.5 * V * V) * T + gaussian() * sqrt(T) * V ) - K;
    return (value>0) ? (value):(0);
}

// Monte Carlo simulation
void MonteCarloCPU ( OptionData *option, OptionValue *callValue, int path){
    long double sum, var_sum, price, emp_stdev;
    double s = option->S;
    double k = option->K;
    double t = option->T;
    double r = option->R;
    double v = option->V;
    
    sum = var_sum = 0.0f;
    srand((unsigned)time(NULL));
    
    for( int i=0; i<path; i++){
        price = callPayoff(s,k,t,r,v);
        sum += price;
        var_sum += price * price;
    }
    
    price = exp(-r*t) * (sum/(double)path);
    emp_stdev = sqrt(
                     ((double)path * var_sum - sum * sum)
                     /
                     ((double)path * (double)(path - 1))
                     );
    
    callValue->Expected = price;
    callValue->Confidence = 1.96 * emp_stdev/sqrt(path);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////                Main Program                /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void printOption (OptionData);
void printPrice (OptionValue);
void printOptionPrice (OptionData, OptionValue);

int main (){
    // Options variables
    OptionData CallData;
    OptionValue gpuCallValue, cpuCallValue;
    double bsPrice=0, difference=0;
    // Parallel computation variables
    int nBlocks, nThreads, nOP;
    // Time variables
    float timeGpu, timeCpu, comp;
    clock_t cpuStart, cpuStop;
    cudaEvent_t gpuStart, gpuStop;
    // Time events inizialize
    HANDLE_ERROR( cudaEventCreate( &gpuStart ));
    HANDLE_ERROR( cudaEventCreate( &gpuStop ));
    
    // Main interface, input
    printf( "\n\t\t-\t Monte Carlo one option pricing with CUDA \t-\n\n");
    printf( "-\tParallel computing parameters:\t-\n");
    printf( "Be careful! These parameters will run even on the CPU (blocks * operations)\n");
    printf( "Digit the number of blocks: ");
    scanf( "%d", &nBlocks);
    printf( "Digit the number of threads: ");
    scanf( "%d", &nThreads);
    printf( "Digit how many operations: ");
    scanf( "%d", &nOP);
    // Control parallel parameters
    adjustSize( &nBlocks, &nThreads, &nOP );
    printf( "\n\n-\tOption parameters:\t-\n");
    printf( "Digit the price of the underlying asset: € ");
    scanf( "%f", &CallData.S );
    printf( "Digit the strike price: € ");
    scanf( "%f", &CallData.K );
    printf( "Digit the time to maturity: (in years) ");
    scanf( "%f", &CallData.T );
    printf( "Digit the volatility: ");
    scanf( "%f", &CallData.V );
    printf( "Digit the risk-free rate: ");
    scanf( "%f", &CallData.R );
    
    printOption( CallData );
    
    // Black & Scholes price
    BlackScholesCall( &bsPrice, CallData );
    printf( "\n\n-\tOption price with Black & Scholes: %lf\t-\n", bsPrice);
    printf( "\n-\tAmount of simulations: %d\t-\n", (nBlocks*nOP));
    // Monte Carlo on GPU
    printf( "\n\n-\tRunning Monte Carlo within GPU:\t-\n");
    printf( "Parallel parameters: %d Blocks, %d Threads with a path of %d interactions per block\n", nBlocks, nThreads, nOP);
    // Time start record for GPU
    HANDLE_ERROR( cudaEventRecord( gpuStart, 0 ));
    // Monte Carlo function
    oneOptMonteCarloGPU( CallData, &gpuCallValue, nBlocks, nThreads, nOP );
    HANDLE_ERROR( cudaEventRecord( gpuStop, 0 ));
    HANDLE_ERROR( cudaEventSynchronize( gpuStop ));
    HANDLE_ERROR( cudaEventElapsedTime( &timeGpu, gpuStart, gpuStop ));
    printf( "Monte Carlo simulation done!\tIt takes %f milliseconds.\n", timeGpu);
    // GPU results
    printPrice( gpuCallValue );
    // Free the memory for the GPU part
    HANDLE_ERROR( cudaEventDestroy( gpuStart ));
    HANDLE_ERROR( cudaEventDestroy( gpuStop ));
    
    /* Comparing Monte Carlo Gpu with Black & Scholes formula */
    difference = fabs(((double)gpuCallValue.Expected - bsPrice ));
    printf( "The difference from the Black-Scholes formula is: € %f\n", difference);
    
    // Monte Carlo on CPU
    printf( "\n\n-\tRunning Monte Carlo within CPU:\t-\n");
    // Time record with cudaEvent and clock
    cpuStart = clock();
    MonteCarloCPU( &CallData, &cpuCallValue, (nBlocks*nOP));
    cpuStop = clock();
    timeCpu = ((float)(cpuStop - cpuStart)/CLOCKS_PER_SEC)*1000;
    printf( "Monte Carlo simulation done!\tIt takes %f milliseconds.\n", timeCpu);
    // CPU result
    printPrice( cpuCallValue );
    // Compared with Black & Scholes formula
    difference = fabs(((double)cpuCallValue.Expected - bsPrice ));
    printf( "The difference from Black-Scholes formula is: € %f\n", difference);
    
    // Comparing time spent with the two methods
    printf( "\n\n-\tComparing results:\t-\n");
    comp = (timeCpu-timeGpu) / timeGpu;
    printf( "The GPU runs the Monte Carlo simulation %.2f %% %s\n", comp*100, (timeCpu>timeGpu)?("faster."):("slower."));
    
    printf( "\n\t\t\tThat's all folks!!!\n");
    return 0;
}

void printOption ( OptionData o){
    printf("\n-\tOption Data\t-\n");
    printf("The underlying asset price is:\t\t € %.2f\n", o.S);
    printf("The strike price is:\t\t\t € %.2f\n", o.K);
    printf("The time to maturity is:\t\t %.2f %s\n", o.T, (o.T>1)?("years"):("year"));
    printf("The risk-neutral rate is:\t\t %.2f %%\n", o.R * 100);
    printf("The volatility is:\t\t\t %.2f %%\n", o.V * 100);
}

void printPrice( OptionValue v){
    float price, price_min, price_max;
    price = v.Expected;
    price_min = v.Expected - v.Confidence;
    price_max = v.Expected + v.Confidence;
    printf( "\n\tThe price is: €%f with confidence 95%%: %f - %f\n\n", price, price_min, price_max);
}

void printOptionPrice ( OptionData o, OptionValue v ){
    printOption( o );
    printPrice( v );
}

