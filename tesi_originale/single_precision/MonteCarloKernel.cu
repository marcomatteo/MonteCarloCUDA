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
    MultiOptionData option;
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

__device__ __constant__ MultiOptionData OPTION;
__device__ __constant__ int N_OPTION, N_PATH, N_GRID;
__device__ __constant__ float INTDEF, LGD;

__device__ void brownianVect(float *bt, curandState threadState){
	int i,j;
	float g[N];
	for(i=0;i<N_OPTION;i++)
		g[i]=curand_normal(&threadState);
	for(i=0;i<N_OPTION;i++){
		float somma = 0;
		for(j=0;j<N_OPTION;j++)
			somma += OPTION.p[i][j] * g[j];
		bt[i] = somma;
	}
	for(i=0;i<N_OPTION;i++)
		bt[i] += OPTION.d[i];
}

__device__ float basketPayoff(float *bt){
	int j;
	float s[N], st_sum=0, price;
    for(j=0;j<N_OPTION;j++)
        s[j] = OPTION.s[j] * expf((OPTION.r - 0.5 * OPTION.v[j] * OPTION.v[j])*OPTION.t+OPTION.v[j] * bt[j] * sqrtf(OPTION.t));
	// Third step: Mean price
	for(j=0;j<N_OPTION;j++)
		st_sum += s[j] * OPTION.w[j];
	// Fourth step: Option payoff
	price = st_sum - OPTION.k;

    return max(price,0);
}
/*
__device__ float geomBrownian( float *s, float *z ){
    float x = (OPTION.r - 0.5 * OPTION.v * OPTION.v) * OPTION.t + OPTION.v * sqrtf(OPTION.t) * *z;
    return *s * expf(x);
}

__device__ float callPayoff(curandState *threadState){
    float z = curand_normal(threadState);
    float s = OPTION.s * expf((OPTION.r - 0.5 * OPTION.v * OPTION.v) * OPTION.t + OPTION.v * sqrtf(OPTION.t) * z);
    return max(s - OPTION.k,0);
}
 */

__global__ void basketOptMonteCarlo(curandState * randseed, OptionValue *d_CallValue){
    int i;
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

    for( i=sumIndex; i<N_PATH; i+=blockDim.x){
    	float price=0.0f, bt[N];
    	// Random Number Generation
   		brownianVect(bt,threadState);
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
/*
__global__ void vanillaOptMonteCarlo(curandState * randseed, OptionValue *d_CallValue){
    int i;
    // Parameters for shared memory
    int sumIndex = threadIdx.x;
    int sum2Index = sumIndex + blockDim.x;
    
    //------------------ SHARED MEMORY DICH ----------------
    extern __shared__ float s_Sum[];
    
    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Copy random number state to local memory
    curandState threadState = randseed[tid];
    
    OptionValue sum = {0, 0};
    
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
    if (sumIndex == 0){
        d_CallValue[blockIdx.x].Expected = s_Sum[sumIndex];
        d_CallValue[blockIdx.x].Confidence = s_Sum[sum2Index];
    }
}

// Test di cva con simulazione percorso sottostante
__global__ void cvaCallOptMC(curandState * randseed, OptionValue *d_CallValue){
    int i,j,k;
    // Parameters for shared memory
    int sumIndex = threadIdx.x;
    int sum2Index = sumIndex + blockDim.x;
    
    //------------------ SHARED MEMORY DICH ----------------
    extern __shared__ float s_Sum[];
    
    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Copy random number state to local memory
    curandState threadState = randseed[tid];
    // Monte Carlo core
    OptionValue sum = {0, 0};
    float dt = OPTION.t / N_GRID;
    for(k=blockIdx.x; k<gridDim.x; k+=gridDim.x){
        for( i=sumIndex; i<10000; i+=blockDim.x){
            float price=0.0f, mean_price = 0.0f;
            float s[2];
            s[0] = OPTION.s;
            for(j=1; j<N_GRID; j++){
                float z = curand_normal(&threadState);
                s[1] = geomBrownian(&s[0], &z);
                float ee = max((((s[1] + s[0])/2)-OPTION.k),0);
                float dp = expf(-(dt*j-1) * (float)INTDEF) - expf(-(dt*j) * (float)INTDEF);
                mean_price += ee * dp * expf(-(dt*i) * OPTION.r);
                s[0] = s[1];
            }
            price = mean_price * LGD;
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
    }
    if (sumIndex == 0){
        d_CallValue[blockIdx.x].Expected = s_Sum[sumIndex];
        d_CallValue[blockIdx.x].Confidence = s_Sum[sum2Index];
    }
}
*/
__global__ void randomSetup( curandState *randSeed ){
    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Each thread block gets different seed, threads within a thread block get different sequence numbers
    curand_init(blockIdx.x + gridDim.x, threadIdx.x, 0, &randSeed[tid]);
}

void MonteCarlo_init(dev_MonteCarloData *data){
	cudaEvent_t start, stop;
	CudaCheck( cudaEventCreate( &start ));
    CudaCheck( cudaEventCreate( &stop ));
    float time;

    int n_option = data->numOpt;
    int n_path = data->path;
    
    /*--------------- CONSTANT MEMORY ----------------*/
    CudaCheck(cudaMemcpyToSymbol(N_OPTION,&n_option,sizeof(int)));
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

    //	Host Memory Allocation
    CudaCheck( cudaEventRecord( start, 0 ));
    CudaCheck(cudaMallocHost(&data->h_CallValue, sizeof(OptionValue)*data->numBlocks));
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Host memory allocation done in ms %f\n", time);
    //	Device Memory Allocation
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
    float time;
    
	/*----------------- SHARED MEMORY -------------------*/
	int i, numShared = sizeof(float) * data->numThreads * 2;
    
    CudaCheck( cudaEventRecord( start, 0 ));
    basketOptMonteCarlo<<<data->numBlocks, data->numThreads, numShared>>>(data->RNG,(OptionValue *)(data->d_CallValue));
    cuda_error_check("\Errore nel lancio del Kernel: ","\n");
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Kernel done in ms %f\n", time);

	//MEMORY CPY: prices per block
    // Time
    CudaCheck( cudaEventRecord( start, 0 ));
	CudaCheck(cudaMemcpy(data->h_CallValue, data->d_CallValue, data->numBlocks * sizeof(OptionValue), cudaMemcpyDeviceToHost));
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Copy from device-to-host done in ms %f\n", time);
    
	// Closing Monte Carlo
    float sum=0, sum2=0, price, empstd;
    long int nSim = data->numBlocks * data->path;
    // Time
    CudaCheck( cudaEventRecord( start, 0 ));
    for ( i = 0; i < data->numBlocks; i++ ){
    	sum += data->h_CallValue[i].Expected;
	    sum2 += data->h_CallValue[i].Confidence;
	}
	price = expf(-data->option.r*data->option.t) * (sum/(float)nSim);
    empstd = sqrtf((float)((float)nSim * sum2 - sum * sum)/((float)nSim * (float)(nSim - 1)));
    data->callValue.Confidence = 1.96 * empstd / (float)sqrtf((float)nSim);
    data->callValue.Expected = price;
    CudaCheck( cudaEventRecord( stop, 0));
    CudaCheck( cudaEventSynchronize( stop ));
    CudaCheck( cudaEventElapsedTime( &time, start, stop ));
    printf( "Call price done in ms %f\n", time);
    
    CudaCheck( cudaEventDestroy( start ));
    CudaCheck( cudaEventDestroy( stop ));
}
/*
void cvaMonteCarlo(dev_MonteCarloData *data, float intdef, float lgd){
    //----------------- SHARED MEMORY -------------------
    int i, numShared = sizeof(float) * data->numThreads * 2;
    if( data->numOpt == 1){
         //--------------- CONSTANT MEMORY ----------------
        CudaCheck(cudaMemcpyToSymbol(INTDEF,&intdef,sizeof(float)));
        CudaCheck(cudaMemcpyToSymbol(LGD,&lgd,sizeof(float)));
        CudaCheck(cudaMemcpyToSymbol(OPTION,&data->sopt,sizeof(OptionData)));
        cvaCallOptMC<<<data->numBlocks, data->numThreads, numShared>>>(data->RNG,(OptionValue *)(data->d_CallValue));
        cuda_error_check("\Errore nel lancio cvaCallOptMC: ","\n");
    }
    //MEMORY CPY: prices per block
    CudaCheck(cudaMemcpy(data->h_CallValue, data->d_CallValue, data->numBlocks * sizeof(OptionValue), cudaMemcpyDeviceToHost));
    // Closing Monte Carlo
    float sum=0, sum2=0, price, empstd;
    long int nSim = data->numBlocks * data->path;
    for ( i = 0; i < data->numBlocks; i++ ){
        sum += data->h_CallValue[i].Expected;
        sum2 += data->h_CallValue[i].Confidence;
    }
    price = sum/(float)nSim;
    empstd = sqrtf((float)((float)nSim * sum2 - sum * sum)/((float)nSim * (float)(nSim - 1)));
    data->callValue.Confidence = 1.96 * empstd / (float)sqrtf((float)nSim);
    data->callValue.Expected = price;
}
*/
extern "C" OptionValue dev_basketOpt(MultiOptionData *option, int numBlocks, int numThreads, int sims){
	dev_MonteCarloData data;
    data.option = *option;
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
    MultiOptionData option;
    option.w[0] = 1;
    option.d[0] = 0;
    option.p[0][0] = 1;
    option.s[0] = opt->s;
    option.v[0] = opt->v;
    option.k = opt->k;
    option.r = opt->r;
    option.t = opt->t;
    
    dev_MonteCarloData data;
    data.option = option;
    data.numBlocks = numBlocks;
    data.numThreads = numThreads;
    data.numOpt = 1;
    data.path = sims / numBlocks;

    MonteCarlo_init(&data);
    MonteCarlo(&data);
    MonteCarlo_closing(&data);

    return data.callValue;
}

extern "C" void dev_cvaEquityOption(CVA *cva, int numBlocks, int numThreads, int sims){
    int i;
    float dt, t;
    dev_MonteCarloData data;
    MultiOptionData option;
    option.w[0] = 1;
    option.d[0] = 0;
    option.p[0][0] = 1;
    option.s[0] = *cva->opt.s;
    option.v[0] = *cva->opt.v;
    option.k = cva->opt.k;
    option.r = cva->opt.r;
    option.t = cva->opt.t;
    
    dt = cva->opt.t / (float)cva->n;
    t = cva->opt.t;
    // Option
    data.option = cva->opt;
    
    // Kernel parameters
    data.numBlocks = numBlocks;
    data.numThreads = numThreads;
    data.numOpt = cva->ns;
    data.path = sims / numBlocks;
    
    MonteCarlo_init(&data);

    // Original option price
    MonteCarlo(&data);
    cva->ee[0] = data.callValue;

    // Expected Exposures (ee), Default probabilities (dp,fp)
    float sommaProdotto1=0;
    //float sommaProdotto2=0;
	for( i=1; i < (cva->n+1); i++){
		if((t -= (dt))<0){
			cva->ee[i].Confidence = 0;
			cva->ee[i].Expected = 0;
		}
		else{
            data.option.t = t;
			MonteCarlo(&data);
            //data.callValue.Expected = (data.callValue.Expected + cva->ee[i-1].Expected)/2;
			cva->ee[i] = data.callValue;
		}
        cva->dp[i] = expf(-(dt*i) * cva->defInt) - expf(-(dt*(i+1)) * cva->defInt);
		//cva->fp[i] = expf(-(dt)*(i-1) * cva->credit.fundingspread / 100 / cva->credit.lgd) - expf(-(dt*i) * cva->credit.fundingspread / 100 / cva->credit.lgd );
        sommaProdotto1 += cva->ee[i].Expected * cva->dp[i];
		//sommaProdotto2 += cva->ee[i].Expected * cva->fp[i];
	}
	// CVA and FVA
	cva->cva = sommaProdotto1 * cva->lgd;
	//cva->fva = -sommaProdotto2*cva->credit.lgd;

	// Closing
	MonteCarlo_closing(&data);
}

// Test cva con simulazione percorso sottostante
/*
extern "C" OptionValue dev_cvaEquityOption_opt(CVA *cva, int numBlocks, int numThreads, int sims){
    dev_MonteCarloData data;
    // Option
    if(cva->ns ==1){
        data.sopt = cva->option;
    }
    else{
        data.mopt = cva->opt;
    }
    // Kernel parameters
    data.numBlocks = numBlocks;
    data.numThreads = numThreads;
    data.numOpt = cva->ns;
    data.path = sims / numBlocks;
    
    MonteCarlo_init(&data);
    cvaMonteCarlo(&data, (float)cva->defInt, (float)cva->lgd);
    
    // Closing
    MonteCarlo_closing(&data);
    
    return data.callValue;
}
*/

