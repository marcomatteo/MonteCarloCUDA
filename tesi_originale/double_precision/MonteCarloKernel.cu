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

__device__ __constant__ MultiOptionData MOPTION;
__device__ __constant__ OptionData OPTION;
__device__ __constant__ int N_OPTION, N_PATH, N_GRID;
__device__ __constant__ float INTDEF, LGD;

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
    for(j=0;j<N_OPTION;j++){
        double geomBt = (MOPTION.r - 0.5 * MOPTION.v[j] * MOPTION.v[j])*MOPTION.t + MOPTION.v[j] * bt[j] * sqrt(MOPTION.t);
	     s[j] = MOPTION.s[j] * exp(geomBt);
    }
	// Third step: Mean price
	for(j=0;j<N_OPTION;j++)
		st_sum += s[j] * MOPTION.w[j];
	// Fourth step: Option payoff
	price = st_sum - MOPTION.k;

    return max(price,0);
}

__device__ double geomBrownian( double *s, double *z ){
    double x = (OPTION.r - 0.5 * OPTION.v * OPTION.v) * OPTION.t + OPTION.v * sqrt(OPTION.t) * *z;
    return *s * exp(x);
}

__device__ double callPayoff(curandState *threadState){
    double s, z = curand_normal(threadState);
    s = geomBrownian(&OPTION.s, &z);
    return max(s - OPTION.k,0);
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

// Test di cva con simulazione percorso sottostante
__global__ void cvaCallOptMC(curandState * randseed, OptionValue *d_CallValue){
    int i,j,k;
    // Parameters for shared memory
    int sumIndex = threadIdx.x;
    int sum2Index = sumIndex + blockDim.x;
    
    /*------------------ SHARED MEMORY DICH ----------------*/
    extern __shared__ double s_Sum[];
    
    // Global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Copy random number state to local memory
    curandState threadState = randseed[tid];
    // Monte Carlo core
    OptionValue sum = {0, 0};
    double dt = OPTION.t / N_GRID;
    for(k=blockIdx.x; k<gridDim.x; k+=gridDim.x){
        for( i=sumIndex; i<10000; i+=blockDim.x){
            double price=0.0f, mean_price = 0.0f;
            double s[2];
            s[0] = OPTION.s;
            for(j=1; j<N_GRID; j++){
                double z = curand_normal(&threadState);
                s[1] = geomBrownian(&s[0], &z);
                double ee = max((((s[1] + s[0])/2)-OPTION.k),0);
                double dp = exp(-(dt*j-1) * (double)INTDEF) - exp(-(dt*j) * (double)INTDEF);
                mean_price += ee * dp * exp(-(dt*i) * OPTION.r);
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

    /*--------------- CONSTANT MEMORY ----------------*/
    if( data->numOpt > 1){
        int n_option = data->numOpt;
        CudaCheck(cudaMemcpyToSymbol(N_OPTION,&n_option,sizeof(int)));
    }
    
    int n_path = data->path;
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
    printf( "RNG done in %f milliseconds\n", time);

    //	Host Memory Allocation
    CudaCheck(cudaMallocHost(&data->h_CallValue, sizeof(OptionValue)*data->numBlocks));
    //	Device Memory Allocation
    CudaCheck(cudaMalloc(&data->d_CallValue, sizeof(OptionValue)*data->numBlocks));

    CudaCheck( cudaEventDestroy( start ));
    CudaCheck( cudaEventDestroy( stop ));
}

void MonteCarlo_closing(dev_MonteCarloData *data){
	//Free memory space
	CudaCheck(cudaFree(data->RNG));
    CudaCheck(cudaFreeHost(data->h_CallValue));
    CudaCheck(cudaFree(data->d_CallValue));
}

void MonteCarlo(dev_MonteCarloData *data){
    double r,t;
	/*----------------- SHARED MEMORY -------------------*/
	int i, numShared = sizeof(double) * data->numThreads * 2;
    
    /*--------------- CONSTANT MEMORY ----------------*/
    if( data->numOpt == 1){
        r = data->sopt.r;
        t = data->sopt.t;
        CudaCheck(cudaMemcpyToSymbol(OPTION,&data->sopt,sizeof(OptionData)));
        vanillaOptMonteCarlo<<<data->numBlocks, data->numThreads, numShared>>>(data->RNG,(OptionValue *)(data->d_CallValue));
        cuda_error_check("\Errore nel lancio vanillaOptMonteCarlo: ","\n");

    }
    else{
        r = data->mopt.r;
        t = data->mopt.t;
        CudaCheck(cudaMemcpyToSymbol(MOPTION,&data->mopt,sizeof(MultiOptionData)));
        basketOptMonteCarlo<<<data->numBlocks, data->numThreads, numShared>>>(data->RNG,(OptionValue *)(data->d_CallValue));
        cuda_error_check("\Errore nel lancio basketOptMonteCarlo: ","\n");
    }

	//MEMORY CPY: prices per block
	CudaCheck(cudaMemcpy(data->h_CallValue, data->d_CallValue, data->numBlocks * sizeof(OptionValue), cudaMemcpyDeviceToHost));

	// Closing Monte Carlo
	long double sum=0, sum2=0, price, empstd;
    long int nSim = data->numBlocks * data->path;
    for ( i = 0; i < data->numBlocks; i++ ){
    	sum += data->h_CallValue[i].Expected;
	    sum2 += data->h_CallValue[i].Confidence;
	}
	price = exp(-r*t) * (sum/(double)nSim);
    empstd = sqrt((double)((double)nSim * sum2 - sum * sum)/((double)nSim * (double)(nSim - 1)));
    data->callValue.Confidence = 1.96 * empstd / (double)sqrt((double)nSim);
    data->callValue.Expected = price;
}

void cvaMonteCarlo(dev_MonteCarloData *data, float intdef, float lgd){
    /*----------------- SHARED MEMORY -------------------*/
    int i, numShared = sizeof(double) * data->numThreads * 2;
    if( data->numOpt == 1){
         /*--------------- CONSTANT MEMORY ----------------*/
        CudaCheck(cudaMemcpyToSymbol(INTDEF,&intdef,sizeof(float)));
        CudaCheck(cudaMemcpyToSymbol(LGD,&lgd,sizeof(float)));
        CudaCheck(cudaMemcpyToSymbol(OPTION,&data->sopt,sizeof(OptionData)));
        cvaCallOptMC<<<data->numBlocks, data->numThreads, numShared>>>(data->RNG,(OptionValue *)(data->d_CallValue));
        cuda_error_check("\Errore nel lancio cvaCallOptMC: ","\n");
    }
    //MEMORY CPY: prices per block
    CudaCheck(cudaMemcpy(data->h_CallValue, data->d_CallValue, data->numBlocks * sizeof(OptionValue), cudaMemcpyDeviceToHost));
    // Closing Monte Carlo
    long double sum=0, sum2=0, price, empstd;
    long int nSim = data->numBlocks * data->path;
    for ( i = 0; i < data->numBlocks; i++ ){
        sum += data->h_CallValue[i].Expected;
        sum2 += data->h_CallValue[i].Confidence;
    }
    price = sum/(double)nSim;
    empstd = sqrt((double)((double)nSim * sum2 - sum * sum)/((double)nSim * (double)(nSim - 1)));
    data->callValue.Confidence = 1.96 * empstd / (double)sqrt((double)nSim);
    data->callValue.Expected = price;
}

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

extern "C" void dev_cvaEquityOption(CVA *cva, int numBlocks, int numThreads, int sims){
    int i;
    double dt, t;
    dev_MonteCarloData data;
    // Option
    if(cva->ns ==1){
        data.sopt = cva->option;
        dt = cva->option.t / (double)cva->n;
        t = cva->option.t;
    }
    else{
        data.mopt = cva->opt;
        dt = cva->opt.t / (double)cva->n;
        t = cva->opt.t;
    }
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
    double sommaProdotto1=0;
    //double sommaProdotto2=0;
	for( i=1; i < (cva->n+1); i++){
		if((t -= (dt))<0){
			cva->ee[i].Confidence = 0;
			cva->ee[i].Expected = 0;
		}
		else{
            if(cva->ns ==1)
                data.sopt.t = t;
            else
                data.mopt.t = t;
			MonteCarlo(&data);
            //data.callValue.Expected = (data.callValue.Expected + cva->ee[i-1].Expected)/2;
			cva->ee[i] = data.callValue;
		}
        cva->dp[i] = exp(-(dt*i) * cva->defInt) - exp(-(dt*(i+1)) * cva->defInt);
		//cva->fp[i] = exp(-(dt)*(i-1) * cva->credit.fundingspread / 100 / cva->credit.lgd) - exp(-(dt*i) * cva->credit.fundingspread / 100 / cva->credit.lgd );
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


