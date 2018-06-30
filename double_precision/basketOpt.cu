/*
 *  basketOpt.cu
 *  Monte Carlo methods in CUDA
 *  Dissertation project
 *  Created on: 06/feb/2018
 *  Author: Marco Matteo Buzzulini
 *  Copyright © 2018 Marco Matteo Buzzulini. All rights reserved.
 */

#include "MonteCarlo.h"

#define NTHREADS 1
#define THREADS 128
#define BLOCKS 512
#define SIMPB 131072

extern "C" OptionValue host_basketOpt(MultiOptionData*, int);
extern "C" OptionValue dev_basketOpt(MultiOptionData *, int, int,int);
extern "C" void Chol( double c[N][N], double a[N][N] );
extern "C" void printMultiOpt( MultiOptionData *o);
extern "C" double randMinMax(double min, double max);

void getRandomSigma( double* std );
void getRandomRho( double* rho );
void pushVett( double* vet, double x );

int main(int argc, const char * argv[]) {
    /*--------------------------- VARIABLES -----------------------------------*/
	double dw = (double)1 / N;

	// Option Data
	MultiOptionData option;
	//	Volatility
	option.v[0] = 0.2;
	option.v[1] = 0.3;
	option.v[2] = 0.2;
	//	Spot prices
	option.s[0] = 100;
	option.s[1] = 100;
	option.s[2] = 100;
	//	Weights
	option.w[0] = dw;
	option.w[1] = dw;
	option.w[2] = dw;
	//	Correlations
	option.p[0][0] = 1;
			option.p[0][1] = -0.5;
					option.p[0][2] = -0.5;
	option.p[1][0] = -0.5;
			option.p[1][1] = 1;
					option.p[1][2] = -0.5;
	option.p[2][0] = -0.5;
			option.p[2][1] = -0.5;
					option.p[2][2] = 1;
	//	Drift vectors for the brownians
	option.d[0] = 0;
	option.d[1] = 0;
	option.d[2] = 0;
	option.k= 100.f;
	option.r= 0.048790164;
	option.t= 1.f;
	if(N!=3){
		getRandomSigma(option.v);
		getRandomRho(&option.p[0][0]);
		pushVett(option.s,100);
		pushVett(option.w,dw);
		pushVett(option.d,0);
	}

	// Simulation variables
	int numBlocks, numThreads[NTHREADS], SIMS, i, j;
	OptionValue CPU_sim, GPU_sim[NTHREADS];
	float CPU_timeSpent=0, GPU_timeSpent[NTHREADS], speedup[NTHREADS];
	double cholRho[N][N], difference[NTHREADS];
	// Timer
	cudaEvent_t d_start, d_stop;

	/*--------------------- START PROGRAM ------------------------------*/
	printf("Basket Option Pricing\n");
	//	CUDA parameters for parallel execution
    numBlocks = BLOCKS;
    numThreads[0] = THREADS;
    //numThreads[1] = 128;
    //numThreads[2] = 1024;
    //numThreads[3] = 512;
    printf("Inserisci il numero simulazioni (x131.072): ");
    scanf("%d",&SIMS);
    SIMS *= SIMPB;
	printf("\nScenari di Monte Carlo: %d\n",SIMS);
	//	Print Option details
	if(N<7)
		printMultiOpt(&option);
	else
		printf("\nBasket Option con %d sottostanti\n",N);
    //	Cholevski factorization
    Chol(option.p, cholRho);
    for(i=0;i<N;i++)
    	for(j=0;j<N;j++)
           	option.p[i][j]=cholRho[i][j];
    // Timer init
    CudaCheck( cudaEventCreate( &d_start ));
    CudaCheck( cudaEventCreate( &d_stop ));
    /* CPU Monte Carlo */
    printf("\nMonte Carlo execution on CPU...\n");
    CudaCheck( cudaEventRecord( d_start, 0 ));
    CPU_sim=host_basketOpt(&option, SIMS);
    CudaCheck( cudaEventRecord( d_stop, 0));
    CudaCheck( cudaEventSynchronize( d_stop ));
    CudaCheck( cudaEventElapsedTime( &CPU_timeSpent, d_start, d_stop ));
    //CPU_timeSpent /= 1000;

    // GPU Monte Carlo
    printf("\nMonte Carlo execution on GPU...\n");
    for(i=0; i<NTHREADS; i++){
        printf("Monte Carlo for (%d,%d) x %d simulations per thread\n", BLOCKS, numThreads[i], SIMS/BLOCKS/numThreads[i]);
    	CudaCheck( cudaEventRecord( d_start, 0 ));
       	GPU_sim[i] = dev_basketOpt(&option, numBlocks, numThreads[i], SIMS);
        CudaCheck( cudaEventRecord( d_stop, 0));
        CudaCheck( cudaEventSynchronize( d_stop ));
        CudaCheck( cudaEventElapsedTime( &GPU_timeSpent[i], d_start, d_stop ));
        //GPU_timeSpent[i] /= 1000;
        difference[i] = abs(GPU_sim[i].Expected - CPU_sim.Expected);
        speedup[i] = abs(CPU_timeSpent / GPU_timeSpent[i]);
        printf("\n");
    }
    // Comparing time spent with the two methods
    printf( "\n-\tResults:\t-\n");
    printf("Simulated price for the option with CPU: Expected price, I.C., time\n%f \n%f \n%.1f \n", CPU_sim.Expected, CPU_sim.Confidence, CPU_timeSpent);
    printf("Simulated price for the option with GPU:\n");
    printf("  : NumThreads : Price : Confidence Interval : Difference from BS price :  Time : Speedup :");
    printf("\n");
    for(i=0; i<NTHREADS; i++){
        printf("%d \n",numThreads[i]);
        printf("%f \n",GPU_sim[i].Expected);
        printf("%f \n",GPU_sim[i].Confidence);
        printf("%f \n",difference[i]);
        printf("%.1f \n",GPU_timeSpent[i]);
        printf("%.2f \n",speedup[i]);
        printf("---\n");
    }
    CudaCheck( cudaEventDestroy( d_start ));
    CudaCheck( cudaEventDestroy( d_stop ));
    return 0;
}

//Simulation std, rho and covariance matrix
void getRandomSigma( double* std ){
    int i,j=0;
    for(i=0;i<N;i++){
        if(j==0){
            std[i]=0.3;
            j=1;
        }
        else{
            std[i]=0.2;
            j=0;
        }
    }
}
void getRandomRho( double* rho ){
    int i,j;
    //creating the vectors of rhos
    for(i=0;i<N;i++){
        for(j=i;j<N;j++){
            double r;
            if(i==j)
                r=1;
            else
                if(j%2==0)
                    r = 0.5;
                else
                    r= -0.5;
            rho[j+i*N] = r;
            rho[i+j*N] = r;
        }
    }
}
void pushVett( double* vet, double x ){
    int i;
    for(i=0;i<N;i++)
        vet[i] = x;
}
