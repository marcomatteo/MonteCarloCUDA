//
//  MonteCarlo.cu
//  tesi
//
//  Created by Marco Matteo Buzzulini on 27/11/17.
//  Copyright © 2017 Marco Matteo Buzzulini. All rights reserved.
//

#include "MonteCarlo.h"

extern "C" double host_bsCall ( OptionData );
extern "C" void host_cvaEquityOption(CVA *, int);
extern "C" void dev_cvaEquityOption(CVA *, int , int , int );
extern "C" void printOption( OptionData o);
extern "C" void Chol( double c[N][N], double a[N][N] );
extern "C" void printMultiOpt( MultiOptionData *o);
extern "C" double randMinMax(double min, double max);

void getRandomSigma( double* std );
void getRandomRho( double* rho );
void pushVett( double* vet, double x );

void Parameters(int *numBlocks, int *numThreads);
void memAdjust(cudaDeviceProp *deviceProp, int *numThreads);
void sizeAdjust(cudaDeviceProp *deviceProp, int *numBlocks, int *numThreads);

////////////////////////////////////////////////////////////////////////////////////////
//                                      MAIN
////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[]) {
    /*--------------------------- DATA INSTRUCTION -----------------------------------*/
    CVA cva;
    //cva.credit.creditspread=180;
    //cva.credit.fundingspread=75;
    cva.defInt = 0.03;
    cva.lgd = 1 - 0.4;
    cva.n = PATH;
    cva.dp = (double*)malloc((cva.n+1)*sizeof(double));
    //cva.fp = (double*)malloc((cva.n+1)*sizeof(double));
    
    // Vector of EE, dim n+1 because starts in 0
    cva.ee = (OptionValue *)malloc(sizeof(OptionValue)*(cva.n+1));
    int numBlocks, numThreads, i, j, SIMS;
    double difference, dt, cholRho[N][N],
    *bs_price = (double*)malloc(sizeof(double)*(cva.n+1));
    cudaEvent_t d_start, d_stop;
    // Option Data
    MultiOptionData opt;
    if(N>1){
        double dw = (double)1 / N;
        //    Volatility
        opt.v[0] = 0.2;
        opt.v[1] = 0.3;
        opt.v[2] = 0.2;
        //    Spot prices
        opt.s[0] = 100;
        opt.s[1] = 100;
        opt.s[2] = 100;
        //    Weights
        opt.w[0] = dw;
        opt.w[1] = dw;
        opt.w[2] = dw;
        //    Correlations
        opt.p[0][0] = 1;
        opt.p[0][1] = -0.5;
        opt.p[0][2] = -0.5;
        opt.p[1][0] = -0.5;
        opt.p[1][1] = 1;
        opt.p[1][2] = -0.5;
        opt.p[2][0] = -0.5;
        opt.p[2][1] = -0.5;
        opt.p[2][2] = 1;
        //    Drift vectors for the brownians
        opt.d[0] = 0;
        opt.d[1] = 0;
        opt.d[2] = 0;
        
        if(N!=3){
            getRandomSigma(opt.v);
            getRandomRho(&opt.p[0][0]);
            pushVett(opt.s,100);
            pushVett(opt.w,dw);
            pushVett(opt.d,0);
        }
    }
    else{
        opt.v[0] = 0.2;
        opt.s[0] = 100;
        opt.w[0] = 1;
        opt.d[0] = 0;
        opt.p[0][0] = 1;
    }
    
    opt.k= 100.f;
    opt.r= 0.05;
    opt.t= 1.f;
    cva.opt = opt;
	
    float GPU_timeSpent=0, CPU_timeSpent=0, speedup;
    
    printf("Expected Exposures of an Equity Option\n");
    //  CUDA parameters
    Parameters(&numBlocks, &numThreads);
    printf("Inserisci il numero di simulazioni Monte Carlo(x131072): ");
    scanf("%d",&SIMS);
    SIMS *= 131072;
    printf("\nScenari di Monte Carlo: %d\n",SIMS);

	//	Print Option details
    printMultiOpt(&opt);
    
    if(N>1){
        //    Cholevski factorization
        Chol(opt.p, cholRho);
        for(i=0;i<N;i++)
            for(j=0;j<N;j++)
                cva.opt.p[i][j]=cholRho[i][j];
    }else{
        OptionData option;
        option.v = opt.v[0];
        option.s = opt.s[0];
        option.k = opt.k;
        option.r = opt.r;
        option.t = opt.t;
        bs_price[0] = host_bsCall(option);
        for(i=1;i<cva.n+1;i++){
            if((opt.t -= dt)<0)
                bs_price[i] = 0;
            else
                bs_price[i] = host_bsCall(option);
        }
    }

	// Timer init
    CudaCheck( cudaEventCreate( &d_start ));
    CudaCheck( cudaEventCreate( &d_stop ));

    //	Black & Scholes price
    dt = opt.t/(double)cva.n;
    

    //	Original Time to mat
    opt.t= 1.f;
    
    // CPU Monte Carlo
    printf("\nCVA execution on CPU:\n");
    CudaCheck( cudaEventRecord( d_start, 0 ));
    host_cvaEquityOption(&cva, SIMS);
    CudaCheck( cudaEventRecord( d_stop, 0));
    CudaCheck( cudaEventSynchronize( d_stop ));
    CudaCheck( cudaEventElapsedTime( &CPU_timeSpent, d_start, d_stop ));
    CPU_timeSpent /= 1000;
    printf("\nPrezzi Simulati:\n");
 
        printf("|\ti\t\t|\tPrezzi BS\t| Differenza Prezzi\t|\tPrezzi\t\t|\tDefault Prob\t|\n");
        for(i=0;i<cva.n+1;i++){
            difference = abs(cva.ee[i].Expected - bs_price[i]);
            printf("|\t%f\t|\t%f\t|\t%f\t|\t%f\t|\t%f\t|\n",dt*i,bs_price[i],difference,cva.ee[i].Expected,cva.dp[i]);
        }
    
    // printf("\nCVA: %f\nFVA: %f\nTotal: %f\n\n",cva.cva,cva.fva,(cva.cva+cva.fva));
    printf("\nCVA: %f\n\n",cva.cva);

    printf("\nTotal execution time: %f s\n\n", CPU_timeSpent);
    printf("--------------------------------------------------\n");
    // GPU Monte Carlo
    printf("\nCVA execution on GPU:\n");
    CudaCheck( cudaEventRecord( d_start, 0 ));
    dev_cvaEquityOption(&cva, numBlocks, numThreads, SIMS);
    CudaCheck( cudaEventRecord( d_stop, 0));
    CudaCheck( cudaEventSynchronize( d_stop ));
    CudaCheck( cudaEventElapsedTime( &GPU_timeSpent, d_start, d_stop ));
    GPU_timeSpent /= 1000;
    speedup = CPU_timeSpent/GPU_timeSpent;
    printf("\nTotal execution time: %f s\n", GPU_timeSpent);
    printf("SpeedUp: %f\n\n", speedup);

    printf("\nPrezzi Simulati:\n");
   	printf("|\ti\t\t|\tPrezzi BS\t| Differenza Prezzi\t|\tPrezzi\t\t|\tDefault Prob\t|\n");
   	for(i=0;i<cva.n+1;i++){
   		difference = abs(cva.ee[i].Expected - bs_price[i]);
   		printf("|\t%f\t|\t%f\t|\t%f\t|\t%f\t|\t%f\t|\n",dt*i,bs_price[i],difference,cva.ee[i].Expected,cva.dp[i]);
   	}
   	//printf("\nCVA: %f\nFVA: %f\nTotal: %f\n\n",cva.cva,cva.fva,(cva.cva+cva.fva));
    printf("\nCVA: %f\n\n",cva.cva);

   	free(cva.dp);
   	// free(cva.fp);
   	free(cva.ee);
   	free(bs_price);
    return 0;
}

//Simulation std, rho and covariance matrix
void getRandomSigma( double* std ){
    int i,j;
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
            float r;
            if(i==j)
                r=1;
            else
                if(j%2==0)
                    r = 0.5;
                else
                    r= -0.5;
            // r=randMinMax(-1, 1);
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

///////////////////////////////////
//    ADJUST FUNCTIONS
///////////////////////////////////

void sizeAdjust(cudaDeviceProp *deviceProp, int *numBlocks, int *numThreads){
    int maxGridSize = deviceProp->maxGridSize[0];
    int maxBlockSize = deviceProp->maxThreadsPerBlock;
    //    Replacing in case of wrong size
    if(*numBlocks > maxGridSize){
        *numBlocks = maxGridSize;
        printf("Warning: maximum size of Grid is %d",*numBlocks);
    }
    if(*numThreads > maxBlockSize){
        *numThreads = maxBlockSize;
        printf("Warning: maximum size of Blocks is %d",*numThreads);
    }
}

void memAdjust(cudaDeviceProp *deviceProp, int *numThreads){
    size_t maxShared = deviceProp->sharedMemPerBlock;
    size_t maxConstant = deviceProp->totalConstMem;
    int sizeDouble = sizeof(double);
    int numShared = sizeDouble * *numThreads * 2;
    if(sizeof(MultiOptionData) > maxConstant){
        printf("\nWarning: Excess use of constant memory: %zu\n",maxConstant);
        printf("A double variable size is: %d\n",sizeDouble);
        printf("In a MultiOptionData struct there's a consumption of %zu constant memory\n",sizeof(MultiOptionData));
        printf("In this Basket Option there's %d stocks\n",N);
        int maxDim = (int)maxConstant/(sizeDouble*5);
        printf("The optimal number of dims should be: %d stocks\n",maxDim);
    }
    if(numShared > maxShared){
        printf("\nWarning: Excess use of shared memory: %zu\n",maxShared);
        printf("A double variable size is: %d\n",sizeDouble);
        int maxThreads = (int)maxShared / (2*sizeDouble);
        printf("The optimal number of thread should be: %d\n",maxThreads);
    }
    printf("\n");
}

void Parameters(int *numBlocks, int *numThreads){
    cudaDeviceProp deviceProp;
    CudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    *numThreads = NTHREADS;
    *numBlocks = BLOCKS;
    sizeAdjust(&deviceProp,numBlocks, numThreads);
    memAdjust(&deviceProp, numThreads);
}
