//
//  MonteCarlo.cu
//  tesi
//
//  Created by Marco Matteo Buzzulini on 27/11/17.
//  Copyright © 2017 Marco Matteo Buzzulini. All rights reserved.
//

#include "MonteCarlo.h"

/////////////////////////////////////////////////////
//////////  KERNEL FUNCTIONS
//////////////////////////////////////////////////////

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
        /*	Price computations for each block
        int nSim = SIM;
	sum.Expected = exp(-(D_R*D_T)) * (s_Sum[0]/(double)nSim);
        sum.Confidence = sqrt((double)((double)nSim * s_Sum2[0] - s_Sum[0] * s_Sum[0])
                         /((double)nSim * (double)(nSim - 1)));
        d_CallValue[blockIndex].Confidence = 1.96 * (double)sum.Confidence / (double)sqrt((double)nSim);
        d_CallValue[blockIndex].Expected = sum.Expected;*/
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

////////////////////////////////////////////////////////////////////////////////////////
//                                      MAIN
////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[]) {
    /*--------------------------- DATA INSTRUCTION -----------------------------------*/
    const double K = 100.f;
    const double R = 0.048790164;
    const double T = 1.f;
    double dw = (double)1/(double)N;
        
    /*--------------------------- SIMULATION VARs -----------------------------------*/
    int SIMS = MAX_BLOCKS*SIM;
    
    /*--------------------------- PREPARATION -----------------------------------*/
    // Static
    double v[N]={ 0.2, 0.3, 0.2 }, s[N]={100, 100, 100 }, w[N]={dw,dw,dw},
    p[N][N]={
        {   1,      -0.5,   -0.5  },
        {   -0.5,   1,      -0.5  },
        {   -0.5,    -0.5,    1   }
    }, d[N]={0,0,0};
    
    double *st,*randRho,*randV,*wp,*drift;
    int i;
    // Dinamic
    srand((unsigned)time(NULL));
    if(RAND==1){
        printf("\n\t-\tExecution mode: RANDOM\t-\n\n");
	st=(double*)malloc(N*sizeof(double));
        wp=(double*)malloc(N*sizeof(double));
        drift=(double*)malloc(N*sizeof(double));
        for(i=0;i<N;i++){
            st[i]=randMinMax(K-10, K+10);
            wp[i]=dw;
            drift[i]=0;
        }
        randRho = getRandomRho(N);
        randV = getRandomSigma(N);
    }
    else{
	printf("\n\t-\tExecution mode: GIVEN DATA\t-\n\n");
        st=s;
        randRho=&p[0][0];
        randV=v;
        wp=w;
        drift=d;
    }
    
    /*--------------------------------- MAIN ---------------------------------------*/
    MultiOptionData option;
    OptionValue CPU_sim, GPU_sim;
    
    float CPU_timeSpent, GPU_timeSpent, speedup;
    double price;
    clock_t start, stop;
    
    Matrix cov;
    //Init cov matrix for the gaussian vect
    cov.cols = N; cov.rows = N;
    //cov.data=getCovMat(randV, randRho, N);
    cov.data=randRho;

    option.s = st;
    option.v = randV;
    option.p = randRho;
    option.d = drift;
    option.w = wp;
    option.k = K;
    option.r = R;
    option.t = T;
    option.n = N;
    
    printMultiOpt(&option);
    //Substitute option data with cholevski correlation matrix
    option.p = Chol(&cov);

    // CPU Monte Carlo
    printf("\nMonte Carlo execution on CPU:\nN^ simulations: %d\n\n",SIMS);
    start = clock();
    CPU_sim=CPUBasketOptCall(&option, SIMS);
    stop = clock();
    CPU_timeSpent = ((float)(stop - start)) / CLOCKS_PER_SEC;
    
    price = CPU_sim.Expected;
    printf("Simulated price for the basket option: € %f with I.C [ %f;%f ]\n", price, price - CPU_sim.Confidence, price + CPU_sim.Confidence);
    printf("Total execution time: %f s\n\n", CPU_timeSpent);
    
    // GPU Monte Carlo
    printf("\nMonte Carlo execution on GPU:\nN^ simulations: %d\n",SIMS);
    start = clock();
    GPUBasketOpt(&option, &GPU_sim);
    stop = clock();
    GPU_timeSpent = ((float)(stop - start)) / CLOCKS_PER_SEC;
    
    price = GPU_sim.Expected;
    printf("Simulated price for the basket option: € %f with I.C [ %f;%f ]\n", price, price-GPU_sim.Confidence, price + GPU_sim.Confidence);
    printf("Total execution time: %f s\n\n", GPU_timeSpent);
    
    // Comparing time spent with the two methods
    printf( "-\tComparing results:\t-\n");
    speedup = abs(CPU_timeSpent / GPU_timeSpent);
    printf( "The GPU's speedup: %.2f \n", speedup);
    //mat_free(&cov);
    if(RAND==1){
        free(st);
        free(randV);
        free(randRho);
        free(wp);
        free(drift);
    }
    return 0;
}


void GPUBasketOpt(MultiOptionData *option, OptionValue *callValue ){
    int i; 
    /*----------------- HOST MEMORY -------------------*/
    OptionValue *h_CallValue;
    //Allocation pinned host memory for prices
    HANDLE_ERROR(cudaHostAlloc(&h_CallValue, sizeof(OptionValue)*(MAX_BLOCKS),cudaHostAllocDefault));
    
    /*--------------- CONSTANT MEMORY ----------------*/
    
    HANDLE_ERROR(cudaMemcpyToSymbol(D_DRIFTVECT,option->d,N*sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(D_CHOLMAT,option->p,N*N*sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(D_S,option->s,N*sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(D_V,option->v,N*sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(D_W,option->w,N*sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(D_K,&option->k,sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(D_T,&option->t,sizeof(double)));
    HANDLE_ERROR(cudaMemcpyToSymbol(D_R,&option->r,sizeof(double)));
    
    /*----------------- DEVICE MEMORY -------------------*/
    OptionValue *d_CallValue;
    HANDLE_ERROR(cudaMalloc(&d_CallValue, sizeof(OptionValue)*(MAX_BLOCKS)));
    
    /*------------ RNGs and TIME VARIABLES --------------*/
    curandState *RNG;
    cudaEvent_t start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ));
    HANDLE_ERROR( cudaEventCreate( &stop ));
    float time;
    
    // RANDOM NUMBER GENERATION KERNEL
    //Allocate states for pseudo random number generators
    HANDLE_ERROR(cudaMalloc((void **) &RNG, MAX_BLOCKS * MAX_THREADS * sizeof(curandState)));
    //Setup for the random number sequence
    randomSetup<<<MAX_BLOCKS, MAX_THREADS>>>(RNG);
    
    //MONTE CARLO KERNEL
    HANDLE_ERROR( cudaEventRecord( start, 0 ));
    MultiMCBasketOptKernel<<<MAX_BLOCKS, MAX_THREADS>>>(RNG,(OptionValue *)(d_CallValue));
    HANDLE_ERROR( cudaEventRecord( stop, 0));
    HANDLE_ERROR( cudaEventSynchronize( stop ));
    HANDLE_ERROR( cudaEventElapsedTime( &time, start, stop ));
    printf( "\nMonte Carlo simulations done in %f milliseconds\n", time);
    HANDLE_ERROR( cudaEventDestroy( start ));
    HANDLE_ERROR( cudaEventDestroy( stop ));
    
    //MEMORY CPY: prices per block
    HANDLE_ERROR(cudaMemcpy(h_CallValue, d_CallValue, MAX_BLOCKS * sizeof(OptionValue), cudaMemcpyDeviceToHost));
    
    // Closing Monte Carlo
    long double sum=0, sum2=0, price, empstd;
    long int nSim = MAX_BLOCKS * SIM;
    for ( i = 0; i < MAX_BLOCKS; i++ ){
        sum += h_CallValue[i].Expected;
        sum2 += h_CallValue[i].Confidence;
    }
    /*callValue->Expected = sum/(double)MAX_BLOCKS;
    callValue->Confidence = sum2/(double)MAX_BLOCKS;*/
    price = exp(-(option->r*option->t)) * (sum/(double)nSim);
    empstd = sqrt((double)((double)nSim * sum2 - sum * sum)
                         /((double)nSim * (double)(nSim - 1)));
    callValue->Confidence = 1.96 * empstd / (double)sqrt((double)nSim);
    callValue->Expected = price;
    
    //Free memory space
    HANDLE_ERROR(cudaFree(RNG));
    HANDLE_ERROR(cudaFreeHost(h_CallValue));
    HANDLE_ERROR(cudaFree(d_CallValue));
}

/////////////////////////////////////////////////////
//////////  PRINT FUNCTIONS
//////////////////////////////////////////////////////

void printVect( double *mat, int c ){
    int i,j,r=1;
    for(i=0; i<r; i++){
        printf("\n!\t");
        for(j=0; j<c; j++){
            printf("\t%f\t",mat[j+i*c]);
        }
        printf("\t!");
    }
    printf("\n\n");
}

void printMat( double *mat, int r, int c ){
    int i,j;
    for(i=0; i<r; i++){
        printf("\n!\t");
        for(j=0; j<c; j++){
            printf("\t%f\t",mat[j+i*c]);
        }
        printf("\t!");
    }
    printf("\n\n");
}

void mat_print( Matrix *mat ){
    printMat(mat->data, mat->rows, mat->cols);
}

void printOption( OptionData o){
    printf("\n-\tOption data\t-\n\n");
    printf("Underlying asset price:\t € %.2f\n", o.s);
    printf("Strike price:\t\t € %.2f\n", o.k);
    printf("Risk free interest rate: %.2f %%\n", o.r * 100);
    printf("Volatility:\t\t\t %.2f %%\n", o.v * 100);
    printf("Time to maturity:\t\t %.2f %s\n", o.t, (o.t>1)?("years"):("year"));
}

void printMultiOpt( MultiOptionData *o){
    int n=o->n;
    printf("\n-\tBasket Option data\t-\n\n");
    printf("Number of assets: %d\n",n);
    printf("Underlying assets prices:\n");
    printVect(o->s, n);
    printf("Volatility:\n");
    printVect(o->v, n);
    printf("Weights:");
    printVect(o->w, n);
    printf("Correlation matrix:\n");
    printMat(o->p, n, n);
    printf("Strike price:\t € %.2f\n", o->k);
    printf("Risk free interest rate %.2f \n", o->r);
    printf("Time to maturity:\t %.2f %s\n", o->t, (o->t>1)?("years"):("year"));
}
//////////////////////////////////////////////////////
//////////   MATRIX FUNCTIONS
//////////////////////////////////////////////////////

void mat_init( Matrix *mat, int rows, int cols){
    mat->data = (double*)malloc(rows*cols*sizeof(double));
    mat->cols = cols;
    mat->rows = rows;
}

void mat_free( Matrix *mat ){
    if(mat->data!=NULL) free( mat->data );
}

//Copia contenuto in una nuova matrice quadrata
void mat_setData( double* data, Matrix *mat ){
    int i,j, rows=mat->rows, cols=mat->cols;
    for(i=0;i<rows;i++)
        for(j=0;j<cols;j++)
            mat->data[j+i*cols] = data[j+i*cols];
}

// Restituisce il prodotto tra matrici
void mat_prod( Matrix *first, Matrix *second, Matrix *result ){
    int n = first->cols;
    if(n != second->rows){
        printf("Non si può effettuare la moltiplicazione\n");
        return;
    }
    double somma;
    int i,j,k;
    result->rows = first->rows;
    result->cols = second->cols;
    for(i=0;i<result->rows;i++){
        for(j=0;j<result->cols;j++){
            somma = 0;
            for(k=0;k<n;k++)
                //somma += first->data[i][k]*second->data[k][j];
                somma += first->data[k+i*first->cols] * second->data[j+k*second->cols];
            //result->data[i][j] = somma;
            result->data[j+i*result->cols] = somma;
        }
    }
}

void ProdMat( double *first, double *second, int rows, int n, int cols, double *result ){
    double somma;
    int i,j,k;
    for(i=0;i<rows;i++){
        for(j=0;j<cols;j++){
            somma = 0;
            for(k=0;k<n;k++)
                //somma += first->data[i][k]*second->data[k][j];
                somma += first[k+i*cols] * second[j+k*cols];
            //result->data[i][j] = somma;
            result[j+i*cols] = somma;
        }
    }
}

double* Chol( Matrix *c ){
    if(c->rows != c->cols){
        printf("Cholevski necessita di una matrice quadrata\n");
        return NULL;
    }
    int i,j,k, n = c->rows;
    double *a=(double*)malloc(n*n*sizeof(double));
    double v[n];
    for( i=0; i<n; i++){
        for( j=0; j<n; j++ ){
            if( j>=i ){
                //Triangolare inferiore
                v[j] = c->data[i+j*n];  //v[j]=c[j][i]
                for(k=0; k<i; k++)    //Scorre tutta
                    //v[j] = v[j] - a[i][k] * a[j][k]
                    v[j] = v[j]-(a[k+i*n] * a[k+j*n]);
                //a[j][i] = v[j] / sqrt( v[i] )
                if(v[i]>0)
                    a[i+j*n] = v[j]/sqrt( v[i] );
                else
                    a[i+j*n] = 0.0f;
            }
            else
                //Triangolare superiore a[j][i]
                a[i+j*n] = 0.0f;
        }
    }
    return a;
}

//////////////////////////////////////////////////////
//////////   FINANCE FUNCTIONS
//////////////////////////////////////////////////////

double randMinMax(double min, double max){
    double x=(double)rand()/(double)(RAND_MAX);
    return max*x+(1.0f-x)*min;
}

//Generator of a normal pseudo-random number with mean mu and volatility sigma
double gaussian( double mu, double sigma ){
    double x = randMinMax(0, 1);
    double y = randMinMax(0, 1);
    return mu + sigma*(sqrt( -2.0 * log(x) ) * cos( 2.0 * M_PI * y ));
}

//Simulation std, rho and covariance matrix
double* getRandomSigma( int n ){
    Matrix std;
    //init the vectors of std
    mat_init(&std, 1, n);
    //creating the vectors of stds
    int i;
    for(i=0;i<n;i++)
        std.data[i] = randMinMax(0, 1);
    //printf("Stampo per debug vettore sigma:\n");
    //mat_print(&std);
    return std.data;
}
double* getRandomRho( int n ){
    Matrix rho;
    //init the vectors of rhos
    mat_init(&rho, n, n);
    int i,j;
    //creating the vectors of rhos
    for(i=0;i<n;i++){
        for(j=i;j<n;j++){
            double r;
            if(i==j)
                r=1;
            else
                r=randMinMax(-1, 1);
            rho.data[j+i*n] = r;
            rho.data[i+j*n] = r;
        }
    }
    //printf("Stampo per debug matrice rho:\n");
    //mat_print(&rho);
    return rho.data;
}

double* getCovMat(double *std, double *rho, int n){
    Matrix sigma;
    int i,j;
    //init the variance-covariance matrix
    mat_init(&sigma, n, n);
    //creating variance-covariance matrix
    for(i=0;i<n;i++)
        for(j=i;j<n;j++){
            double val = std[i]*std[j]*rho[j+i*n];
            sigma.data[j+i*n] = val;
            sigma.data[i+j*n] = val;
        }
    return sigma.data;
}
//----------------------------------------------------
//Simulation of a Gaussian vector X~N(m,∑)
//Step 1: Compute the square root of the matrix ∑, generate lower triangular matrix A
//Step 2: Simulate n indipendent standard random variables G ~ N(0,1)
//Step 3: Return m + A*G
void simGaussVect(double *drift, double *volatility, int n, double *result){
    int i;
    double *g=NULL;
    g=(double*)malloc(n*sizeof(double));
    //RNGs
    for(i=0;i<n;i++)
        g[i]=gaussian(0, 1);
    Matrix gauss, r, vol;
    gauss.rows = n;     r.rows=n;       vol.cols = n;
    gauss.cols = 1;     r.cols=1;       vol.rows = n;
    gauss.data = &g[0]; r.data=result;  vol.data = volatility;
    //A*G
    mat_prod(&vol,&gauss,&r);
    //X=m+A*G
    for(i=0;i<n;i++){
        r.data[i] += drift[i];
    }
    free(g);
}

void multiStockValue(double *s, double *v, double *g, double t, double r, int n, double *values){
    int i;
    for(i=0;i<n;i++){
        double mu = (r - 0.5 * v[i] * v[i])*t;
        double si = v[i] * g[i] * sqrt(t);
        values[i] = s[i] * exp(mu+si);
    }
}

OptionValue CPUBasketOptCall(MultiOptionData *option, int sim){
    int i,j, n = option->n;
    
    //Monte Carlo algorithm
    OptionValue callValue;
    double st_sum=0.0f, sum=0.0f, var_sum=0.0f, price, emp_stdev;
    double k = option->k;
    double t = option->t;
    double r = option->r;
    
    //vectors of brownian and ST
    double bt[N];
    double s[N];
    
    for(i=0; i<sim; i++){
        st_sum = 0;
        //Simulation of stock prices
        simGaussVect(option->d, option->p, n, bt);
        multiStockValue(option->s, option->v, bt, t, r, n, s);
        for(j=0;j<n;j++)
            st_sum += s[j]*option->w[j];
        price = st_sum - k;
        if(price<0)
            price = 0.0f;
        sum += price;
        var_sum += price*price;
    }
    emp_stdev = sqrt(((double)sim * var_sum - sum * sum)/((double)sim * (double)(sim - 1)));
    
    callValue.Expected = exp(-r*t) * (sum/(double)sim);
    callValue.Confidence = 1.96 * emp_stdev/sqrt(sim);
    return callValue;
}
