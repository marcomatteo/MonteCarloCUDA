//
//  MonteCarlo.cu
//  tesi
//
//  Created by Marco Matteo Buzzulini on 27/11/17.
//  Copyright © 2017 Marco Matteo Buzzulini. All rights reserved.
//

#include "MonteCarlo.h"

//	Host utility functions declarations
double* Chol( double *c, int n);

///////////////////////////////////
//	PRINT FUNCTIONS
///////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////////////
//                                      MAIN
////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[]) {
    /*--------------------------- DATA INSTRUCTION -----------------------------------*/
	const double
	    v[N]={ 0.2, 0.3, 0.2 },
	    s[N]={ 100, 100, 100 },
	    w[N]={ dw, dw, dw },
	    p[N][N]={
	        {   1,      -0.5,   -0.5  },
	        {   -0.5,   1,      -0.5  },
	        {   -0.5,    -0.5,    1   }
	    },
	    d[N]={0,0,0};
		K = 100.f;
		R = 0.048790164;
		T = 1.f;
		dw = (double)1/(double)N;
        
    /*--------------------------- CPU PATHULATION -----------------------------------*/
    int SIMS = MAX_BLOCKS*PATH;
    
    /*--------------------------------- MAIN ---------------------------------------*/
    MultiOptionData option;
    OptionValue CPU_sim, GPU_sim;
    
    float CPU_timeSpent, GPU_timeSpent, speedup;
    double price;
    clock_t h_start, h_stop;
    cudaEvent_t d_start, d_stop;
    CudaCheck( cudaEventCreate( &d_start ));
    CudaCheck( cudaEventCreate( &d_stop ));
    
    //	Setting up the option
    option.s = s;
    option.v = v;
    option.p = &p[0][0];
    option.d = d;
    option.w = w;
    option.k = K;
    option.r = R;
    option.t = T;
    option.n = N;
    //	Print Option details
    printMultiOpt(&option);

    //	Cholevski factorization
    option.p = Chol(&p[0][0], option.n);

    // CPU Monte Carlo
    printf("\nMonte Carlo execution on CPU:\nN^ simulations: %d\n\n",SIMS);
    h_start = clock();
    CPU_sim=CPUBasketOptCall(&option, SIMS);
    h_stop = clock();
    CPU_timeSpent = ((float)(h_stop - h_start)) / CLOCKS_PER_SEC;
    
    price = CPU_sim.Expected;
    printf("Simulated price for the basket option: € %f with I.C [ %f;%f ]\n", price, price - CPU_sim.Confidence, price + CPU_sim.Confidence);
    printf("Total execution time: %f s\n\n", CPU_timeSpent);
    
    // GPU Monte Carlo
    printf("\nMonte Carlo execution on GPU:\nN^ simulations: %d\n",SIMS);
    CudaCheck( cudaEventRecord( d_start, 0 ));
    GPUBasketOpt(&option, &GPU_sim);
    CudaCheck( cudaEventRecord( d_stop, 0));
    CudaCheck( cudaEventSynchronize( d_stop ));
    CudaCheck( cudaEventElapsedTime( &GPU_timeSpent, d_start, d_stop ));
    GPU_timeSpent /= CLOCKS_PER_SEC;
    
    price = GPU_sim.Expected;
    printf("Simulated price for the basket option: € %f with I.C [ %f;%f ]\n", price, price-GPU_sim.Confidence, price + GPU_sim.Confidence);
    printf("Total execution time: %f s\n\n", GPU_timeSpent);
    
    // Comparing time spent with the two methods
    printf( "-\tComparing results:\t-\n");
    speedup = abs(CPU_timeSpent / GPU_timeSpent);
    printf( "The GPU's speedup: %.2f \n", speedup);
    return 0;
}


double* Chol( double *c, int n ){
    int i,j,k;
    double *a=(double*)malloc(n*n*sizeof(double));
    double v[n];
    for( i=0; i<n; i++){
        for( j=0; j<n; j++ ){
            if( j>=i ){
                //Triangolare inferiore
            	//v[j]=c[j][i]
                v[j] = c[i+j*n];
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
