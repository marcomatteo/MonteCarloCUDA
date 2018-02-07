/*
 * Options.c
 *
 *  Created on: 06/feb/2018
 *      Author: marco
 */

#include "Matrix.h"

//////////////////////////////////////////////////////
//////////   MATRIX FUNCTIONS
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
