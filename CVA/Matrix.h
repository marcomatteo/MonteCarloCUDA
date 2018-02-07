/*
 * Matrix.h
 *
 *  Created on: 06/feb/2018
 *      Author: marco
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Matrix structure
typedef struct{
    // the matrix is converted to a 1-dimensional vector
    // m[i][j] = m[j+i*cols] for i<rows and j<cols
    double *data;
    int rows;
    int cols;
} Matrix;

/////////////////////////////////////////////////////////////
// Matrix Functions
/////////////////////////////////////////////////////////////
void printVect( double *vet, int l );
void printMat( double *mat, int r, int c );
void mat_print( Matrix *mat );

void mat_init( Matrix *mat, int rows, int cols);
void mat_free( Matrix *mat );
void mat_setData( double *data, Matrix *mat );

void mat_prod( Matrix *first, Matrix *second, Matrix *result);
void ProdMat( double *first, double *second, int rows, int n, int cols, double *result);


#endif /* MATRIX_H_ */
