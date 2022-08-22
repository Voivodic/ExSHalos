#include "sampler_h.h"

/*Invert a given matrix using GSL*/
gsl_matrix *invert_matrix(gsl_matrix *matrix, int size){
    int i, j;

    /*Alloc the inverse matrix*/
    gsl_matrix *inv = gsl_matrix_alloc(size, size);
    for(i=0;i<size;i++)
        for(j=0;j<size;j++)
            gsl_matrix_set(inv, i, j, gsl_matrix_get(matrix, i, j));

    /*Invert the matrix using the cholesky decomposition*/
    gsl_linalg_cholesky_decomp1(inv);
    gsl_linalg_cholesky_invert(inv);

    return inv;
}