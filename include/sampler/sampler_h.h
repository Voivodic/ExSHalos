#ifndef SAMPLER_H
#define SAMPLER_H

#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <fftw3.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

//#define DOUBLEPRECISION_FFTW
#define CONCAT(prefix, name) prefix ## name

#ifdef DOUBLEPRECISION_FFTW
	#define FFTW(x) CONCAT(fftw_, x)
#else
	#define FFTW(x) CONCAT(fftwf_, x)
#endif

#ifdef DOUBLEPRECISION_FFTW
  typedef double fft_real;
  typedef fftw_complex fft_complex;
  #define NP_OUT_TYPE PyArray_FLOAT64
  #define H5_OUT_TYPE H5T_NATIVE_DOUBLE
#else
  typedef float fft_real;
  typedef fftwf_complex fft_complex;
  #define NP_OUT_TYPE PyArray_FLOAT32
  #define H5_OUT_TYPE H5T_NATIVE_FLOAT
#endif

#define FALSE 0
#define TRUE 1

/*Define some number used in the code*/
#define check_memory(p, name) if(p == NULL){printf("Problems to alloc %s.\n", name); exit(0);} //Check the memory allocation

/*Define the global structure variables*/
extern int seed;

/*Invert a given matrix using GSL*/
gsl_matrix *invert_matrix(gsl_matrix *matrix, int size);

#endif