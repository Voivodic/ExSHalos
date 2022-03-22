#ifndef SPECTRUM_H
#define SPECTRUM_H

#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <fftw3.h>

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
#else
  typedef float fft_real;
  typedef fftwf_complex fft_complex;
  #define NP_OUT_TYPE PyArray_FLOAT32
#endif

#define FALSE 0
#define TRUE 1

/*Evaluate the ciclic sum of x and y*/
int mod(int x, int y, int nd);

/*Define the cyclic sum for floats*/
fft_real cysumf(fft_real x, fft_real y, fft_real L);

/*Give a indice for the partricle*/
void ind(fft_real x[], int xt[], fft_real Ld, int nd);

/*Define de sinc function*/
fft_real sinc(fft_real x);

/*Define window function for NGP and CIC*/
fft_real W(fft_real k1, fft_real k2, fft_real k3, fft_real Lb, fft_real R, int window);

/*Define the bin for the mode*/
int Indice(fft_real k, fft_real kmin, fft_real dk);

/*Compute the density grids in Fourier space corrected by the interlacing and window function*/
void Compute_gridk(fft_real *grid, fft_complex **gridk, int nd, fft_real L, int ntype, int interlacing, fft_real R, int window);

#endif