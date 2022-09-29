#ifndef ANALYTICAL_H
#define ANALYTICAL_H

#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <fftw3.h>

#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>
#include "fftlog.h"

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

/*Structure for the interpolation used in the integrals*/
struct func_params{ 
    gsl_interp_accel *facc; 
    gsl_spline *fspline; 
};

#endif