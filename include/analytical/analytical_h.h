#ifndef ANALYTICAL_H
#define ANALYTICAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <fftw3.h>
#include "fftlog.h"
#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_filter.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>

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
  #define NP_OUT_TYPE NPY_DOUBLE
#else
  typedef float fft_real;
  typedef fftwf_complex fft_complex;
  #define NP_OUT_TYPE NPY_FLOAT
#endif

#define FALSE 0
#define TRUE 1

/*Structure for the interpolation used in the integrals*/
struct finterp_params{ 
  gsl_interp_accel *facc; 
  gsl_spline *fspline; 
};

/*Supress the linear power spectrum on small scales*/
void P_smooth(double *k, double *Plin, double *P, int N, double Lambda);

/*Define the generic function to be integrated*/
double finterp(double x, void *p);

#endif
