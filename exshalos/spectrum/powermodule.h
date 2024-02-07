#ifndef POWERMODULE_H
#define POWERMODULE_H

#include "spectrum_h.h"
#include <gsl/gsl_sf.h>

/*Computes the cross power spectrum of the different tracers*/
void Power_Spectrum(fft_real *grid, int nd, fft_real L, int ntype, int window, fft_real R, int interlacing, int Nk, fft_real k_min, fft_real k_max, long double *Kmean, long double *P, long *count_k, int l_max, int direction);

/*Compute all the power spectra between one particle and the field*/
void Power_Spectrum_individual(fft_real *grid, fft_real *pos, int np, int nd, fft_real L, int ntype, int window, fft_real R, int interlacing, int Nk, fft_real k_min, fft_real k_max, long double *Kmean, long double *P, long *count_k, int l_max, int direction);

#endif