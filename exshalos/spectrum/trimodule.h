#ifndef TRIMODULE_H
#define TRIMODULE_H

#include "spectrum_h.h"

/*Compute all the cross trispectra for the covariance*/
int Tri_Spectrum(fft_real *grid, int nd, fft_real L, int ntype, int window, fft_real R, int interlacing, int Nk, fft_real k_min, fft_real k_max, long double *K1, long double *K2, long double **T, long double **Tu, long double *IT, long double *KP, long double **P, long double *IP);

#endif
