#ifndef BIMODULE_H
#define BIMODULE_H

#include "spectrum_h.h"

/*Compute all the cross bispectra*/
int Bi_Spectrum(fft_real *grid, int nd, fft_real L, int ntype, int window, fft_real R, int interlacing, int Nk, fft_real k_min, fft_real k_max, long double *K1, long double *K2, long double *K3, long double **B, long double *I, long double *KP, long double **P, long double *IP, int verbose);

#endif