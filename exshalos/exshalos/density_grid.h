#ifndef DENSITY_GRID_H
#define DENSITY_GRID_H

#include "exshalos_h.h"

/*Read the density grid*/
void Read_Den(char *denfile, fft_real *delta);

/*Interpole the power spectrum*/
void Inter_Power(fft_real *K, fft_real *P, int Nk, fft_real R_max, gsl_spline *spline);

/*Compute the Gaussian density grid*/
void Compute_Den(fft_real *K, fft_real *P, int Nk, fft_real R_max, fft_real *delta, fft_complex *deltak, int seed);

/*Compute deltak given delta*/
void Compute_Denk(fft_real *delta, fft_complex *deltak);

/*Save the density field*/
void Save_Den(char *denfile, fft_real *delta);

/*Compute the mean and std of the density field*/
void Compute_MS(fft_real *delta);

#endif
