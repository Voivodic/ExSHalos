#ifndef DENSITY_GRID_H
#define DENSITY_GRID_H

#include "exshalos_h.h"

/*Window function in the Fourier space*/
fft_real W(fft_real k, fft_real R);

/*Evaluate the square root of matter variance*/
fft_real calc_sigma(fft_real *k, fft_real *P, int Nk, fft_real R);

/*Evaluate the mass function for a given sigma*/
fft_real fh(fft_real sigma, int model);

/*Compute sigma(M) as function of the number of cells*/
void Compute_Sig(int Nr, fft_real *R, fft_real *M, fft_real *Sig, fft_real *Sig_grid, fft_real *K, fft_real *P, int Nk);

/*Compute the integral over the mass function and interpolate it*/
void Compute_nh(int model, int Nr, fft_real *R, fft_real *M, fft_real *Sig, gsl_spline *spline_I, gsl_spline *spline_InvI);

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
