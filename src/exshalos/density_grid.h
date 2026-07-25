#ifndef DENSITY_GRID_H
#define DENSITY_GRID_H

#include "exshalos_h.h"

/*Read the density grid*/
void Read_Den(char *denfile, fft_real *delta);

/*Interpole the power spectrum*/
void Inter_Power(fft_real *K, fft_real *P, int Nk, fft_real R_max, gsl_spline *spline);

/*Compute the Gaussian density grid*/
void Compute_Den(fft_real *K, fft_real *P, int Nk, fft_real R_max, fft_real *delta, fft_complex *deltak, int fixed, fft_real phase, fft_real k_smooth);

/*Compute deltak given delta*/
void Compute_Denk(fft_real *delta, fft_complex *deltak);

/*Save the density field*/
void Save_Den(char *denfile, fft_real *delta);

/*Compute the mean and std of the density field*/
void Compute_MS(fft_real *delta);

/*Compute the density field to a given power*/
void Compute_Den_to_n(fft_real *delta, fft_real *delta_n, int n, int renormaliormalized);

/*Compute the potential field of a given density field*/
void Compute_Phi(fft_real *delta, fft_complex *deltak, fft_real *phi);

/*Compute the laplacian of the dencity field*/
void Compute_Laplacian_Delta(fft_real *delta, fft_complex *deltak, fft_real *laplacian);

/*Compute the tidal field*/
void Compute_Tidal(fft_real *delta, fft_complex *deltak, fft_real *tidal);

/*Get the position in the flat array of the tidal field*/
int Get_IndK(int i, int j);

/*Compute K2 given the density field or the tidal field and the subtraction of the delta field (given by a):K2 =  K^2 - a*delta^2*/
void Compute_K2(fft_real *delta, fft_complex *deltak, fft_real *tidal, fft_real *K2, fft_real a, int renormalized);

/*Compute K3 given the density field or the tidal field*/
void Compute_K3(fft_real *delta, fft_complex *deltak, fft_real *tidal, fft_real *K3, fft_real a, fft_real b);

#endif
