#ifndef GRIDMODULE_H
#define GRIDMODULE_H

#include "spectrum_h.h"

/*Define the Hubble function in units of h*/
fft_real H(fft_real Om0, fft_real z);

/*Give the density to each grid using the NGP density assignment*/
long double NGP(fft_real *grid, fft_real *pos, int nd, fft_real L, fft_real mass);

/*Give the density to each grid using the CIC density assignment*/
long double CIC(fft_real *grid, fft_real *pos, int nd, fft_real L, fft_real mass);

/*Give the density to each grid using a sphere*/
long double Sphere(fft_real *grid, fft_real *pos, int nd, fft_real L, fft_real R, fft_real mass);

/*Give the density to each grid using a sphere*/
long double Exp(fft_real *grid, fft_real *pos, int nd, fft_real L, fft_real R, fft_real R_times, fft_real mass);

/*Compute the density grid given a list of particles*/
long double Density_Grid(fft_real *grid, int nd, fft_real L, fft_real *pos, fft_real mass, int window, fft_real R, fft_real R_times);

/*Compute the density grids for each type of tracer*/
void Tracer_Grid(fft_real *grid, int nd, fft_real L, int direction, fft_real *pos, fft_real *vel, size_t np, fft_real *mass, int *type, int ntype, int window, fft_real R, fft_real R_times, int interlacing, fft_real Om0, fft_real z, int folds);

#endif