#ifndef POPULATE_HALOS_H
#define POPULATE_HALOS_H

#include "hod_h.h"

/*Mean value of central galaxies*/
fft_real Ncentral_tot(fft_real M);

/*Mean value of satellite galaxies*/
fft_real Nsatellite_tot(fft_real M);

/*Halo concentration*/
fft_real f_c(fft_real Mv);

/*Unnormalized density profile used to populate the halos*/
fft_real Profile(fft_real x, fft_real c);

/*Compute construct the interpolations used to generate the radial position of the galaxies*/
void Interpolate_r_Eps(fft_real cmin, fft_real cmax, gsl_spline **spline_r, gsl_interp_accel *acc, fft_real rmax);

/*Compute the radial distance given epsilon*/
fft_real Generate_r(size_t ind_h, fft_real w1, fft_real w2, gsl_spline **spline_r, gsl_interp_accel *acc, fft_real Eps);

/*Compute the number of galaxies and their positions and velocities in each halo*/
size_t Populate_total(size_t nh, fft_real *posh, fft_real *velh, fft_real *Massh, fft_real *Ch, fft_real *posg, fft_real *velg, long *gal_type, gsl_rng *rng_ptr);

// Populate the halos with particles
size_t Populate_Particles(size_t nh, fft_real *posh, fft_real *velh, fft_real *Massh, fft_real *Ch, fft_real *posg, fft_real *velg, long *gal_type, fft_real massp, fft_real M_frac, size_t np, gsl_rng *rng_ptr){

#endif
