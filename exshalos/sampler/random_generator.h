#ifndef RANDOM_GENERATOR_H
#define RANDOM_GENERATOR_H

#include "sampler_h.h"

/*Generate an array of random number following a given PDF*/
void Generate_Random_Array(gsl_spline *spline_rho, unsigned long long nps, double rmin, double rmax, fft_real *rng, gsl_rng *rng_ptr, int Inter_log, int NRs, int Neps, double Tot);

#endif