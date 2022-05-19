#ifndef SPLIT_GALAXIES_H
#define SPLIT_GALAXIES_H

#include "hod_h.h"

/*Relative occupancy of central galaxies for each type*/
fft_real Occ_cen(fft_real log10Mh, int type);

/*Relative occupancy of satellite galaxies for each type*/
fft_real Occ_sat(fft_real log10Mh, int type);

/*Determine the type of each galaxy*/
void Galaxy_Types(size_t ng, fft_real *Massh, long *flag, int *type, gsl_rng *rng_ptr);

#endif