#ifndef SPLIT_GALAXIES_H
#define SPLIT_GALAXIES_H

#include "hod_h.h"

/*Relative occupancy of central red galaxies*/
fft_real Occ_red_cen(fft_real Mh);

/*Relative occupancy of central blue galaxies*/
fft_real Occ_blue_cen(fft_real Mh);

/*Relative occupancy of satellite red galaxies*/
fft_real Occ_red_sat(fft_real Mh);

/*Relative occupancy of satellite blue galaxies*/
fft_real Occ_blue_sat(fft_real Mh);

/*Determine the type of each galaxy*/
void Galaxy_Types(size_t ng, fft_real *Massh, long *flag, int *gal_type, gsl_rng *rng_ptr);
#endif