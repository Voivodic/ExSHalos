#ifndef BOX_H
#define BOX_H

#include "exshalos_h.h"
#include "density_grid.h"
#include "find_halos.h"
#include "lpt.h"

/*Generate a halo catalogue in a box for a given linear power spectrum*/
size_t Generate_Halos_Box_from_Pk(fft_real *K, fft_real *P, int Nk, fft_real R_max, fft_real k_smooth, HALOS **halos, fft_real **posh, fft_real **velh, long *flag, fft_real *delta, fft_real *S, fft_real *V, int fixed, fft_real phase);

/*Generate a halo catalogue in a box from a given density grid*/
size_t Generate_Halos_Box_from_Grid(fft_real *K, fft_real *P, int Nk, fft_real k_smooth, HALOS **halos, fft_real **posh, fft_real **velh, long *flag, fft_real *delta, fft_real *S, fft_real *V, int IN_disp);

#endif