#ifndef LPT_H
#define LPT_H

#include "exshalos_h.h"

/*Compute the first order displacements*/
void Compute_1LPT(fft_complex *deltak, fft_real **posh, fft_real **velh, fft_real *S, fft_real *V, size_t *flag, fft_real k_smooth);

/*Compute the second order displacements*/
void Compute_2LPT(fft_real **posh, fft_real **velh, fft_real *S, fft_real *V, size_t *flag, fft_real k_smooth);

/*Compute the final position of each particle*/
void Compute_Pos(fft_real *S);

/*Compute the final position and velocity of each halo*/
void Compute_Posh(HALOS *halos, fft_real **posh, fft_real **velh, size_t nh);

#endif
