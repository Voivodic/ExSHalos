#ifndef ABUNDANCE_H
#define ABUNDANCE_H

#include "spectrum_h.h"

/*Measure the abundance of a given halo catalogue*/
void Measure_Abundance(fft_real *Mh, size_t nh, fft_real Mmin, fft_real Mmax, int Nm, fft_real *Mmean, fft_real *dn, fft_real *dn_err, fft_real Lx, fft_real Ly, fft_real Lz);

#endif