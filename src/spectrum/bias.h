#ifndef BIAS_H
#define BIAS_H

#include "spectrum_h.h"

/*Measure the masked and unmasked histograms of the initial density field*/
void Measure_Histogram(fft_real *delta, fft_real *Mh, long *flag, fft_real Mmin, fft_real Mmax, int Nm, fft_real dmin, fft_real dmax, int Nbins, size_t ng, size_t nh, long *hist_unmasked, long *hist_masked, int Central);

#endif