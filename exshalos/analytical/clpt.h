#ifndef CLPT_H
#define CLPT_H

#include "analytical_h.h"

/*Compute xi_l,m for a given power spectrum using a gaussian smoothing and the interpolation in the r->0 limit*/
void Xi_lm(double *k, double *P, int Nk, double *rlm, double *xilm, int Nr, int l, int mk, int mr, int K, double alpha, double Rmax);

/*Compute the matter-matter power spectrum for 1LPT*/
void CLPT_P11(double *k, double *Plin, int N, double *P11, int nmin, int nmax, double kmax);

#endif
