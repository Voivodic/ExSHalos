#ifndef CLPT_H
#define CLPT_H

#include "analytical_h.h"

/*Supress the linear power spectrum on small scales*/
void P_smooth(double *k, double *Plin, double *P, int N, double Lambda);

/*Define the generic function to be integrated*/
double func(double x, void *p);

/*Compute the matter-matter power spectrum for 1LPT*/
void CLPT_P11(double *k, double *Plin, int N, double *P11, int nmin, int nmax, double kmax);

#endif
