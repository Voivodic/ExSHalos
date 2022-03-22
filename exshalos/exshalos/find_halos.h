#ifndef FIND_HALOS_H
#define FIND_HALOS_H

#include "exshalos_h.h"

/*Check if the current position is a peak of the density grid*/
char Check_Peak(fft_real *delta, fft_real den, int i, int j, int k);

/*Count the number of peaks*/
size_t Count_Peaks(fft_real *delta);

/*Save the positions and density of each peak*/
void Find_Peaks(fft_real *delta, size_t np, PEAKS *peaks);

/*Partition function for the quicksort*/
size_t partition_peaks(PEAKS *a, size_t l, size_t r);

/*The quicksort algorithm to sort the peaks list*/
void quickSort_peaks(PEAKS *a, size_t l, size_t r);

/*Barrier used for the halo definition*/
fft_real Barrier(fft_real S);

/*It grows the spheres around the peaks to create the halos*/
size_t Grow_Halos(size_t np, size_t *flag, fft_real *Sig_Grid, fft_real *delta, PEAKS *peaks, HALOS *halos);

/*Compute the number of grid cells inside each possible sphere*/
void Compute_Spheres(int Ncells, char *spheresfile);

/*Read the number of grid cells inside each possible sphere*/
void Read_Spheres(int **sphere, char *spheresfile);

/*Find the index of the next sphere*/
int Next_Count(int *spheres, int Ncells, int count);

/*Compute the mass of each halo*/
void Compute_Mass(size_t nh, int *sphere, HALOS *halos, gsl_interp_accel *acc, gsl_spline *spline_I, gsl_spline *spline_InvI, fft_real *Massh);

#endif
