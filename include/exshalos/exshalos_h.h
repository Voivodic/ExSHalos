#ifndef EXSHALOS_H
#define EXSHALOS_H

/*Import external libraries used by this module*/
#include <fftw3.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fftlog.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_spline.h>

// #define DOUBLEPRECISION_FFTW
#define CONCAT(prefix, name) prefix##name

#ifdef DOUBLEPRECISION_FFTW
#define FFTW(x) CONCAT(fftw_, x)
#else
#define FFTW(x) CONCAT(fftwf_, x)
#endif

#ifdef DOUBLEPRECISION_FFTW
typedef double fft_real;
typedef fftw_complex fft_complex;
#define NP_OUT_TYPE NPY_DOUBLE
#else
typedef float fft_real;
typedef fftwf_complex fft_complex;
#define NP_OUT_TYPE NPY_FLOAT
#endif

#define FALSE 0
#define TRUE 1

/*Define some number used in the code*/
#define check_memory(p, name)                                                  \
  if (p == NULL) {                                                             \
    printf("Problems to alloc %s.\n", name);                                   \
    exit(0);                                                                   \
  } // Check the memory allocation
#define Lc_MAX 1.0e+2   // Maximum size of a cell
#define Mc_MIN 1.0e+5   // Minimum mass of a cell
#define M_max 1e+16     // Maximum mass of a halo
#define nbar_max 1.0e-2 // Maximum number density for the galaxies

/*Structure for the peaks in the density field*/
typedef struct Halos_centers {
  int x[3];     /*Index of the halo center*/
  fft_real den; /*Density of the halo's central cell*/
} PEAKS;

/*Structure for the final halos*/
typedef struct Halos {
  int x[3];            /*Index of central cell of teh halo*/
  int count;           /*Number of cells in the halo*/
  fft_real Mh;         /*Mass of the halo*/
  fft_real Prof[1000]; /*Density profile of the halo in Lagrangian space*/
} HALOS;

/*Structure with cosmology*/
typedef struct Cosmo {
  fft_real Om0;      // Omega_m value today (z=0)
  fft_real redshift; // Redshift of the final catalogues
  fft_real dc; // Value of the critical density for the halo formation linearly
               // extrapoleted
  fft_real Mstar; // The mass in which delta_c == sigma(M) (used in the
                  // concentration)
  fft_real rhoc;  // Critical density in unitis of M_odot/Mpc*h^2
  fft_real Hz;    // Hubble constant at the final redshift
  fft_real Omz;   // Matter contrast density at the final redshift
  fft_real rhomz; // Matter density at the final redshift
  fft_real Dv;    // Overdensity used to put galaxies in the halos
} COSMO;

/*Structure with parameters of the barrier and other options regarding the
 * halos*/
typedef struct Barrier {
  int Nmin;       // Number of particles in the smaller final halo
  int seed;       // Seed for the random generator
  fft_real a;     // Parameter a of the EB
  fft_real beta;  // Parameter beta of the EB
  fft_real alpha; // Parameter alpha of the EB
} BARRIER;

/*Structure with properties of the box*/
typedef struct Box {
  fft_real Lc;    // Size of each cell
  fft_real Mcell; // Mass of each cell
  int nd[3];      // Number of cells along each direction
  int nz2;        // Quantity used to alloc the complex arrays used in the FFTW3
  size_t ng;      // Total number of cells in the grid
  int nmin;       // Number of cells along the smaller direction
  fft_real L[3];  // Size of the box along each direction
  fft_real Mtot;  // Total mass in the box
  fft_real kl[3]; // The fundamental frequency along each direction
  fft_real
      Normx; // The normalization needed when aplyed the FFTW3 from k to x space
  fft_real
      Normk; // The normalization needed when aplyed the FFTW3 from x to k space
} BOX;

/*Structure with the output options*/
typedef struct Output {
  char OUT_DEN;   // Output the density grid?
  char OUT_HALOS; // How to output the halos?
  char OUT_LPT;   // Output the displacements?
  char OUT_VEL;   // Output the velocities?
  char DO_EB;   // Parameter with the information about the utilization (or not)
                // of the EB
  char DO_HOD;  // Populate the halos with no galaxies (0), one type of galaxy
                // (1) or multiple types (2)?
  char DO_2LPT; // Parameter with the information about the use (or not) of
                // second order lagrangian perturbation theory
  char OUT_PROF; // Output the profile of the halos in Lagrangian space?
  char VERBOSE;  // Print the information about the current state of the
                 // catalogue generation: yes (1) or no (0)?
} OUTPUT;

/*Structure with properties of the lightcone*/
typedef struct Ligthcone {
  fft_real Pobs[3];   // Position of the observer in units of the box size
  fft_real LoS[3];    // Direction of the line of sight
  int Nrep[3];        // Number of replicas of the box need along each direction
  fft_real dist_min;  // Minimum comoving distance of this slice
  fft_real dist_max;  // Maximum comoving distance of this slice
  fft_real theta_max; // Minimum angle theta
  fft_real cos_min;   // Cossine of the minimum angle theta
  int nsnap;          // Number of this snapshot
} LIGHTCONE;

/*Structure with the HOD parameters*/
typedef struct HOD {
  fft_real logMmin; // Parameter log(M_min)
  fft_real siglogM; // Parameter sigma_log(M)
  fft_real logM0;   // Parameter log(M_0)
  fft_real logM1;   // Parameter log(M_1)
  fft_real alpha;   // Parameter alpha
} HOD;

/*Define the global structure variables*/
extern COSMO cosmo;
extern BARRIER barrier;
extern BOX box;
extern OUTPUT out;
extern LIGHTCONE lightcone;
extern HOD hod;

/*Define the distance between two cells*/
size_t dist2(size_t i, size_t j, size_t k);

/*Define the cyclic sum for floats*/
fft_real cysumf(fft_real x, fft_real y, fft_real L);

/*Define the cyclic sum for ints*/
int cysum(int i, int j, int nd);

/*Window function in the Fourier space*/
fft_real W(fft_real k, fft_real R);

/*Set the parameters of the box*/
void set_box(int ndx, int ndy, int ndz, fft_real Lc);

/*Set the parameters of the cosmology*/
void set_cosmology(fft_real Om0, fft_real redshift, fft_real dc);

/*Set the parameters of the barrier*/
void set_barrier(int Nmin, fft_real a, fft_real beta, fft_real alpha, int seed);

/*Set the parameters of the outputs*/
void set_out(char OUT_DEN, char OUT_HALOS, char OUT_LPT, char OUT_VEL,
             char DO_2LPT, char DO_EB, char DO_HOD, char OUT_PROF,
             char VERBOSE);

/*Set the parameters of the lightcone*/
void set_lightcone();

#endif
