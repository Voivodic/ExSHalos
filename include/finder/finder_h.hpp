#ifndef FINDER_HPP
#define FINDER_HPP

/*External libraries used by this module*/
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// #define DOUBLEPRECISION_FFTW
#define CONCAT(prefix, name) prefix##name

#ifdef DOUBLEPRECISION_FFTW
#define FFTW(x) CONCAT(fftw_, x)
#else
#define FFTW(x) CONCAT(fftwf_, x)
#endif

#ifdef DOUBLEPRECISION_FFTW
    typedef double fft_real;
#define NP_OUT_TYPE NPY_DOUBLE
#else
typedef float fft_real;
#define NP_OUT_TYPE NPY_FLOAT
#endif

#define FALSE 0
#define TRUE 1

/*Define function that checks if an array was allocated correctly*/
#define check_memory(p, name)                                                  \
    if (p == NULL) {                                                           \
        printf("Problems to alloc %s.\n", name);                               \
        exit(0);                                                               \
    }

/*Define some hard coded numbers*/
#define PART_ALLOC_PER_BLOCK 8
#define Lc_MAX	1.0e+2  //Maximum size of a cell
#define Mc_MIN	1.0e+5  //Minimum mass of a cell

/*Structure with properties of the box*/
typedef struct Box {
    fft_real Lc;    // Size of each cell
    fft_real Mcell; // Mass of each cell
    int nd[3];      // Number of cells along each direction
    int nz2;   // Quantity used to alloc the complex arrays used in the FFTW3
    size_t ng; // Total number of cells in the grid
    int nmin;  // Number of cells along the smaller direction
    fft_real L[3];  // Size of the box along each direction
    fft_real Mtot;  // Total mass in the box
    fft_real kl[3]; // The fundamental frequency along each direction
    fft_real Normx; // Normalization needed when going from k to x space
    fft_real Normk; // TNormalization needed when going from x to k space
} BOX;

/*Structure with cosmology*/
typedef struct Cosmo{
	fft_real Om0;		//Omega_m value today (z=0)
	fft_real redshift;	//Redshift of the final catalogues
	fft_real dc;		//Value of the critical density for the halo formation linearly extrapoleted 
	fft_real Mstar;		//The mass in which delta_c == sigma(M) (used in the concentration)
	fft_real rhoc;		//Critical density in unitis of M_odot/Mpc*h^2
	fft_real Hz;		//Hubble constant at the final redshift
	fft_real Omz;		//Matter contrast density at the final redshift
	fft_real rhomz;		//Matter density at the final redshift
	fft_real Dv;		//Overdensity used to put galaxies in the halos
} COSMO;

/*Declare the global structure variables*/
extern COSMO cosmo;
extern BOX box;

/*Set the parameters of the box*/
void set_box(int ndx, int ndy, int ndz, fft_real Lc);

/*Set the parameters of the cosmology*/
void set_cosmology(fft_real Om0, fft_real redshift, fft_real dc);

#endif
