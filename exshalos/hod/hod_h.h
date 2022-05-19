#ifndef HOD_H
#define HOD_H

#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

//#define DOUBLEPRECISION_FFTW
#define CONCAT(prefix, name) prefix ## name

#ifdef DOUBLEPRECISION_FFTW
	#define FFTW(x) CONCAT(fftw_, x)
#else
	#define FFTW(x) CONCAT(fftwf_, x)
#endif

#ifdef DOUBLEPRECISION_FFTW
  typedef double fft_real;
  typedef fftw_complex fft_complex;
  #define NP_OUT_TYPE PyArray_FLOAT64
  #define H5_OUT_TYPE H5T_NATIVE_DOUBLE
#else
  typedef float fft_real;
  typedef fftwf_complex fft_complex;
  #define NP_OUT_TYPE PyArray_FLOAT32
  #define H5_OUT_TYPE H5T_NATIVE_FLOAT
#endif

#define FALSE 0
#define TRUE 1

/*Define some number used in the code*/
#define check_memory(p, name) if(p == NULL){printf("Problems to alloc %s.\n", name); exit(0);} //Check the memory allocation
#define R_MIN 1.0e-3	//Minimum radial normalized radius of a galaxy 
#define R_MAX 10.0		//Maximum radial normalized radius of a galaxy
#define NRs 500			//Number of radial bins used to integrate the density profile and interpolate it
#define NCs 100			//Number of concentration bins used
#define Ng_max 10		//Maximum number of galaxies allowed (times the number of halos)
#define Neps 1000		//Number of bins in Eps used to compute the random radial variables
#define Tot 0.001		//Tolerance used to solve the non-linear equation F^{-1}(r) = eps
#define Lc_MAX	1.0e+2  //Maximum size of a cell
#define Mc_MIN	1.0e+5  //Minimum mass of a cell

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

/*Structure with properties of the box*/
typedef struct Box{
	fft_real Lc; 	//Size of each cell
	fft_real Mcell;	//Mass of each cell
	int nd[3];		//Number of cells along each direction
	int nz2;		//Quantity used to alloc the complex arrays used in the FFTW3
	size_t ng;		//Total number of cells in the grid
	int nmin;		//Number of cells along the smaller direction
	fft_real L[3];	//Size of the box along each direction
	fft_real Mtot;	//Total mass in the box
	fft_real kl[3]; //The fundamental frequency along each direction
	fft_real Normx;	//The normalization needed when aplyed the FFTW3 from k to x space
	fft_real Normk;	//The normalization needed when aplyed the FFTW3 from x to k space
} BOX;

/*Structure with the HOD parameters*/
typedef struct HOD{
	fft_real logMmin;	//Parameter log(M_min)
	fft_real siglogM;	//Parameter sigma_log(M)
	fft_real logM0;	    //Parameter log(M_0)
	fft_real logM1;	    //Parameter log(M_1)
	fft_real alpha;	    //Parameter alpha
	fft_real sigma;		//Parameter of the exclusion term of the halo density profile
} HOD;

/*Structure with the parameters of the color split*/
typedef struct Split{
	fft_real **params_cen;	//Parameters to split the central galaxies	
	fft_real **params_sat;	//Parameters to split the satellite galaxies
	int ntypes;				//Number of different types of galaxies
	int order_cen;			//Order of the polynomium used to split the central galaxies
	int order_sat;			//Order of the polynomium used to split the satellite galaxies
} SPLIT;

/*Structure with the output options*/
typedef struct Output{
	char OUT_VEL;		//Output the velocities?
	char DO_HOD;      	//Populate the halos with no galaxies (0), one type of galaxy (1) or multiple types (2)?
	char IN_C;			//There were concentrations given
	char VERBOSE; 		//Print the information about the current state of the catalogue generation: yes (1) or no (0)?
} OUTPUT;

/*Define the global structure variables*/
COSMO cosmo;
BOX box;
HOD hodp;
SPLIT split;
OUTPUT out;
int seed;

/*Define the cyclic sum for floats*/
fft_real cysumf(fft_real x, fft_real y, fft_real L);

/*Define the cyclic sum for ints*/
int cysum(int i, int j, int nd);

/*Set the parameters of the box*/
void set_box(int ndx, int ndy, int ndz, fft_real Lc);

/*Set the parameters of the cosmology*/
void set_cosmology(fft_real Om0, fft_real redshift, fft_real dc);

/*Set the HOD parameters*/
void set_hod(fft_real logMmin, fft_real siglogM, fft_real logM0, fft_real logM1, fft_real alpha, fft_real sigma);

/*Set the parameters used to make the split between the different colors*/
void set_split(fft_real *params_cen, fft_real *params_sat, int ntypes, int order_cen, int order_sat);

/*Set the parameters of the outputs*/
void set_out(char OUT_VEL, char DO_HOD, char IN_C, char VERBOSE);

#endif