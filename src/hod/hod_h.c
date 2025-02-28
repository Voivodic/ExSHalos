#include "hod_h.h"

/*Define the cyclic sum for floats*/
fft_real cysumf(fft_real x, fft_real y, fft_real L){
	fft_real resp;

	resp = x + y;
	if(resp>=L)	resp -= L;
	if(resp<0)	resp += L;

	return resp;
}

/*Define the cyclic sum*/
int cysum(int i, int j, int nd){
	int resp;

	resp = i+j;
	if(resp>=nd)	resp -= nd;
	if(resp<0)	resp += nd;

	return resp;
}

/*Set the parameters of the cosmology*/
void set_cosmology(fft_real Om0, fft_real redshift, fft_real dc){
	/*Set the first parameters*/
	cosmo.Om0 = Om0;			//Omega_m value today (z=0)
	cosmo.redshift = redshift;	//Redshift of the final catalogues			 

	/*Set the derivad parameters*/
	cosmo.rhoc = 2.775e+11;			//Critical density in unitis of M_odot/Mpc*h^2
	cosmo.Hz = 100.0*sqrt(cosmo.Om0*pow(1.0 + cosmo.redshift, 3.0) + (1.0 - cosmo.Om0));	//Hubble constant at the final redshift
	cosmo.Omz = cosmo.Om0*pow(1.0 + cosmo.redshift, 3.0)/(cosmo.Om0*pow(1.0 + cosmo.redshift, 3.0) + (1.0 - cosmo.Om0));//Matter contrast density at the final redshift
	cosmo.rhomz = cosmo.Om0*cosmo.rhoc;			//Matter density at the final redshift
	cosmo.Dv = (18*M_PI*M_PI + 82.0*(cosmo.Omz - 1.0) - 39.0*pow(cosmo.Omz - 1.0, 2.0))/cosmo.Omz;		//Overdensity used to put galaxies in the halos
	/*Value of the critical density for the halo formation linearly extrapoleted*/
	if(dc <= 0.0)	cosmo.dc = 1.686*pow(cosmo.Omz, 0.0055);
	else 			cosmo.dc = dc;	
}

/*Set the parameters of the box*/
void set_box(int ndx, int ndy, int ndz, fft_real Lc){
	/*Set the number of cells in each direction*/
	box.nd[0] = ndx;	//Number of cells along the x direction
	box.nd[1] = ndy;	//Number of cells along the y direction
	box.nd[2] = ndz;	//Number of cells along the z direction
	box.ng = ((size_t) ndx)*((size_t) ndy)*((size_t) ndz);	//Total number of cells in the grid
	box.nz2 = ndz/2 + 1;//Number of cell along the z direction in Fourier space

	/*If the size of each cell was given compute the mass of each cell*/
	if(box.Lc < Lc_MAX){	
		box.Lc = Lc;
		box.Mcell = cosmo.rhomz*pow(Lc, 3.0);
	}
	/*If the mass of each cell was given compute the size of each cell*/
	else if(box.Lc > Mc_MIN){			
		box.Mcell = box.Lc;
		box.Lc = pow(box.Mcell/cosmo.rhomz, 1.0/3.0);
	}
	/*Notify an unexpected behavior and exit*/
	else{					
		printf("A cell larger than %f [Mpc/h] or with a mass smaller than %e [M_odot/h] is not expected. Please, change this value or change the definition of Lc_MAX and Mc_MIN in the code.\n", Lc_MAX, Mc_MIN);
		exit(0);
	}

	/*Set the other derived parameters*/
	box.L[0] = box.Lc*box.nd[0];				//Compute the size of the box along the x-direction
	box.L[1] = box.Lc*box.nd[1];				//Compute the size of the box along the y-direction
	box.L[2] = box.Lc*box.nd[2];				//Compute the size of the box along the z-direction
	box.Mtot = cosmo.rhomz*box.L[0]*box.L[1]*box.L[2];			//Compute the total mass in the box
	box.kl[0] = 2.0*M_PI/box.L[0];			//Compute the fundamental frequency in the x-direction
	box.kl[1] = 2.0*M_PI/box.L[1];			//Compute the fundamental frequency in the y-direction
	box.kl[2] = 2.0*M_PI/box.L[2];			//Compute the fundamental frequency in the z-direction
	box.Normx = 1.0/sqrt(box.L[0]*box.L[1]*box.L[2]);		//Compute the normalization needed when aplyed the FFTW3 from k to x space
	box.Normk = sqrt(box.L[0]*box.L[1]*box.L[2])/(box.nd[0]*box.nd[1]*box.nd[2]);	//Compute the normalization needed when aplyed the FFTW3 from x to k space
	box.nmin = box.nd[0];				//Determine the smaller direction
	if(box.nmin > box.nd[1])	box.nmin = box.nd[1];
	if(box.nmin > box.nd[2])	box.nmin = box.nd[2];
}

/*Set the HOD parameters*/
void set_hod(fft_real logMmin, fft_real siglogM, fft_real logM0, fft_real logM1, fft_real alpha, fft_real sigma){
    hodp.logMmin = logMmin;	//Parameter log(M_min)
	hodp.siglogM = siglogM;	//Parameter sigma_log(M)
	hodp.logM0 = logM0;	    //Parameter log(M_0)
	hodp.logM1 = logM1;	    //Parameter log(M_1)
	hodp.alpha = alpha;	    //Parameter alpha
	hodp.sigma = sigma;		//Parameter of the exclusion term of the halo density profile
}

/*Set the parameters used to make the split between the different colors*/
void set_split(fft_real *params_cen, fft_real *params_sat, int ntypes, int order_cen, int order_sat){
	int i, j;

	split.ntypes = ntypes;
	split.order_cen = order_cen;
	split.order_sat = order_sat;

	split.params_cen = (fft_real **)malloc((ntypes - 1)*sizeof(fft_real *));
	split.params_sat = (fft_real **)malloc((ntypes - 1)*sizeof(fft_real *));
	for(i=0;i<ntypes-1;i++){
		split.params_cen[i] = (fft_real *)malloc(order_cen*sizeof(fft_real));
		for(j=0;j<order_cen;j++)
			split.params_cen[i][j] = params_cen[i*order_cen + j];
		split.params_sat[i] = (fft_real *)malloc(order_sat*sizeof(fft_real));	
		for(j=0;j<order_sat;j++)
			split.params_sat[i][j] = params_sat[i*order_sat + j];
	}
}

/*Set the parameters of the outputs*/
void set_out(char OUT_VEL, char DO_HOD, char IN_C, char VERBOSE){
	out.OUT_VEL = OUT_VEL;		//Output the velocities?
	out.DO_HOD = DO_HOD;      	//Populate the halos with no galaxies (0), one type of galaxy (1) or multiple types (2)?
	out.IN_C = IN_C;			//There were concentrations given
	out.VERBOSE = VERBOSE;		//Print the information about the current state of the catalogue generation: yes (1) or no (0)?
}

/*Defining the global variables*/
COSMO cosmo;
BOX box;
HOD hodp;
SPLIT split;
OUTPUT out;
int seed;