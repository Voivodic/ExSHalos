#include "finder_h.hpp"

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
