#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <fftw3.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "fftlog.h"

#define check_memory(p, name) if(p == NULL){printf("Problems to alloc %s.\n", name); return 0;}
#define Lc_MAX	1.0e+2
#define Mc_MIN	1.0e+5
#define M_max 	6.0e+15

/*Structure for the peaks in the density field*/
typedef struct Halos_centers {
	int x[3];		/*Index of the halo center*/
	float den;		/*Density of the halo's central cell*/
} PEAKS;

/*Structure for the final halos*/
typedef struct Halos {
	int x[3];		/*Index of central cell of teh halo*/
	int cont;		/*Number of cells in the halo*/
} HALOS;

/*Barrier used for the halo definition*/
float Barrier(float S, float dc, char barrier, float a, float b, float alpha){
	float resp;

	/*The Press-Schechter barrier*/
	if(barrier == 0)
		resp = dc;

	/*The Sheth-Tormen barrier*/
	else if(barrier == 1)
		resp = sqrt(a)*dc*(1.0 + b*pow(S/(a*dc*dc), alpha));

	return resp;
}

/*Partition function for the quicksort*/
long int partition_peaks( PEAKS a[], long l, long r) {
   	long i, j, k;
	PEAKS pivot, t;
  	pivot.den = a[l].den;
	for(k=0;k<3;k++) pivot.x[k] = a[l].x[k];	
   	i = l; j = r+1;
		
   	while( 1){
   	do ++i; while( a[i].den >= pivot.den && i < r );
   	do --j; while( a[j].den < pivot.den );
   	if( i >= j ) break;
   	t.den = a[i].den; a[i].den = a[j].den; a[j].den = t.den;
	for(k=0;k<3;k++){ t.x[k] = a[i].x[k]; a[i].x[k] = a[j].x[k]; a[j].x[k] = t.x[k];}
   	}
   	t.den = a[l].den; a[l].den = a[j].den; a[j].den= t.den;
	for(k=0;k<3;k++){ t.x[k] = a[l].x[k]; a[l].x[k] = a[j].x[k]; a[j].x[k] = t.x[k];}
   	return j;
}

/*The quicksort algorithm to sort the peaks list*/
void quickSort_peaks( PEAKS a[], long l, long r){
	long j;

   	if( l < r ){
   	// divide and conquer
        j = partition_peaks( a, l, r);
       	quickSort_peaks( a, l, j-1);
      	quickSort_peaks( a, j+1, r);
   	}	
}

/*Define the distance between two cells*/
long int dist2(long int i, long int j, long int k){
	long int resp;

	resp = i*i + j*j + k*k;

	return resp;
}

/*Define the cyclic sum for floats*/
float cysumf(float x, float y, float L){
	float resp;

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

/*Window function in the Fourier space*/
double W(double k, double R){
	double resp;

	resp = 3.0/(pow(k*R,2))*(sin(k*R)/(k*R) - cos(k*R));
	return resp;
}

/*Evaluate the square root of matter variance*/
double calc_sigma(double *k, double *P, int Nk, double R){
	int i;
	double resp;

	resp = 0.0;
	for(i=0;i<Nk-1;i++)
		resp += (k[i+1] - k[i])/2.0*(P[i]*pow(k[i]*W(k[i],R), 2) + P[i+1]*pow(k[i+1]*W(k[i+1],R), 2));

	return resp/(2.0*M_PI*M_PI);
}

/*Evaluate the mass function for a given sigma*/
double fh(double sigma, int model, double dc){
	double resp, nu;
	double B, d, e, f, g;

	//Press-Schechter
	if(model == 0){
		nu = dc/sigma;
		resp = sqrt(2.0/M_PI)*nu*exp(-nu*nu/2.0);
	}

	//Tinker Delta = 300
	else if(model == 1){
		B = 0.466;
		d = 2.06;
		e = 0.99;
		f = 0.48;
		g = 1.310;
	
		resp = B*(pow(sigma/e, -d) + pow(sigma, -f))*exp(-g/(sigma*sigma));
	}

	return resp;
}

/*Find the index of the next sphere*/
int Next_Count(int *spheres, int Ncells, int count){
	int i, resp;

	for(i=0;i<Ncells;i++)
		if(spheres[i] == count){
			resp = i + 1;
			break;
		}

	return resp;
}

/*Halo concentration*/
float f_c(float Mv, float Mstar, float z){
	float resp;

	resp = 9.0/(1.0 + z)*pow(Mv/Mstar, -0.13);

	return resp;
}

/*Generate a random number from 0 to Rv following the NFW profile*/
float Generate_NFW(float rv, float c, float A, int seed){
	float Int, rs, r, rtmp;

	gsl_rng *rng_ptr;
	rng_ptr = gsl_rng_alloc (gsl_rng_taus);
	gsl_rng_set(rng_ptr, seed);

	Int = gsl_rng_uniform(rng_ptr);
	rs = rv/c;

	r = gsl_rng_uniform(rng_ptr);
	rtmp = r + 1.0;
	while(fabs(r-rtmp) > 0.001){
		rtmp = r;
		r = r - ((log(1.0 + r*c) - r*c/(1.0 + r*c) - A*Int)*pow(1.0 + r*c, 2))/(c*(2.0*r*c + r*r*c*c));
	}
    gsl_rng_free(rng_ptr);

	return r*rv;
}

/*Mean value of central galaxies*/
float Ncentral(float M, float logMmin, float siglogM){
	float resp;

	resp = 0.5*(1.0 + erf((log10(M) - logMmin)/siglogM));

	return resp;
}

/*Mean value of satellite galaxies*/
float Nsatellite(float M, float logM0, float logM1, float alpha){
	float resp;

	resp = pow((M - pow(10.0, logM0))/pow(10.0, logM1), alpha);

	return resp;
}


int main(int argc,char *argv[])
{
FILE *power, *den_grid, *halo_cat, *disp_cat, *light_cat, *collapse_cat;
char powerfile[100], denfile[100], halofile[100], dispfile[100], lightfile[100], collapsefile[100];
char DO_2LPT, BARRIER, out_inter, out_halos, out_collapse, DEN_GRID, DISP_CAT, DO_HOD;
int i, j, k, nx, ny, nz, nz2, Nmin, N_cores, Nk, Nr, Ncells, seed, cont_tmp, grows, grows_tmp, tmp, nmin, m, m_tmp, nsnap;
long np, cont, nh, l;
float Lc, Om0, redshift, Growth, dc, EB_a, EB_b, EB_alpha, rhoc, Hz, Omz, rhomz, Mtot, Mcell, Lx, Ly, Lz, klx, kly, klz, Normx, Normk, Dv, kx, ky, kz, kmod, *Sig_grid, sigtmp, std, den, den_tmp, factx, facty, factz, fact, phixx, phixy, phixz, phiyy, phiyz, phizz, Pobs[3], LoS[3], dist_min, dist_max, theta_min, cos_min;
double *K, *P, *R, *M, *Sig, Rmin, Rmax, *R_xi, *Xi;
fftwf_plan p1, p2;
fftwf_complex *deltak, *deltak_tmp;
float *delta;
int *flag, *sphere;
PEAKS *peaks, *tmpp;
HALOS *halos;

if (argc != 35){
	printf("\nWrong number of arguments.\n");
	printf("arg1: Name of the power spectrum file.\n");
	printf("arg2: Size (in Mpc/h) or mass (in M_odot/h) of each cell.\n");
	printf("arg3-5: Number of cells along each direction.\n");
	printf("arg6: Some seed for the random number generator.\n");
	printf("arg7: Use the 2LPT to move the halos? Yes (1) or No (0).\n");
	printf("arg8: The Value of Omega_m today.\n");
	printf("arg9: The readshift z.\n");
	printf("arg10: The rate between the growth function at the final resdshit and at the redshift of the input power spectrum.\n");
	printf("arg11: The value of critical density delta _{c}. Put 0 to use the fit.\n");
	printf("arg12: The minimum number of partiles in a halo of the final catalogue.\n");
	printf("arg13: The number of cores to use in the parallel parts.\n");
	printf("arg14: Prefix for the outputs.\n");
	printf("arg15: Which barrier would you like to use to find the halos?\n\tThe statical barrier (SB) (0);\n\tThe ellipsoidal barrier (EB) (1).\n");
	printf("arg16: Which intermediate results would you like to save?:\n\tNo one (0);\n\tThe gaussian density grid (1);\n\tThe particles displaced with LPT (2);\n\tBoth (3).\n");
	printf("arg17: How do you want the final halo catalogue?\n\tNo halo catalogue (0);\n\tThe positions and velocities in the real space (1);\n\tThe positions and velocities in the real space light cone (2);\n\tThe positions and velocities in redshift space light cone(3).\n");
	printf("arg18-20: The three parameters for the ellipsoidal barrier: a, b and alpha.\n");
	printf("arg21: Read the density grid (0) or compute it (1)?\n");
	printf("arg22: Read the displacement field (0) or compute it (1)?\n");
	printf("arg23-25: Position of the observer in units of the box size.\n");
	printf("arg26-28: Direction of the line of sight.\n");
	printf("arg29-30: Minimum and maximum comoving distance of the halos in this snapshot in the light cone.\n");
	printf("arg31: Angular aperture of the light cone in units of pi.\n");
	printf("arg32: Save the information about the collapsed particles in the light cone? Yes (1) or No (0).\n");
   	printf("arg33: Populate the halos with a HOD?\n\tNo (0);\n\tYes, with a single type of galaxy (1)\n\tYes, with multiple types of galaxies(2).\n"); 
	printf("arg34: Number of this snapshot.\n");

	exit(0);
}

/*Get the name of all files*/
sprintf(powerfile, "%s", argv[1]);
sprintf(denfile, "%s_den.dat", argv[14]);
sprintf(halofile, "%s_halos.dat", argv[14]);
sprintf(dispfile, "%s_disp.dat", argv[14]);

/*Parameters with specifications of the box and options for this simulation*/
Lc = atof(argv[2]);			//Size or mass of each cell
nx = atoi(argv[3]);			//Number of cells along the x-direction
ny = atoi(argv[4]);			//Number of cells along the y-direction
nz = atoi(argv[5]);			//Number of cells along the z-direction
seed = atoi(argv[6]);			//Seed for the random generator (same seed gives the same final catalogue)
DO_2LPT = (char)atoi(argv[7]);		//Parameter with the information about the use (or not) of second order lagrangian perturbation theory
Nmin = atoi(argv[12]);			//Number of particles in the smaller final halo
N_cores = atoi(argv[13]);		//Number of cores used by openmp in the parallel parts
BARRIER = (char)atoi(argv[15]);		//Parameter with the information about the utilization (or not) of the EB
out_inter = (char)atoi(argv[16]);	//Parameter with the information about which intermediate results must be output	
out_halos = (char)atoi(argv[17]);	//Parameter with the information about what to save in the final halo catalogue
out_collapse = (char)atoi(argv[32]);	//Parameter with the information about the collapsed particles in the light cone
DEN_GRID = (char)atoi(argv[21]);	//Compute a new density field (1) or just read it from a file (0)?
DISP_CAT = (char)atoi(argv[22]);	//Compute the displacement field (1) or just read it from a file (0)?
DO_HOD = (char)atoi(argv[33]);      //Populate the halos with no galaxies (0), one type of galaxy (1) or multiple types (2)?

/*Some physical parametrs used in this simulation*/
Om0 = atof(argv[8]);			//Omega_m value today (z=0)
redshift = atof(argv[9]);		//Redshift of the final catalogues
Growth = atof(argv[10]);		//Ratio between the growth function at the final redshift and the redshift of the inpur power spectrum
dc = atof(argv[11]);			//Value of the critical density for the halo formation linearly extrapoleted using linear theory to the redshift of the final catalogues

/*Parameters for the EB*/
EB_a = atof(argv[18]);			//Parameter a of the EB
EB_b = atof(argv[19]);			//Parameter b of the EB
EB_alpha = atof(argv[20]);		//Parameter alpha of the EB

/*Parameters for the construction of the light cone*/
Pobs[0] = atof(argv[23]);		//Position x of the observer in units of the box size
Pobs[1] = atof(argv[24]);		//Position y of the observer in units of the box size
Pobs[2] = atof(argv[25]);		//Position z of the observer in units of the box size
LoS[0] = atof(argv[26]);		//Component x of the direction of the line of sight
LoS[1] = atof(argv[27]);		//Component y of the direction of the line of sight
LoS[2] = atof(argv[28]);		//Component z of the direction of the line of sight
/*Normalize the LoS vector*/
kmod = 0.0;
for(i=0;i<3;i++)
	kmod += LoS[i]*LoS[i];
for(i=0;i<3;i++)
	LoS[i] = LoS[i]/sqrt(kmod);
dist_min = atof(argv[29]);		//Minimum comoving distance of this slice
dist_max = atof(argv[30]);		//Maximum comoving distance of this slice
theta_min = atof(argv[31])*M_PI;	//Minimum angle theta
cos_min = cos(theta_min);		//Cossine of the minimum angle theta
nsnap = atoi(argv[34]);			//Number of this snapshot
sprintf(lightfile, "%s_%d_LightCone.dat", argv[14], nsnap);
sprintf(collapsefile, "%s_%d_Collapse.dat", argv[14], nsnap);
	
/*Some derived parameters used in this simulation*/
rhoc = 2.775e+11;			//Critical density in unitis of M_odot/Mpc*h^2
Hz = 100.0*sqrt(Om0*pow(1.0 + redshift, 3.0) + (1.0 - Om0));			//Hubble constant at the final redshift
Omz = Om0*pow(1.0 + redshift, 3.0)/(Om0*pow(1.0 + redshift, 3.0) + (1.0 - Om0));//Matter contrast density at the final redshift
rhomz = Om0*rhoc;			//Matter density at the final redshift
Dv = (18*M_PI*M_PI + 82.0*(Omz - 1.0) - 39.0*pow(Omz - 1.0, 2.0))/Omz;		//Overdensity used to put galaxies in the halos
if(Lc < Lc_MAX)				//If the size of each cell was given compute the mass of each cell
	Mcell = rhomz*pow(Lc, 3.0);
else if(Lc > Mc_MIN){			//If the mass of each cell was given compute the size of each cell
	Mcell = Lc;
	Lc = pow(Mcell/rhomz, 1.0/3.0);
}
else{					//Notify an unexpected behavior and exit
	printf("A cell larger than %f [Mpc/h] or with a mass smaller than %e [M_odot/h] is not expected. Please, change this value or change the definition of Lc_MAX and Mc_MIN in the code.\n", Lc_MAX, Mc_MIN);
	exit(0);
}
Lx = Lc*nx;				//Compute the size of the box along the x-direction
Ly = Lc*ny;				//Compute the size of the box along the y-direction
Lz = Lc*nz;				//Compute the size of the box along the z-direction
Mtot = rhomz*Lx*Ly*Lz;			//Compute the total mass in the box
klx = 2.0*M_PI/Lx;			//Compute the fundamental frequency in the x-direction
kly = 2.0*M_PI/Ly;			//Compute the fundamental frequency in the y-direction
klz = 2.0*M_PI/Lz;			//Compute the fundamental frequency in the z-direction
Normx = 1.0/sqrt(Lx*Ly*Lz);		//Compute the normalization needed when aplyed the FFTW3 from k to x space
Normk = sqrt(Lx*Ly*Lz)/(nx*ny*nz);	//Compute the normalization needed when aplyed the FFTW3 from x to k space
nz2 = nz/2 + 1;				//Quantity used to alloc the complex arrays used in the FFTW3
nmin = nx;				//Determine the smaller direction
if(nmin > ny)	nmin = ny;
if(nmin > nz)	nmin = nz;

/*Compute the number of repetitions of this box to construct the light cone*/
float Pos[3], dist, cost, vr, Mass;
int Nrep_x, Nrep_y, Nrep_z;
if(out_halos == 2 || out_halos == 3){
	Nrep_x = floor(dist_max/Lx) + 1;
	Nrep_y = floor(dist_max/Ly) + 1;
	Nrep_z = floor(dist_max/Lz) + 1;
}

/*Parameters of the HOD model*/
int Ngals, Ncen, Nsat;
float r,  phi, theta, Rv, C, A;
float logMmin, siglogM, logM0, logM1, alpha;
logMmin = 12.44005264;
siglogM = 0.79560376;
logM0 = 11.98154109;
logM1 = 12.99600074;
alpha = 1.13717828;

/*Check some inputs before to start*/
if(out_inter == 0 && out_halos == 0){
	printf("You need to choose something to output! arg16, arg17 and/or arg18 must be >0!\n");
	exit(0);
}

if(nx<0 || ny<0 || nz<0){
	printf("You are trying to use n = (%d, %d, %d) and it is not possible!\n", nx, ny, nz);
	exit(0);
}

if(DO_2LPT < 0 || DO_2LPT >1){
	printf("You are trying to use DO_2LPT = %d and it is not possible! Setting DO_2LPT = 0.\n", DO_2LPT);
	DO_2LPT = 0;
}

if(Growth <= 0.0){
	printf("You gave a value of the ratio between the growths of %f and it is not physical!\n", Growth);
	exit(0);
}

if(Nmin < 0){
	printf("You gave a negative number for the number of particles in the smaller halo (%d). Settin it in 1.\n", Nmin);
	Nmin = 1;
}

if(N_cores < 0){
	printf("You gave a negative number for the number of cores (%d). Settin it in 1.\n", N_cores);
	N_cores = 1;
}

if(BARRIER != 0 && BARRIER != 1){
	printf("You need to chose a valid barrier for the void detection! Your choice were %d.\n", BARRIER);
	exit(0);
}

if(Om0>1.0 || Om0<0.0){
	printf("Your Omega _{m} = %f! Put some valid value between 0.0 and 1.0.\n", Om0);
	exit(0);
}

if(dc < 0.0){
	printf("Your delta_{c} = %f < 0. Using the fit.\n", dc);
	dc = 1.686*pow(Omz, 0.0055);
}

if(dc == 0.0)
	dc = 1.686*pow(Omz, 0.0055);

if(out_halos > 1 && theta_min > 1.0){
	printf("Theta min must be equal or smaller than 1! Setting it to 1.\n");
	theta_min = 1.0;
	cos_min = -1.0;
}

if(out_halos > 1 && LoS[0] == 0.0 && LoS[1] == 0.0 && LoS[2] == 0.0){
	printf("You must give a non vanishing vector for the direction of the line of sight!\n");
	exit(0);
}

if(out_collapse == 1 && out_halos < 2){
	printf("It is not possible to save the information about the collapsed particles without the creation of a light cone. Ignoring this parameter.\n");
	out_collapse = 0;
}

printf("\nRunning the ExSHalos!\n\
Omega_m = %.3f, z = %.3f, Growth = %.3f, H = %.2f, d_c = %.3f and Delta_virial = %.1f\n\
L = (%.5f, %.5f, %.5f), N_cells = (%d, %d, %d), M_tot = %.5e, M_cell = %.5e and seed = %d.\n", Omz, redshift, Growth, Hz, dc, Dv, Lx, Ly, Lz, nx, ny, nz, Mtot, Mcell, seed);


omp_set_num_threads(N_cores);	//Set the number of cores used by the openmp

/**************************************/
/*   Constructing the density grids   */
/**************************************/
printf("\nConstructing the density grid in real and fourier space!\n");

/*Opennning the power spectrum file*/
power = fopen(powerfile, "r");
if (power == NULL) {
	printf("Unable to open %s\n", powerfile);
	exit(0);
}

/*Measuring the number of k's*/
Nk = -1;
while(!feof(power)){
	fscanf(power, "%f %f", &kx, &ky);
	Nk ++;
}
rewind(power);

/*Reading the power spectrum*/
K = (double *)malloc(Nk*sizeof(double));
check_memory(K, "K")
P = (double *)malloc(Nk*sizeof(double));
check_memory(P, "P")
for(i=0;i<Nk;i++){
	fscanf(power, "%lf %lf", &K[i], &P[i]);
	P[i] = pow((double)Growth, 2.0)*P[i];
}
fclose(power);

/*Evaluating the Sigma(R)*/
Nr = Nk;
R = (double *)malloc(Nr*sizeof(double));
check_memory(R, "R")
M = (double *)malloc(Nr*sizeof(double));
check_memory(M, "M")
Sig = (double *)malloc(Nr*sizeof(double));
check_memory(Sig, "Sig")
Rmin = (double)pow(Mcell*0.9*3.0/(4.0*M_PI*rhomz), 1.0/3.0);
Rmax = (double)pow(M_max*3.0/(4.0*M_PI*rhomz), 1.0/3.0);
for(i=0;i<Nr;i++){
	R[i] = pow(10, log10(Rmin) + i*(log10(Rmax) - log10(Rmin))/(Nr-1));
	M[i] = 4.0/3.0*M_PI*(double)rhomz*pow(R[i], 3);
}
for(i=0;i<Nr;i++)
	Sig[i] = sqrt(calc_sigma(K, P, Nk, R[i]));

/*Interpolating the Sigma(M)*/
gsl_interp_accel *acc = gsl_interp_accel_alloc();
gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, Nr);
gsl_spline_init(spline, M, Sig, Nr);

/*Evaluate the integral of the mass function*/
double *Int;
Int = (double *)malloc(Nr*sizeof(double));
check_memory(Int, "Int")
Int[0] = 0.0;

for(i=1;i<Nr;i++)
	Int[i] = Int[i-1] - (log(Sig[i]) - log(Sig[i-1]))/2.0*(fh(Sig[i], 1, (double) dc)/pow(R[i], -3.0) + fh(Sig[i-1], 1, (double) dc)/pow(R[i-1], -3.0));

/*Interpolate the integral of the mass function as function of mass and its inverse*/
gsl_interp_accel *acc_I = gsl_interp_accel_alloc();
gsl_interp_accel *acc_InvI = gsl_interp_accel_alloc();
gsl_spline *spline_I = gsl_spline_alloc(gsl_interp_cspline, Nr);
gsl_spline *spline_InvI = gsl_spline_alloc(gsl_interp_cspline, Nr);
gsl_spline_init(spline_I, M, Int, Nr);
gsl_spline_init(spline_InvI, Int, M, Nr);
free(Int);

/*Compute the Sigma as function of the number of cells in the halo*/
Ncells = floor(M_max/Mcell);
Sig_grid = (float *)malloc(Ncells*sizeof(float));
check_memory(Sig_grid, "Sig_grid")
Sig_grid[0] = 1e+30;
for(i=1;i<Ncells;i++)
	Sig_grid[i] = pow(gsl_spline_eval(spline, i*Mcell, acc), 2.0);
gsl_spline_free(spline);
gsl_interp_accel_free(acc);
free(R);
free(M);
free(Sig);

/*Read the density grid*/
if(DEN_GRID == 0){	
		delta = (float*)fftwf_malloc((size_t)nx*(size_t)ny*(size_t)nz*sizeof(float));
		check_memory(delta, "delta")

		printf("Reading the density grid\n");
		den_grid = fopen(denfile, "rb");
		if (den_grid == NULL) {
			printf("Unable to open %s\n", denfile);
			exit(0);
		}

		fread(&nx, sizeof(int), 1, den_grid);
		fread(&ny, sizeof(int), 1, den_grid);
		fread(&nz, sizeof(int), 1, den_grid);
		fread(&Lc, sizeof(float), 1, den_grid);
		for(i=0;i<nx;i++)
			for(j=0;j<ny;j++)
				for(k=0;k<nz;k++){
					size_t ind = (size_t)(i*ny + j)*(size_t)nz + (size_t)k;
	
					fread(&delta[ind], sizeof(float), 1, den_grid);
					delta[ind] = Growth*delta[ind];
				}
		fclose(den_grid);
}

/*Construct the density grid*/
if(DEN_GRID == 1){

	/*Compute the Power spectrum in the box*/
	R_xi = (double *)malloc(Nk*sizeof(double));
	check_memory(R_xi, "R_xi")
	Xi = (double *)malloc(Nk*sizeof(double));
	check_memory(Xi, "Xi")
	pk2xi(Nk, K, P, R_xi, Xi);
	for(i=0;i<Nk;i++)
		if(R_xi[i] > (double)pow(Lx*Ly*Lz, 1.0/3.0)/2.0)	
			Xi[i] = 0.0;
	xi2pk(Nk, R_xi, Xi, K, P);
	free(R_xi);
	free(Xi);

	/*Interpolate the power spectrum*/
	acc = gsl_interp_accel_alloc();
	spline = gsl_spline_alloc(gsl_interp_cspline, Nk);
	gsl_spline_init(spline, K, P, Nk);
	free(K);
	free(P);

	/*Allocating the density grids*/
	delta = (float*)fftwf_malloc((size_t)nx*(size_t)ny*(size_t)nz*sizeof(float));
	check_memory(delta, "delta")
	deltak_tmp = (fftwf_complex *) fftwf_malloc((size_t)nx*(size_t)ny*(size_t)nz2*sizeof(fftwf_complex));
	check_memory(deltak_tmp, "deltak_tmp")
	deltak = (fftwf_complex *) fftwf_malloc((size_t)nx*(size_t)ny*(size_t)nz2*sizeof(fftwf_complex));
	check_memory(deltak, "deltak")

	/*Alloc the needed quantities for the random generator*/
	gsl_rng *rng_ptr;
	rng_ptr = gsl_rng_alloc (gsl_rng_taus);
	gsl_rng_set(rng_ptr, seed);

	/*Constructing the Fourier space density grid*/
	#pragma omp parallel for private(i, j, k, kx, ky, kz, kmod, std)
	for(i=0;i<nx;i++){
		if(2*i<nx) kx = (float)i*klx;
		else kx = (float)(i-nx)*klx;
	
		for(j=0;j<ny;j++){
			if(2*j<ny) ky = (float)j*kly;
			else ky = (float)(j-ny)*kly;
	
			for(k=0;k<nz2;k++){
				kz = (float)k*klz;
				if(k == nz/2)	kz = -(float)nz/2.0*klz;
	
				size_t ind = (size_t)(i*ny + j)*(size_t)nz2 + (size_t)k;
				kmod = sqrt(kx*kx + ky*ky + kz*kz);
	
				if(kmod == 0.0)	kmod = pow(klx*kly*klz, 1.0/3.0)/4.0;
				std = sqrt(gsl_spline_eval(spline, kmod, acc)/2.0);
	
				/*Generate Gaussian random number with std*/
				deltak[ind][0] = (float)gsl_ran_gaussian(rng_ptr, std); 
				deltak[ind][1] = (float)gsl_ran_gaussian(rng_ptr, std);
				deltak_tmp[ind][0] = deltak[ind][0];
				deltak_tmp[ind][1] = deltak[ind][1];
	
				if(isnan(deltak_tmp[ind][0]))	printf("Problem with deltak_tmp[%ld][0]\n", ind);
				if(isnan(deltak_tmp[ind][1]))	printf("Problem with deltak_tmp[%ld][1]\n", ind);
			}
		}
	}
	gsl_spline_free(spline);
	gsl_interp_accel_free(acc);
	gsl_rng_free (rng_ptr);

	/*Execute the FFTW3 to compute the density grid in real space*/
	p1 = fftwf_plan_dft_c2r_3d(nx, ny, nz, deltak_tmp, delta, FFTW_ESTIMATE); 
	fftwf_execute(p1);
	fftwf_free(deltak_tmp);

	/*Save the density grid*/
	if(out_inter == 1 || out_inter == 3){
		printf("Saving the density grid\n");
		den_grid = fopen(denfile, "wb");
		if (den_grid == NULL) {
			printf("Unable to open %s\n", denfile);
			exit(0);
		}

		fwrite(&nx, sizeof(int), 1, den_grid);
		fwrite(&ny, sizeof(int), 1, den_grid);
		fwrite(&nz, sizeof(int), 1, den_grid);
		fwrite(&Lc, sizeof(float), 1, den_grid);
		for(i=0;i<nx;i++)
			for(j=0;j<ny;j++)
				for(k=0;k<nz;k++){
					size_t ind = (size_t)(i*ny + j)*(size_t)nz + (size_t)k;
	
					fwrite(&delta[ind], sizeof(float), 1, den_grid);
				}
		fclose(den_grid);
	}
}

/*Compute the mean and std of the linear density field*/
kx = 0.0;
ky = 0.0;
for(i=0;i<nx;i++)
	for(j=0;j<ny;j++)
		for(k=0;k<nz;k++){
			size_t ind = (size_t)(i*ny + j)*(size_t)nz + (size_t)k;

			delta[ind] = delta[ind]*Normx;
			kx += delta[ind]*delta[ind];
			ky += delta[ind];
		}
kx = kx/((float)nx*(float)ny*(float)nz);
ky = ky/((float)nx*(float)ny*(float)nz);
printf("Mean = %f and Sigma = %f\n", ky, sqrt(kx - ky*ky));


/*************************/
/*   Finding the halos   */
/*************************/
if(out_halos != 0){
printf("\nFinding the spherical halos!\n");

/*Alloc the flag array*/
flag = (int *)malloc((size_t)nx*(size_t)ny*(size_t)nz*sizeof(int));
check_memory(flag, "flag")

/*Initialize the flag array*/
for(i=0;i<nx;i++)
	for(j=0;j<ny;j++)
		for(k=0;k<nz;k++){
			size_t ind = (size_t)(i*ny + j)*(size_t)nz + (size_t)k;
	
			flag[ind] = -1;
		}

/*Counting the number of peaks*/
np = 0;
for(i=0;i<nx;i++)
	for(j=0;j<ny;j++)
		for(k=0;k<nz;k++){
			size_t ind = (size_t)(i*ny + j)*(size_t)nz + (size_t)k;
			den = delta[ind];

			if(den > delta[(size_t)(cysum(i, 1, nx)*ny + j)*(size_t)nz + (size_t)k] && den > delta[(size_t)(cysum(i, -1, nx)*ny + j)*(size_t)nz + (size_t)k] && den > delta[(size_t)(i*ny + cysum(j, 1, ny))*(size_t)nz + (size_t)k] && den > delta[(size_t)(i*ny + cysum(j, -1, ny))*(size_t)nz + (size_t)k] && den > delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 1, nz)] && den > delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -1, nz)])
				np++;	
		}

/*Alloc the array with the peaks and final halos*/
peaks = (PEAKS *)malloc(np*sizeof(PEAKS));
halos = (HALOS *)malloc(np*sizeof(HALOS));
cont = 0;

/*Save the position and density of each peak*/
for(i=0;i<nx;i++)
	for(j=0;j<ny;j++)
		for(k=0;k<nz;k++){
			size_t ind = (size_t)(i*ny + j)*(size_t)nz + (size_t)k;
			den = delta[ind];

			if(den > delta[(size_t)(cysum(i, 1, nx)*ny + j)*(size_t)nz + (size_t)k] && den > delta[(size_t)(cysum(i, -1, nx)*ny + j)*(size_t)nz + (size_t)k] && den > delta[(size_t)(i*ny + cysum(j, 1, ny))*(size_t)nz + (size_t)k] && den > delta[(size_t)(i*ny + cysum(j, -1, ny))*(size_t)nz + (size_t)k] && den > delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 1, nz)] && den > delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -1, nz)]){
				peaks[cont].x[0] = i;
				peaks[cont].x[1] = j;
				peaks[cont].x[2] = k;
				peaks[cont].den = den;
				cont ++;
			}
		}

/*Check the new number of peaks and elements in the peaks array*/
if(cont != np){
	printf("The number of peaks does not match. %ld != %ld!\n", np, cont);
	exit(0);
}

/*Sort the peaks*/
quickSort_peaks(peaks, 0, np-1);

/*Grow the spherical halos around the density peaks*/
nh = 0;
printf("We have %ld peaks\n", np);
for(l=0;l<np;l++){

	/*If this peak is already in a halo jump to teh next one*/
	if(flag[(size_t)(peaks[l].x[0]*ny + peaks[l].x[1])*(size_t)nz + (size_t)peaks[l].x[2]] != -1)
		continue;

	/*Check if this peak is near to the slice used to construct the light cone*/
	if(out_halos == 2 || out_halos == 3){
		m = 1;
		for(i=-Nrep_x;i<=Nrep_x;i++)
			for(j=-Nrep_y;j<=Nrep_y;j++)
				for(k=-Nrep_z;k<=Nrep_z;k++){

					/*Compute the distance for this replic*/
					Pos[0] = (peaks[l].x[0] + 0.5)*Lc + Lx*i - Pobs[0];
					Pos[1] = (peaks[l].x[1] + 0.5)*Lc + Ly*j - Pobs[1];
					Pos[2] = (peaks[l].x[2] + 0.5)*Lc + Lz*k - Pobs[2];
					dist = 0.0;
					for(m=0;m<3;m++)
						dist += Pos[m]*Pos[m];
					dist = sqrt(dist);

					if(dist <= dist_min - Rmax || dist > dist_max + Rmax)	m = 0;

					/*Compute the angle theta*/		
					cost = 0.0;
					for(m=0;m<3;m++)
						cost += Pos[m]*LoS[m];
					cost = cost/dist;

					if(theta_min + Rmax/dist < M_PI && cost < cos(theta_min + Rmax/dist))	m = 0;
				}
		if(m == 0)	
			continue;
	}

	den = peaks[l].den;
	den_tmp = peaks[l].den;
	cont = 0;
	cont_tmp = 1;
	grows_tmp = 0;

	/*Grows the shells up to the minimum of the barrier*/
	while(den_tmp >= Barrier(Sig_grid[Ncells - 1], dc, BARRIER, EB_a, EB_b, EB_alpha)){
		if(cont < cont_tmp)	grows = grows_tmp;
		grows_tmp ++;
		den = den_tmp;
		cont = cont_tmp;
		den_tmp = den*(float)cont;
		tmp = floor(sqrt((double) grows_tmp));
		if(tmp > nmin/2)	tmp = nmin/2;

		for(i=-tmp;i<=tmp;i++)
			for(j=-tmp;j<=tmp;j++)
				for(k=-tmp;k<=tmp;k++)
					if(dist2(i, j, k) == grows_tmp){
						size_t ind = (size_t)(cysum(peaks[l].x[0], i, nx)*ny + cysum(peaks[l].x[1], j, ny))*(size_t)nz + (size_t)cysum(peaks[l].x[2], k, nz);

						if(flag[ind] != -1)
							den_tmp += -Mtot;
						else
							den_tmp += delta[ind];
						cont_tmp ++;						
					}

		den_tmp = den_tmp/(float)cont_tmp;
	}

	/*Decrease the shells up to the correct value of the barrier*/
	while(den < Barrier(Sig_grid[cont], dc, BARRIER, EB_a, EB_b, EB_alpha) && cont > 0){
		den_tmp = den;
		cont_tmp = cont;
		den = den*(float)cont;
		tmp = floor(sqrt((double) grows));
		if(tmp > nmin/2)	tmp = nmin/2;

		for(i=-tmp;i<=tmp;i++)
			for(j=-tmp;j<=tmp;j++)
				for(k=-tmp;k<=tmp;k++)
					if(dist2(i, j, k) == grows){
						size_t ind = (size_t)(cysum(peaks[l].x[0], i, nx)*ny + cysum(peaks[l].x[1], j, ny))*(size_t)nz + (size_t)cysum(peaks[l].x[2], k, nz);

						den -= delta[ind];
						cont --;						
					}

		if(cont > 0)	den = den/(float)cont;
		if(cont < cont_tmp)	grows_tmp = grows;

		grows --;
	}

	if(cont == 0)
		continue;

	/*Put the correct flags to the cells*/
	tmp = floor(sqrt((double) grows_tmp));
	for(i=-tmp;i<=tmp;i++)
		for(j=-tmp;j<=tmp;j++)
			for(k=-tmp;k<=tmp;k++)
				if(dist2(i, j, k) < grows_tmp){
					size_t ind = (size_t)(cysum(peaks[l].x[0], i, nx)*ny + cysum(peaks[l].x[1], j, ny))*(size_t)nz + (size_t)cysum(peaks[l].x[2], k, nz);

					if(flag[ind] != -1)
						printf("(1): This flag != -1! Flag = %d and the new one is %ld\n", flag[ind], nh);		

					flag[ind] = nh;
				}

	/*Save the halo information*/
	if(cont >= Nmin){
		halos[nh].cont = cont;
		for(i=0;i<3;i++)
			halos[nh].x[i] = peaks[l].x[i];
		nh ++;
	}
	else{
		for(i=-tmp;i<=tmp;i++)
			for(j=-tmp;j<=tmp;j++)
				for(k=-tmp;k<=tmp;k++)
					if(dist2(i, j, k) < grows_tmp){
						size_t ind = (size_t)(cysum(peaks[l].x[0], i, nx)*ny + cysum(peaks[l].x[1], j, ny))*(size_t)nz + (size_t)cysum(peaks[l].x[2], k, nz);
						flag[ind] = -2;
					}
	}
}
free(peaks);
free(Sig_grid);

/*Find the possible number of particles in a halo
sphere = (int *)malloc(Ncells*sizeof(int));
m = 0;
for(l=0;l<10000;l++){
	if(l%100 == 0)
		printf("l = %ld\n", l);

	tmp = floor(sqrt((float) l));
	cont = 0;
	
	for(i=-tmp;i<=tmp;i++)
		for(j=-tmp;j<=tmp;j++)
			for(k=-tmp;k<=tmp;k++)
				if(dist2(i, j, k) == l)
					cont ++;
	
	if(cont > 0){
		if(m > 0)	sphere[m] = sphere[m-1] + cont;
		else		sphere[m] = cont;

		m ++;
	}
}

/*Save this information
den_grid = fopen("Spheres.dat", "wb");
	if (den_grid == NULL) {
		printf("Unable to open spheres.dat\n");
		exit(0);
	}

fwrite(&m, sizeof(int), 1, den_grid);
for(i=0;i<m;i++)
	fwrite(&sphere[i], sizeof(int), 1, den_grid);
fclose(den_grid);*/

/*Read the data with the number of cells in each sphere*/
den_grid = fopen("Spheres.dat", "rb");
if (den_grid == NULL) {
	printf("Unable to open spheres.dat\n");
	exit(0);
}

fread(&m, sizeof(int), 1, den_grid);
sphere = (int *)malloc(m*sizeof(int));
for(i=0;i<m;i++)
	fread(&sphere[i], sizeof(int), 1, den_grid);
fclose(den_grid);


printf("We have %ld halos\n", nh);
}
/********************************/
/*   Displacing the particles   */
/********************************/
printf("\nDisplacing the particles using 1LPT!\n");

/*Define the arrays to store the final position, velocity and mass of each halo*/
float **velh, **posh, *Massh;
if(out_halos != 0){
	gsl_rng *rng_ptr;
	rng_ptr = gsl_rng_alloc (gsl_rng_taus);
	gsl_rng_set(rng_ptr, seed);

    Massh = (float *)malloc(nh*sizeof(float));
	velh = (float **)malloc(nh*sizeof(float *));
	posh = (float **)malloc(nh*sizeof(float *));
	for(i=0;i<nh;i++){
		velh[i] = (float *)malloc(3*sizeof(float));
		posh[i] = (float *)malloc(3*sizeof(float));
		for(j=0;j<3;j++){
			posh[i][j] = 0.0;
			velh[i][j] = 0.0;
		}

        cont = Next_Count(sphere, Ncells, halos[i].cont);

		den_tmp = gsl_spline_eval(spline_I, halos[i].cont*Mcell, acc_I) + (gsl_spline_eval(spline_I, sphere[cont]*Mcell, acc_I) - gsl_spline_eval(spline_I, halos[i].cont*Mcell, acc_I))*gsl_rng_uniform(rng_ptr);
		Massh[i] = gsl_spline_eval(spline_InvI, den_tmp, acc_InvI);
	}
    free(sphere);
}

/*Read the displacement field*/
if(DISP_CAT == 0){
	/*Open the output file for the displacement field*/
	printf("Reading the displacement field\n");
	disp_cat = fopen(dispfile, "rb");

	if (disp_cat == NULL) {
		printf("Unable to open %s\n", dispfile);
		exit(0);
	}
	
	fread(&nx, sizeof(int), 1, disp_cat);
	fread(&ny, sizeof(int), 1, disp_cat);
	fread(&nz, sizeof(int), 1, disp_cat);
	fread(&Lc, sizeof(float), 1, disp_cat);

	/*Read the displacement and add to each halo*/
	for(i=0;i<nx;i++)
		for(j=0;j<ny;j++)
			for(k=0;k<nz;k++){
				size_t ind = (size_t)(i*ny + j)*(size_t)nz + (size_t)k;
				
				if(DO_2LPT == 0){
					fread(&kx, sizeof(float), 1, disp_cat);
					fread(&ky, sizeof(float), 1, disp_cat);
					fread(&kz, sizeof(float), 1, disp_cat);
					
					if(out_halos != 0){
						tmp = flag[ind];
						if(tmp < 0)	continue;	

						posh[tmp][0] += Growth*kx;
						posh[tmp][1] += Growth*ky;
						posh[tmp][2] += Growth*kz;
	
						velh[tmp][0] += Growth*pow(Omz, 5.0/9.0)*Hz/(1.0 + redshift)*kx;
						velh[tmp][1] += Growth*pow(Omz, 5.0/9.0)*Hz/(1.0 + redshift)*ky;
						velh[tmp][2] += Growth*pow(Omz, 5.0/9.0)*Hz/(1.0 + redshift)*kz;
					}
				}
				else{
					fread(&kx, sizeof(float), 1, disp_cat);
					fread(&ky, sizeof(float), 1, disp_cat);
					fread(&kz, sizeof(float), 1, disp_cat);
					
					fread(&factx, sizeof(float), 1, disp_cat);
					fread(&facty, sizeof(float), 1, disp_cat);
					fread(&factz, sizeof(float), 1, disp_cat);

					if(out_halos != 0){

						tmp = flag[ind];
						if(tmp < 0)	continue;

						posh[tmp][0] += Growth*(kx - Growth*3.0/7.0*pow(Omz, -1.0/143)*factx);
						posh[tmp][1] += Growth*(ky - Growth*3.0/7.0*pow(Omz, -1.0/143)*facty);
						posh[tmp][2] += Growth*(kz - Growth*3.0/7.0*pow(Omz, -1.0/143)*factz);
	
						velh[tmp][0] += Growth*(pow(Omz, 5.0/9.0)*Hz/(1.0 + redshift)*kx); //- Growth*3.0/7.0*pow(Omz, -1.0/143)*2.0*pow(Omz, 6.0/11.0)*Hz/(1.0 + redshift)*factx);
						velh[tmp][1] += Growth*(pow(Omz, 5.0/9.0)*Hz/(1.0 + redshift)*ky); //- Growth*3.0/7.0*pow(Omz, -1.0/143)*2.0*pow(Omz, 6.0/11.0)*Hz/(1.0 + redshift)*facty);
						velh[tmp][2] += Growth*(pow(Omz, 5.0/9.0)*Hz/(1.0 + redshift)*kz); //- Growth*3.0/7.0*pow(Omz, -1.0/143)*2.0*pow(Omz, 6.0/11.0)*Hz/(1.0 + redshift)*factz);
					}
				}
			}
	fclose(disp_cat);
}

/*Compute the displacement field*/
if(DISP_CAT == 1){

	/*Define the arrays with the displacement field used in 2LPT*/
	float *S1, *S2, *S3;
	if(DO_2LPT == 1){
		S1 = (float *)malloc((size_t)nx*(size_t)ny*(size_t)nz*sizeof(float));
		S2 = (float *)malloc((size_t)nx*(size_t)ny*(size_t)nz*sizeof(float));
		S3 = (float *)malloc((size_t)nx*(size_t)ny*(size_t)nz*sizeof(float));
	}
	
	/*Alloc deltak*/
	if(DEN_GRID == 0){
		deltak = (fftwf_complex *) fftwf_malloc((size_t)nx*(size_t)ny*(size_t)nz2*sizeof(fftwf_complex));
		check_memory(deltak, "deltak")
	}

	/*Redefine the FFTW3 plan to compute the displacements*/
	fftwf_destroy_plan(p1);
	p1 = NULL;

	p1 = fftwf_plan_dft_c2r_3d(nx, ny, nz, deltak, delta, FFTW_ESTIMATE); 

	/*Divide the fourier space density by the green's function*/
	#pragma omp parallel for private(i, j, k, kx, ky, kz, factx, facty, factz, fact)
	for(i=0;i<nx;i++){
		if(2*i<nx) kx = i*klx;
		else kx = (i-nx)*klx;

		factx = 1.0/90.0*(2.0*cos(3.0*kx*Lc) - 27.0*cos(2.0*kx*Lc) + 270.0*cos(kx*Lc) - 245.0)/(Lc*Lc);
	
		for(j=0;j<ny;j++){
			if(2*j<ny) ky = j*kly;
			else ky = (j-ny)*kly;

			facty = 1.0/90.0*(2.0*cos(3.0*ky*Lc) - 27.0*cos(2.0*ky*Lc) + 270.0*cos(ky*Lc) - 245.0)/(Lc*Lc);
	
			for(k=0;k<nz2;k++){
				kz = k*klz;
				if(k == nz/2) kz = -(float)nz/2.0*klz;

				factz = 1.0/90.0*(2.0*cos(3.0*kz*Lc) - 27.0*cos(2.0*kz*Lc) + 270.0*cos(kz*Lc) - 245.0)/(Lc*Lc);

				size_t ind = (size_t)(i*ny + j)*(size_t)nz2 + (size_t)k;
				if(kx != 0.0 || ky != 0.0 || kz != 0.0){
					fact = factx + facty + factz;
					
					deltak[ind][0] = deltak[ind][0]/fact;
					deltak[ind][1] = deltak[ind][1]/fact;
				}
				else{
					deltak[ind][0] = 0.0;
					deltak[ind][1] = 0.0;
				}
			}
		}
	}

	/*Compute the potential at first order*/
	fftwf_execute(p1);

	/*Compute the first order displacements and update the position and velocity of each halo*/
	if(DO_2LPT == 1){
	#pragma omp parallel for private(i, j, k, tmp)
	for(i=0;i<nx;i++)
		for(j=0;j<ny;j++)
			for(k=0;k<nz;k++){
				size_t ind = (size_t)(i*ny + j)*(size_t)nz + (size_t)k;
	
				S1[ind] = -(1.0*delta[(size_t)(cysum(i, 3, nx)*ny + j)*(size_t)nz + (size_t)k] - 9.0*delta[(size_t)(cysum(i, 2, nx)*ny + j)*(size_t)nz + (size_t)k] + 45.0*delta[(size_t)(cysum(i, 1, nx)*ny + j)*(size_t)nz + (size_t)k] - 45.0*delta[(size_t)(cysum(i, -1, nx)*ny + j)*(size_t)nz + (size_t)k] + 9.0*delta[(size_t)(cysum(i, -2, nx)*ny + j)*(size_t)nz + (size_t)k] - 1.0*delta[(size_t)(cysum(i, -3, nx)*ny + j)*(size_t)nz + (size_t)k])*Normx/(60.0*Lc);
				S2[ind] = -(1.0*delta[(size_t)(i*ny + cysum(j, 3, ny))*(size_t)nz + (size_t)k] - 9.0*delta[(size_t)(i*ny + cysum(j, 2, ny))*(size_t)nz + (size_t)k] + 45.0*delta[(size_t)(i*ny + cysum(j, 1, ny))*(size_t)nz + (size_t)k] - 45.0*delta[(size_t)(i*nx + cysum(j, -1, ny))*(size_t)nz + (size_t)k] + 9.0*delta[(size_t)(i*ny + cysum(j, -2, ny))*(size_t)nz + (size_t)k] - 1.0*delta[(size_t)(i*ny + cysum(j, -3, ny))*(size_t)nz + (size_t)k])*Normx/(60.0*Lc);
				S3[ind] = -(1.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 3, nz)] - 9.0*delta[(size_t)(i*ny + j)*nz + (size_t)cysum(k, 2, nz)] + 45.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 1, nz)] - 45.0*delta[(size_t)(i*ny + j)*(size_t)nz + cysum(k, -1, nz)] + 9.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -2, nz)] - 1.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -3, nz)])*Normx/(60.0*Lc);

				if(out_halos != 0){
					tmp = flag[ind];
					if(tmp < 0)	continue;
	
					posh[tmp][0] += S1[ind];
					posh[tmp][1] += S2[ind];
					posh[tmp][2] += S3[ind];
	
					velh[tmp][0] += pow(Omz, 5.0/9.0)*Hz/(1.0 + redshift)*S1[ind];
					velh[tmp][1] += pow(Omz, 5.0/9.0)*Hz/(1.0 + redshift)*S2[ind];
					velh[tmp][2] += pow(Omz, 5.0/9.0)*Hz/(1.0 + redshift)*S3[ind];
				}
			}
	}

	else{
	/*Open the output file for the displacement field*/
	if(out_inter == 2 || out_inter == 3){
		printf("Saving the displaced particles\n");
		disp_cat = fopen(dispfile, "wb");

		if (disp_cat == NULL) {
			printf("Unable to open %s\n", dispfile);
			exit(0);
		}
	
		fwrite(&nx, sizeof(int), 1, disp_cat);
		fwrite(&ny, sizeof(int), 1, disp_cat);
		fwrite(&nz, sizeof(int), 1, disp_cat);
		fwrite(&Lc, sizeof(float), 1, disp_cat);
	}

	#pragma omp parallel for private(i, j, k, tmp, kx, ky, kz)
	for(i=0;i<nx;i++)
		for(j=0;j<ny;j++)
			for(k=0;k<nz;k++){
				size_t ind = (size_t)(i*ny + j)*(size_t)nz + (size_t)k;	

				/*save the displacement field*/
				if(out_inter == 2 || out_inter == 3){
	
					kx = -(1.0*delta[(size_t)(cysum(i, 3, nx)*ny + j)*(size_t)nz + (size_t)k] - 9.0*delta[(size_t)(cysum(i, 2, nx)*ny + j)*(size_t)nz + (size_t)k] + 45.0*delta[(size_t)(cysum(i, 1, nx)*ny + j)*(size_t)nz + (size_t)k] - 45.0*delta[(size_t)(cysum(i, -1, nx)*ny + j)*(size_t)nz + (size_t)k] + 9.0*delta[(size_t)(cysum(i, -2, nx)*ny + j)*(size_t)nz + (size_t)k] - 1.0*delta[(size_t)(cysum(i, -3, nx)*ny + j)*(size_t)nz + (size_t)k])*Normx/(60.0*Lc);
					ky = -(1.0*delta[(size_t)(i*ny + cysum(j, 3, ny))*(size_t)nz + (size_t)k] - 9.0*delta[(size_t)(i*ny + cysum(j, 2, ny))*(size_t)nz + (size_t)k] + 45.0*delta[(size_t)(i*ny + cysum(j, 1, ny))*(size_t)nz + (size_t)k] - 45.0*delta[(size_t)(i*nx + cysum(j, -1, ny))*(size_t)nz + (size_t)k] + 9.0*delta[(size_t)(i*ny + cysum(j, -2, ny))*(size_t)nz + (size_t)k] - 1.0*delta[(size_t)(i*ny + cysum(j, -3, ny))*(size_t)nz + (size_t)k])*Normx/(60.0*Lc);
					kz = -(1.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 3, nz)] - 9.0*delta[(size_t)(i*ny + j)*nz + (size_t)cysum(k, 2, nz)] + 45.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 1, nz)] - 45.0*delta[(size_t)(i*ny + j)*(size_t)nz + cysum(k, -1, nz)] + 9.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -2, nz)] - 1.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -3, nz)])*Normx/(60.0*Lc);

					fwrite(&kx, sizeof(float), 1, disp_cat);
					fwrite(&ky, sizeof(float), 1, disp_cat);
					fwrite(&kz, sizeof(float), 1, disp_cat);

					if(out_halos != 0){
						tmp = flag[ind];
						if(tmp < 0)	continue;
				
						posh[tmp][0] += kx;
						posh[tmp][1] += ky;
						posh[tmp][2] += kz;

						velh[tmp][0] += pow(Omz, 5.0/9.0)*Hz/(1.0 + redshift)*kx;
						velh[tmp][1] += pow(Omz, 5.0/9.0)*Hz/(1.0 + redshift)*ky;
						velh[tmp][2] += pow(Omz, 5.0/9.0)*Hz/(1.0 + redshift)*kz;
					}
				}

				/*Do not save the displacements*/
				else if(out_halos != 0){					
					tmp = flag[ind];
					if(tmp < 0)	continue;

					kx = -(1.0*delta[(size_t)(cysum(i, 3, nx)*ny + j)*(size_t)nz + (size_t)k] - 9.0*delta[(size_t)(cysum(i, 2, nx)*ny + j)*(size_t)nz + (size_t)k] + 45.0*delta[(size_t)(cysum(i, 1, nx)*ny + j)*(size_t)nz + (size_t)k] - 45.0*delta[(size_t)(cysum(i, -1, nx)*ny + j)*(size_t)nz + (size_t)k] + 9.0*delta[(size_t)(cysum(i, -2, nx)*ny + j)*(size_t)nz + (size_t)k] - 1.0*delta[(size_t)(cysum(i, -3, nx)*ny + j)*(size_t)nz + (size_t)k])*Normx/(60.0*Lc);
					ky = -(1.0*delta[(size_t)(i*ny + cysum(j, 3, ny))*(size_t)nz + (size_t)k] - 9.0*delta[(size_t)(i*ny + cysum(j, 2, ny))*(size_t)nz + (size_t)k] + 45.0*delta[(size_t)(i*ny + cysum(j, 1, ny))*(size_t)nz + (size_t)k] - 45.0*delta[(size_t)(i*nx + cysum(j, -1, ny))*(size_t)nz + (size_t)k] + 9.0*delta[(size_t)(i*ny + cysum(j, -2, ny))*(size_t)nz + (size_t)k] - 1.0*delta[(size_t)(i*ny + cysum(j, -3, ny))*(size_t)nz + (size_t)k])*Normx/(60.0*Lc);
					kz = -(1.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 3, nz)] - 9.0*delta[(size_t)(i*ny + j)*nz + (size_t)cysum(k, 2, nz)] + 45.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 1, nz)] - 45.0*delta[(size_t)(i*ny + j)*(size_t)nz + cysum(k, -1, nz)] + 9.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -2, nz)] - 1.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -3, nz)])*Normx/(60.0*Lc);
				
					posh[tmp][0] += kx;
					posh[tmp][1] += ky;
					posh[tmp][2] += kz;

					velh[tmp][0] += pow(Omz, 5.0/9.0)*Hz/(1.0 + redshift)*kx;
					velh[tmp][1] += pow(Omz, 5.0/9.0)*Hz/(1.0 + redshift)*ky;
					velh[tmp][2] += pow(Omz, 5.0/9.0)*Hz/(1.0 + redshift)*kz;
				}			
			}
	if(out_inter == 2 || out_inter == 3)
		fclose(disp_cat);
	}
	
	
	if(DO_2LPT == 1){
	printf("Displacing the particles using 2LPT!\n");

	/*Evaluating the second order contribution*/
	p2 = fftwf_plan_dft_r2c_3d(nx, ny, nz, delta, deltak, FFTW_ESTIMATE); 

	/*Compute the second order "density"*/
	#pragma omp parallel for private(i, j, k)
	for(i=0;i<nx;i++)
	       	for(j=0;j<ny;j++)
			for(k=0;k<nz;k++){
				phixx = (1.0*S1[(size_t)(cysum(i, 3, nx)*ny + j)*(size_t)nz + (size_t)k] - 9.0*S1[(size_t)(cysum(i, 2, nx)*ny + j)*(size_t)nz + (size_t)k] + 45.0*S1[(size_t)(cysum(i, 1, nx)*ny + j)*(size_t)nz + (size_t)k] - 45.0*S1[(size_t)(cysum(i, -1, nx)*ny + j)*(size_t)nz + (size_t)k] + 9.0*S1[(size_t)(cysum(i, -2, nx)*ny + j)*(size_t)nz + (size_t)k] - 1.0*S1[(size_t)(cysum(i, -3, nx)*ny + j)*(size_t)nz + (size_t)k])/(60.0*Lc);
				phixy = (1.0*S1[(size_t)(i*ny + cysum(j, 3, ny))*(size_t)nz + (size_t)k] - 9.0*S1[(size_t)(i*ny + cysum(j, 2, ny))*(size_t)nz + (size_t)k] + 45.0*S1[(size_t)(i*ny + cysum(j, 1, ny))*(size_t)nz + (size_t)k] - 45.0*S1[(size_t)(i*nx + cysum(j, -1, ny))*(size_t)nz + (size_t)k] + 9.0*S1[(size_t)(i*ny + cysum(j, -2, ny))*(size_t)nz + (size_t)k] - 1.0*S1[(size_t)(i*ny + cysum(j, -3, ny))*(size_t)nz + (size_t)k])/(60.0*Lc);
				phixz = (1.0*S1[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 3, nz)] - 9.0*S1[(size_t)(i*ny + j)*nz + (size_t)cysum(k, 2, nz)] + 45.0*S1[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 1, nz)] - 45.0*S1[(size_t)(i*ny + j)*(size_t)nz + cysum(k, -1, nz)] + 9.0*S1[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -2, nz)] - 1.0*S1[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -3, nz)])/(60.0*Lc);
				phiyy = (1.0*S2[(size_t)(i*ny + cysum(j, 3, ny))*(size_t)nz + (size_t)k] - 9.0*S2[(size_t)(i*ny + cysum(j, 2, ny))*(size_t)nz + (size_t)k] + 45.0*S2[(size_t)(i*ny + cysum(j, 1, ny))*(size_t)nz + (size_t)k] - 45.0*S2[(size_t)(i*nx + cysum(j, -1, ny))*(size_t)nz + (size_t)k] + 9.0*S2[(size_t)(i*ny + cysum(j, -2, ny))*(size_t)nz + (size_t)k] - 1.0*S2[(size_t)(i*ny + cysum(j, -3, ny))*(size_t)nz + (size_t)k])/(60.0*Lc);
				phiyz = (1.0*S2[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 3, nz)] - 9.0*S2[(size_t)(i*ny + j)*nz + (size_t)cysum(k, 2, nz)] + 45.0*S2[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 1, nz)] - 45.0*S2[(size_t)(i*ny + j)*(size_t)nz + cysum(k, -1, nz)] + 9.0*S2[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -2, nz)] - 1.0*S2[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -3, nz)])/(60.0*Lc);
				phizz = (1.0*S3[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 3, nz)] - 9.0*S3[(size_t)(i*ny + j)*nz + (size_t)cysum(k, 2, nz)] + 45.0*S3[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 1, nz)] - 45.0*S3[(size_t)(i*ny + j)*(size_t)nz + cysum(k, -1, nz)] + 9.0*S3[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -2, nz)] - 1.0*S3[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -3, nz)])/(60.0*Lc);

				delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)k] = 1.0*(phixx*phiyy + phixx*phizz + phiyy*phizz - pow(phixy, 2.0) - pow(phixz, 2.0) - pow(phiyz, 2.0));
			}

	/*Go to fourier space to solve the posson equation*/
	fftwf_execute(p2);

	/*Divide the fourier space density by the green's function*/
	#pragma omp parallel for private(i, j, k, kx, ky, kz, fact, factx, facty, factz)
	for(i=0;i<nx;i++){
		if(2*i<nx) kx = i*klx;
		else kx = (i-nx)*klx;
	
		factx = 1.0/90.0*(2.0*cos(3.0*kx*Lc) - 27.0*cos(2.0*kx*Lc) + 270.0*cos(kx*Lc) - 245.0)/(Lc*Lc);

		for(j=0;j<ny;j++){
			if(2*j<ny) ky = j*kly;
			else ky = (j-ny)*kly;

			facty = 1.0/90.0*(2.0*cos(3.0*ky*Lc) - 27.0*cos(2.0*ky*Lc) + 270.0*cos(ky*Lc) - 245.0)/(Lc*Lc);
	
			for(k=0;k<nz2;k++){
				kz = k*klz;
				if(k == nz/2) kz = -(float)nz/2.0*klz;

				factz = 1.0/90.0*(2.0*cos(3.0*kz*Lc) - 27.0*cos(2.0*kz*Lc) + 270.0*cos(kz*Lc) - 245.0)/(Lc*Lc);
			
				size_t ind = (size_t)(i*ny + j)*(size_t)nz2 + (size_t)k;
				if(kx != 0.0 || ky != 0.0 || kz != 0.0){
					fact = factx + facty + factz;

					deltak[ind][0] = deltak[ind][0]/fact*Normk;
					deltak[ind][1] = deltak[ind][1]/fact*Normk;
				}
				else{
					deltak[ind][0] = 0.0;
					deltak[ind][1] = 0.0;
				}
			}
		}
	}

	/*Come back to real space*/
	fftwf_execute(p1);

	/*Open the output file for the displacement field*/
	if(out_inter == 2 || out_inter == 3){
		printf("Saving the displaced particles\n");
		disp_cat = fopen(dispfile, "wb");

		if (disp_cat == NULL) {
			printf("Unable to open %s\n", dispfile);
			exit(0);
		}
	
		fwrite(&nx, sizeof(int), 1, disp_cat);
		fwrite(&ny, sizeof(int), 1, disp_cat);
		fwrite(&nz, sizeof(int), 1, disp_cat);
		fwrite(&Lc, sizeof(float), 1, disp_cat);
	}

	/*Compute the second order displacements and velocities*/
	#pragma omp parallel for private(i, j, k, kx, ky, kz, tmp)
	for(i=0;i<nx;i++)
		for(j=0;j<ny;j++)
			for(k=0;k<nz;k++){
				size_t ind = (size_t)(i*ny + j)*(size_t)nz + (size_t)k;	

				/*save the displacement field*/
				if(out_inter == 2 || out_inter == 3){
	
					kx = (1.0*delta[(size_t)(cysum(i, 3, nx)*ny + j)*(size_t)nz + (size_t)k] - 9.0*delta[(size_t)(cysum(i, 2, nx)*ny + j)*(size_t)nz + (size_t)k] + 45.0*delta[(size_t)(cysum(i, 1, nx)*ny + j)*(size_t)nz + (size_t)k] - 45.0*delta[(size_t)(cysum(i, -1, nx)*ny + j)*(size_t)nz + (size_t)k] + 9.0*delta[(size_t)(cysum(i, -2, nx)*ny + j)*(size_t)nz + (size_t)k] - 1.0*delta[(size_t)(cysum(i, -3, nx)*ny + j)*(size_t)nz + (size_t)k])*Normx/(60.0*Lc);
					ky = (1.0*delta[(size_t)(i*ny + cysum(j, 3, ny))*(size_t)nz + (size_t)k] - 9.0*delta[(size_t)(i*ny + cysum(j, 2, ny))*(size_t)nz + (size_t)k] + 45.0*delta[(size_t)(i*ny + cysum(j, 1, ny))*(size_t)nz + (size_t)k] - 45.0*delta[(size_t)(i*nx + cysum(j, -1, ny))*(size_t)nz + (size_t)k] + 9.0*delta[(size_t)(i*ny + cysum(j, -2, ny))*(size_t)nz + (size_t)k] - 1.0*delta[(size_t)(i*ny + cysum(j, -3, ny))*(size_t)nz + (size_t)k])*Normx/(60.0*Lc);
					kz = (1.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 3, nz)] - 9.0*delta[(size_t)(i*ny + j)*nz + (size_t)cysum(k, 2, nz)] + 45.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 1, nz)] - 45.0*delta[(size_t)(i*ny + j)*(size_t)nz + cysum(k, -1, nz)] + 9.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -2, nz)] - 1.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -3, nz)])*Normx/(60.0*Lc);

					fwrite(&S1[ind], sizeof(float), 1, disp_cat);
					fwrite(&S2[ind], sizeof(float), 1, disp_cat);
					fwrite(&S3[ind], sizeof(float), 1, disp_cat);

					fwrite(&kx, sizeof(float), 1, disp_cat);
					fwrite(&ky, sizeof(float), 1, disp_cat);
					fwrite(&kz, sizeof(float), 1, disp_cat);

					if(out_halos != 0){
						tmp = flag[ind];
						if(tmp < 0)	continue;

						kx = -3.0/7.0*pow(Omz, -1.0/143)*kx;
						ky = -3.0/7.0*pow(Omz, -1.0/143)*ky;
						kz = -3.0/7.0*pow(Omz, -1.0/143)*kz;
				
						posh[tmp][0] += kx;
						posh[tmp][1] += ky;
						posh[tmp][2] += kz;

						//velh[tmp][0] += 2.0*pow(Omz, 6.0/11.0)*Hz/(1.0 + redshift)*kx;
						//velh[tmp][1] += 2.0*pow(Omz, 6.0/11.0)*Hz/(1.0 + redshift)*ky;
						//velh[tmp][2] += 2.0*pow(Omz, 6.0/11.0)*Hz/(1.0 + redshift)*kz;
					}
				}

				/*Do not save the displacements*/
				else if(out_halos != 0){					
					tmp = flag[ind];
					if(tmp < 0)	continue;

					kx = (1.0*delta[(size_t)(cysum(i, 3, nx)*ny + j)*(size_t)nz + (size_t)k] - 9.0*delta[(size_t)(cysum(i, 2, nx)*ny + j)*(size_t)nz + (size_t)k] + 45.0*delta[(size_t)(cysum(i, 1, nx)*ny + j)*(size_t)nz + (size_t)k] - 45.0*delta[(size_t)(cysum(i, -1, nx)*ny + j)*(size_t)nz + (size_t)k] + 9.0*delta[(size_t)(cysum(i, -2, nx)*ny + j)*(size_t)nz + (size_t)k] - 1.0*delta[(size_t)(cysum(i, -3, nx)*ny + j)*(size_t)nz + (size_t)k])*Normx/(60.0*Lc);
					ky = (1.0*delta[(size_t)(i*ny + cysum(j, 3, ny))*(size_t)nz + (size_t)k] - 9.0*delta[(size_t)(i*ny + cysum(j, 2, ny))*(size_t)nz + (size_t)k] + 45.0*delta[(size_t)(i*ny + cysum(j, 1, ny))*(size_t)nz + (size_t)k] - 45.0*delta[(size_t)(i*nx + cysum(j, -1, ny))*(size_t)nz + (size_t)k] + 9.0*delta[(size_t)(i*ny + cysum(j, -2, ny))*(size_t)nz + (size_t)k] - 1.0*delta[(size_t)(i*ny + cysum(j, -3, ny))*(size_t)nz + (size_t)k])*Normx/(60.0*Lc);
					kz = (1.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 3, nz)] - 9.0*delta[(size_t)(i*ny + j)*nz + (size_t)cysum(k, 2, nz)] + 45.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, 1, nz)] - 45.0*delta[(size_t)(i*ny + j)*(size_t)nz + cysum(k, -1, nz)] + 9.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -2, nz)] - 1.0*delta[(size_t)(i*ny + j)*(size_t)nz + (size_t)cysum(k, -3, nz)])*Normx/(60.0*Lc);

					kx = -3.0/7.0*pow(Omz, -1.0/143)*kx;
					ky = -3.0/7.0*pow(Omz, -1.0/143)*ky;
					kz = -3.0/7.0*pow(Omz, -1.0/143)*kz;
				
					posh[tmp][0] += kx;
					posh[tmp][1] += ky;
					posh[tmp][2] += kz;

					//velh[tmp][0] += 2.0*pow(Omz, 6.0/11.0)*Hz/(1.0 + redshift)*kx;
					//velh[tmp][1] += 2.0*pow(Omz, 6.0/11.0)*Hz/(1.0 + redshift)*ky;
					//velh[tmp][2] += 2.0*pow(Omz, 6.0/11.0)*Hz/(1.0 + redshift)*kz;
				}			
			}

	if(out_inter == 2 || out_inter == 3)
		fclose(disp_cat);

	/*Free the FFTW memory*/
	fftwf_destroy_plan(p2);
	free(S1);
	free(S2);
	free(S3);
	}
	fftwf_destroy_plan(p1);
	fftwf_free(deltak);
}
fftwf_free(delta);
if(out_collapse == 0 && out_halos != 0)
	free(flag);

/*Compute the final position and velocity of the halos*/
if(out_halos != 0){
	for(i=0;i<nh;i++){
		posh[i][0] = cysumf(halos[i].x[0]*Lc + Lc/2.0, posh[i][0]/halos[i].cont, Lx);
		posh[i][1] = cysumf(halos[i].x[1]*Lc + Lc/2.0, posh[i][1]/halos[i].cont, Ly);
		posh[i][2] = cysumf(halos[i].x[2]*Lc + Lc/2.0, posh[i][2]/halos[i].cont, Lz);

		velh[i][0] = velh[i][0]/halos[i].cont;
		velh[i][1] = velh[i][1]/halos[i].cont;
		velh[i][2] = velh[i][2]/halos[i].cont;
	}
}

/*Saving the positions and velocities in real space*/
if(out_halos == 1){
	printf("Saving the halos\n");
	halo_cat = fopen(halofile, "w");

	if (halo_cat == NULL) {
		printf("Unable to open %s\n", halofile);
		exit(0);
	}

	fprintf(halo_cat, "%ld\n", nh);
	for(i=0;i<nh;i++){
		fprintf(halo_cat, "%f %f %f %f %f %f %e %d\n", posh[i][0], posh[i][1], posh[i][2], velh[i][0], velh[i][1], velh[i][2], Massh[i], halos[i].cont);
	}	
	fclose(halo_cat);
}

/*Putting galaxies in the halos*/
if(DO_HOD == 1){	
	printf("Saving the galaxies\n");
	sprintf(halofile, "%s_gals.dat", argv[14]);
	halo_cat = fopen(halofile, "w");

	if (halo_cat == NULL) {
		printf("Unable to open %s\n", halofile);
		exit(0);
	}

	cont = 0;
	for(i=0;i<nh;i++){
		/*Compute the number of central and satellite galaxies*/
		if(Ncentral(Massh[i], logMmin, siglogM) >= gsl_rng_uniform(rng_ptr))
			Ncen = 1;
		else
			Ncen = 0;		
		Nsat = gsl_ran_poisson(rng_ptr, (double) Nsatellite(Massh[i], logM0, logM1, alpha));
		Ngals = Ncen + Nsat;
		if(Ngals == 0) continue;

		/*Save the central galaxy*/
		if(Ncen == 1){
			fprintf(halo_cat, "%f %f %f %f %f %f %d\n", posh[i][0], posh[i][1], posh[i][2], velh[i][0], velh[i][1], velh[i][2], i);
			cont ++;
		}

		/*Put the satellite galaxies following the NFW profile*/
		if(Nsat > 0){
			Rv = pow(3.0*Massh[i]/(4.0*M_PI*Dv*rhom), 1.0/3.0);
			C = f_c(Massh[i], (float) Mstar, z);
			A = log(1.0 + C) - C/(1.0 + C);
		}

		for(j=0;j<Nsat;j++){
			phi = 2.0*M_PI*gsl_rng_uniform(rng_ptr);	
			theta = M_PI*gsl_rng_uniform(rng_ptr);

			r = Generate_NFW(Rv, C, A, seed);

			kx = cysumf(posh[i][0], r*sin(theta)*cos(phi), Lx); 
			ky = cysumf(posh[i][1], r*sin(theta)*sin(phi), Ly);
			kz = cysumf(posh[i][2], r*cos(theta), Lz);
			fprintf(halo_cat, "%f %f %f ", kx, ky, kz);

			kx = velh[i][0];
			ky = velh[i][1];
			kz = velh[i][2];
			fprintf(halo_cat, "%f %f %f %d\n", kx, ky, kz, i);

			cont ++;
		}
	}
	fclose(halo_cat);

	n_bar = cont/(Lx*Ly*Lz);
	printf("n_bar = %f\n", n_bar);
}

/********************************/
/*Put the halos in the lightcone*/
/********************************/
if(out_halos == 2 || out_halos == 3){
printf("\nPutting the halos in the light cone!\n");

printf("The code is using (%d, %d, %d) replicas to construct the light cone.\n", Nrep_x, Nrep_y, Nrep_z);
printf("This snapshot is in the range %f - %f [Mpc/h] with theta_min = %f.\n", dist_min, dist_max, theta_min);

/*Open the light cone file*/
light_cat = fopen(lightfile, "wb");

if (light_cat == NULL) {
	printf("Unable to open %s\n", lightfile);
	exit(0);
}
cont = 0;
fwrite(&cont, sizeof(long), 1, light_cat);

/*Run over all the halos and save then in the light cone file*/
for(l=0;l<nh;l++){
	for(i=-Nrep_x;i<=Nrep_x;i++)
		for(j=-Nrep_y;j<=Nrep_y;j++)
			for(k=-Nrep_z;k<=Nrep_z;k++){

				/*Compute the distance for this replic*/
				Pos[0] = posh[l][0] + Lx*i - Pobs[0];
				Pos[1] = posh[l][1] + Ly*j - Pobs[1];
				Pos[2] = posh[l][2] + Lz*k - Pobs[2];
				dist = 0.0;
				for(m=0;m<3;m++)
					dist += Pos[m]*Pos[m];
				dist = sqrt(dist);

				if(out_halos == 3){
					/*Compute the distance in redshift space*/
					vr = 0.0;
					for(m=0;m<3;m++)
						vr += velh[l][m]*Pos[m];
					vr = vr/dist;

					for(m=0;m<3;m++)
						Pos[m] = Pos[m] + vr/Hz*(1.0 + redshift)*Pos[m]/dist;
					dist = dist + vr/Hz*(1.0 + redshift);
				}

				if(dist <= dist_min || dist > dist_max)	continue;

				/*Compute the angle theta*/		
				cost = 0.0;
				for(m=0;m<3;m++)
					cost += Pos[m]*LoS[m];
				cost = cost/dist;

				if(cost < cos_min)	continue;

				/*Save the information about this halo*/
				fwrite(&Pos[0], sizeof(float), 1, light_cat);
				fwrite(&Pos[1], sizeof(float), 1, light_cat);
				fwrite(&Pos[2], sizeof(float), 1, light_cat);
				fwrite(&velh[l][0], sizeof(float), 1, light_cat);
				fwrite(&velh[l][1], sizeof(float), 1, light_cat);
				fwrite(&velh[l][2], sizeof(float), 1, light_cat);
				fwrite(&Massh[l], sizeof(float), 1, light_cat);
				cont ++;

				/*Put galaxies in this halo (one type)*/
				if(DO_HOD == 1){
					/*Compute the number of central and satellite galaxies*/
					if(Ncentral(Massh[l], logMmin, siglogM) >= gsl_rng_uniform(rng_ptr))
						Ncen = 1;
					else
						Ncen = 0;		
					Nsat = gsl_ran_poisson(rng_ptr, (double) Nsatellite(Massh[l], logM0, logM1, alpha));
					Ngals = Ncen + Nsat;

					/*Save the total number of galaxies*/
					fwrite(&Ngals, sizeof(int), 1, light_cat);

					/*Save the central galaxy*/
					fwrite(&Pos[0], sizeof(float), 1, light_cat);
					fwrite(&Pos[1], sizeof(float), 1, light_cat);
					fwrite(&Pos[2], sizeof(float), 1, light_cat);
					fwrite(&velh[l][0], sizeof(float), 1, light_cat);
					fwrite(&velh[l][1], sizeof(float), 1, light_cat);
					fwrite(&velh[l][2], sizeof(float), 1, light_cat);

					/*Put the satellite galaxies following the NFW profile*/
					if(Nsat > 0){
						Rv = pow(3.0*Massh[l]/(4.0*M_PI*Dv*rhom), 1.0/3.0);
						C = f_c(Massh[l], (float) Mstar, z);
						A = log(1.0 + C) - C/(1.0 + C);
					}

					for(m=0;m<Nsat;m++){
						phi = 2.0*M_PI*gsl_rng_uniform(rng_ptr);	
						theta = M_PI*gsl_rng_uniform(rng_ptr);

						r = Generate_NFW(Rv, C, A, seed);

						kx = cysumf(Pos[0], r*sin(theta)*cos(phi), Lx); 
						ky = cysumf(Pos[1], r*sin(theta)*sin(phi), Ly);
						kz = cysumf(Pos[2], r*cos(theta), Lz);
						fwrite(&kx, sizeof(float), 1, light_cat);
						fwrite(&ky, sizeof(float), 1, light_cat);
						fwrite(&kz, sizeof(float), 1, light_cat);

						kx = velh[i][0];
						ky = velh[i][1];
						kz = velh[i][2];
						fwrite(&kx, sizeof(float), 1, light_cat);
						fwrite(&ky, sizeof(float), 1, light_cat);
						fwrite(&kz, sizeof(float), 1, light_cat);
					}
				}
			}
}
rewind(light_cat);
fwrite(&cont, sizeof(long), 1, light_cat);
fclose(light_cat);

if(out_collapse == 1){

/*Open the file to save the information about the collapsed particles*/
collapse_cat = fopen(collapsefile, "wb");

if (collapse_cat == NULL) {
	printf("Unable to open %s\n", collapsefile);
	exit(0);
}

/*Save the information about the colapsed particles*/
int a, b, c;

cont = 0;
fwrite(&cont, sizeof(long), 1, collapse_cat);

#pragma omp parallel for private(i, j, k, a, b, c, Pos, dist, cost, tmp)
for(i=0;i<nx;i++)
	for(j=0;j<ny;j++)
		for(k=0;k<nz;k++){
			size_t ind = (size_t)(i*ny + j)*(size_t)nz + (size_t)k;	

			for(a=-Nrep_x;a<=Nrep_x;a++)
				for(b=-Nrep_y;b<=Nrep_y;b++)
					for(c=-Nrep_z;c<=Nrep_z;c++){

						/*Compute the distance for this replic*/
						Pos[0] = i*Lc + Lc/2.0 + Lx*a;
						Pos[1] = j*Lc + Lc/2.0 + Ly*b;
						Pos[2] = k*Lc + Lc/2.0 + Lz*c;
						dist = 0.0;
						for(m=0;m<3;m++)
							dist += Pos[m]*Pos[m];
						dist = sqrt(dist);

						if(dist <= dist_min || dist > dist_max)	continue;

						/*Compute the angle theta*/		
						cost = 0.0;
						for(m=0;m<3;m++)
							cost += Pos[m]*LoS[m];
						cost = cost/dist;

						if(cost < cos_min)	continue;

						tmp = flag[ind];
						cont ++;

						fwrite(&ind, sizeof(size_t), 1, collapse_cat);
						fwrite(&tmp, sizeof(int), 1, collapse_cat);
						fwrite(&redshift, sizeof(float), 1, collapse_cat);
					}
		}
rewind(collapse_cat);
fwrite(&cont, sizeof(long), 1, collapse_cat);
fclose(collapse_cat);

free(flag);
}
}

/*******************/
/* Free the memory */
/*******************/
gsl_spline_free(spline_I);
gsl_spline_free(spline_InvI);
gsl_interp_accel_free(acc_I);
gsl_interp_accel_free(acc_InvI);

if(out_halos != 0){
	free(Massh);
	for(i=0;i<nh;i++){
		free(velh[i]);
		free(posh[i]);
	}

	/*Free the rest*/
	free(velh);
	free(posh);
	free(halos);
	gsl_rng_free(rng_ptr);
}

return 0;
}
