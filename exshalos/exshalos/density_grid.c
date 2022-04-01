#include "density_grid.h"

/*Window function in the Fourier space*/
fft_real W(fft_real k, fft_real R){
	fft_real resp;

	resp = 3.0/(pow(k*R,2))*(sin(k*R)/(k*R) - cos(k*R));
	return resp;
}

/*Read the density grid*/
void Read_Den(char *denfile, fft_real *delta){
    FILE *den_grid;
    int i, j, k, nx, ny, nz;
    fft_real Lc;
    size_t ind;

    /*Open the density grid file*/
    den_grid = fopen(denfile, "rb");
    if(den_grid == NULL){
        printf("Unable to open %s\n", denfile);
        exit(0);
    }

    fread(&nx, sizeof(int), 1, den_grid);
    fread(&ny, sizeof(int), 1, den_grid);
    fread(&nz, sizeof(int), 1, den_grid);
    fread(&Lc, sizeof(fft_real), 1, den_grid);
    for(i=0;i<nx;i++)
        for(j=0;j<ny;j++)
            for(k=0;k<nz;k++){
                ind = (size_t)(i*ny + j)*(size_t)nz + (size_t)k;

                fread(&delta[ind], sizeof(fft_real), 1, den_grid);
                delta[ind] = cosmo.Growth*delta[ind];
            }
    fclose(den_grid);
}

/*Interpole the power spectrum*/
void Inter_Power(fft_real *K, fft_real *P, int Nk, fft_real R_max, gsl_spline *spline){
    int i;
    double *K_tmp, *P_tmp, *R_xi, *Xi;

	/*Alloc the arrays*/
	R_xi = (double *)malloc(Nk*sizeof(double));
	check_memory(R_xi, "R_xi")
	Xi = (double *)malloc(Nk*sizeof(double));
	check_memory(Xi, "Xi")
    K_tmp = (double *)malloc(Nk*sizeof(double));
	check_memory(K_tmp, "K_tmp")
	P_tmp = (double *)malloc(Nk*sizeof(double));
	check_memory(P_tmp, "P_tmp")

    for(i=0;i<Nk;i++){
        K_tmp[i] = (double) K[i];
        P_tmp[i] = (double) P[i];
    }
  
    /*Compute the Power spectrum in the box*/
    if(R_max < 100000.0){
        pk2xi(Nk, K_tmp, P_tmp, R_xi, Xi);
        for(i=0;i<Nk;i++)
            if(R_xi[i] > R_max)	
                Xi[i] = 0.0;
        xi2pk(Nk, R_xi, Xi, K_tmp, P_tmp);
    }

	/*Interpolate the power spectrum*/
	gsl_spline_init(spline,  K_tmp, P_tmp, Nk);

	free(R_xi);
	free(Xi);
    free(K_tmp);
    free(P_tmp);
}

/*Compute the Gaussian density grid*/
void Compute_Den(fft_real *K, fft_real *P, int Nk, fft_real R_max, fft_real *delta, fft_complex *deltak, int seed){
    int i, j, k;
    fft_real kx , ky, kz, kmod, std;
    size_t ind, ind2;
    fft_complex *deltak_tmp;
    FFTW(plan) p1;

    /*Interpolate the power spectrum*/
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, Nk);

	Inter_Power(K, P, Nk, R_max, spline);

	/*Allocating the density grids*/
	deltak_tmp = (fft_complex *) FFTW(malloc)((size_t)box.nd[0]*(size_t)box.nd[1]*(size_t)box.nz2*sizeof(fft_complex));
	check_memory(deltak_tmp, "deltak_tmp")

	/*Alloc the needed quantities for the random generator*/
	gsl_rng *rng_ptr;
	rng_ptr = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(rng_ptr, seed);

	/*Constructing the Fourier space density grid*/
	for(i=0;i<box.nd[0];i++){
		if(i<box.nd[0]/2) kx = i*box.kl[0];
		else kx = (i-box.nd[0])*box.kl[0];
	
		for(j=0;j<box.nd[1];j++){
			if(j<box.nd[1]/2) ky = j*box.kl[1];
			else ky = (j-box.nd[1])*box.kl[1];
	
            #pragma omp parallel for private(k, ind, ind2, kmod, std) 
			for(k=0;k<box.nz2;k++){
				kz = k*box.kl[2];
				if(k == box.nd[2]/2)	kz = -box.nd[2]/2.0*box.kl[2];

                ind = (size_t)(i*box.nd[1] + j)*((size_t)box.nz2) + (size_t)k;
                
                if(k == 0 && j == 0 && i > box.nd[0]/2){
                    ind2 = (size_t)(box.nd[0] - i)*box.nd[1]*((size_t)box.nz2);

                    deltak[ind][0] = deltak[ind2][0];
                    deltak[ind][1] = deltak[ind2][1];
                    deltak_tmp[ind][0] = deltak[ind][0];
                    deltak_tmp[ind][1] = deltak[ind][1];

                    continue;
                }

                if(k == 0 && j > box.nd[1]/2){
                    ind2 = (size_t)(i*box.nd[1] + (box.nd[1] - j))*((size_t)box.nz2);

                    deltak[ind][0] = deltak[ind2][0];
                    deltak[ind][1] = deltak[ind2][1];
                    deltak_tmp[ind][0] = deltak[ind][0];
                    deltak_tmp[ind][1] = deltak[ind][1];

                    continue;
                }
	
				kmod = (fft_real) sqrt(kx*kx + ky*ky + kz*kz);	
				if(kmod == 0.0)	kmod = (fft_real) pow(box.kl[0]*box.kl[1]*box.kl[2], 1.0/3.0)/4.0;
				std = (fft_real) sqrt(gsl_spline_eval(spline, (double) kmod, acc)/2.0);
	
				/*Generate Gaussian random number with std*/
				deltak[ind][0] = (fft_real)gsl_ran_gaussian(rng_ptr, (double) std); 
				deltak[ind][1] = (fft_real)gsl_ran_gaussian(rng_ptr, (double) std);
				deltak_tmp[ind][0] = deltak[ind][0];
				deltak_tmp[ind][1] = deltak[ind][1];
	
				if(isnan(deltak_tmp[ind][0]))	printf("Problem with deltak_tmp[%ld][0]\n", ind);
				if(isnan(deltak_tmp[ind][1]))	printf("Problem with deltak_tmp[%ld][1]\n", ind);
			}
		}
	}

	gsl_rng_free(rng_ptr);
    gsl_interp_accel_free(acc);
    gsl_spline_free(spline);

	/*Execute the FFTW3 to compute the density grid in real space*/
	p1 = FFTW(plan_dft_c2r_3d)(box.nd[0], box.nd[1], box.nd[2], deltak_tmp, delta, FFTW_ESTIMATE); 
	FFTW(execute)(p1);
	FFTW(free)(deltak_tmp);
    FFTW(destroy_plan)(p1);

    /*Correct the values of delta*/
    #pragma omp parallel for private(ind) 
    for(ind=0;ind<box.ng;ind++)
	    delta[ind] = delta[ind]*box.Normx;
}

/*Compute deltak given delta*/
void Compute_Denk(fft_real *delta, fft_complex *deltak){
    size_t i;
    FFTW(plan) p1;

    /*Compute the FFT*/
    p1 = FFTW(plan_dft_r2c_3d)(box.nd[0], box.nd[1], box.nd[2], delta, deltak, FFTW_ESTIMATE); 
	FFTW(execute)(p1);
    FFTW(destroy_plan)(p1);

    /*Normalize deltak*/
    #pragma omp parallel for private(i) 
    for(i=0;i<(size_t)box.nd[0]*(size_t)box.nd[1]*(size_t)box.nz2;i++){
        deltak[i][0] = box.Normk*deltak[i][0];
        deltak[i][1] = box.Normk*deltak[i][1];
    }
}

/*Save the density field*/
void Save_Den(char *denfile, fft_real *delta){
    size_t i;
    FILE *den_grid;

    /*Open the density grid file*/
    den_grid = fopen(denfile, "wb");
    if (den_grid == NULL){
        printf("Unable to open %s\n", denfile);
        exit(0);
    }

    /*Save the density grid*/
    fwrite(&box.nd[0], sizeof(int), 1, den_grid);
    fwrite(&box.nd[1], sizeof(int), 1, den_grid);
    fwrite(&box.nd[2], sizeof(int), 1, den_grid);
    fwrite(&box.Lc, sizeof(fft_real), 1, den_grid);
    for(i=0;i<box.ng;i++)
        fwrite(&delta[i], sizeof(fft_real), 1, den_grid);
    fclose(den_grid);
}

/*Compute the mean and std of the density field*/
void Compute_MS(fft_real *delta){
    size_t i;
    double mean, std;

    mean = 0.0;
    std = 0.0;
    #pragma omp parallel for private(i) 
    for(i=0;i<box.ng;i++){
        std += (double) delta[i]*delta[i];
        mean += (double) delta[i];
    }
    mean = mean/((double)box.nd[0]*(double)box.nd[1]*(double)box.nd[2]);
    std = std/((double)box.nd[0]*(double)box.nd[1]*(double)box.nd[2]);

    printf("Mean = %lf and Sigma = %lf\n", mean, sqrt(std - mean*mean));
}
