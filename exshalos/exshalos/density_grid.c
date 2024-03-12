#include "density_grid.h"

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
                //delta[ind] = cosmo.Growth*delta[ind];
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
	gsl_spline_init(spline, K_tmp, P_tmp, Nk);

	free(R_xi);
	free(Xi);
    free(K_tmp);
    free(P_tmp);
}

/*Compute the Gaussian density grid*/
void Compute_Den(fft_real *K, fft_real *P, int Nk, fft_real R_max, fft_real *delta, fft_complex *deltak, int fixed, fft_real phase, fft_real k_smooth){
    int i, j, k;
    fft_real kx , ky, kz, kmod, std, theta, A;
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
	gsl_rng_set(rng_ptr, barrier.seed);

	/*Constructing the Fourier space density grid*/
	for(i=0;i<box.nd[0];i++){
		if(i<box.nd[0]/2) kx = i*box.kl[0];
		else kx = (i-box.nd[0])*box.kl[0];
	
		for(j=0;j<box.nd[1];j++){
			if(j<box.nd[1]/2) ky = j*box.kl[1];
			else ky = (j-box.nd[1])*box.kl[1];
	
            //#pragma omp parallel for private(k, ind, ind2, kmod, std) 
			for(k=0;k<box.nz2;k++){
				kz = k*box.kl[2];
				if(k == box.nd[2]/2)	kz = -box.nd[2]/2.0*box.kl[2];

                ind = (size_t)(i*box.nd[1] + j)*((size_t)box.nz2) + (size_t)k;

                if((k == 0 || k == box.nd[2]/2) && (i > box.nd[0]/2 || j > box.nd[1]/2)){
                    if(i > box.nd[0]/2 && j > box.nd[1]/2){
                        ind2 = (size_t)((box.nd[0] - i)*box.nd[1] + (box.nd[1] - j))*((size_t)box.nz2) + k;

                        deltak[ind][0] = deltak[ind2][0];
                        deltak[ind][1] = -deltak[ind2][1];
                        deltak_tmp[ind][0] = deltak[ind][0];
                        deltak_tmp[ind][1] = deltak[ind][1];

                        continue;
                    }
                    else if(i == 0 || i == box.nd[0]/2){
                        ind2 = (size_t)(i*box.nd[1] + (box.nd[1] - j))*((size_t)box.nz2) + k;

                        deltak[ind][0] = deltak[ind2][0];
                        deltak[ind][1] = -deltak[ind2][1];
                        deltak_tmp[ind][0] = deltak[ind][0];
                        deltak_tmp[ind][1] = deltak[ind][1];

                        continue;                        
                    }
                    else if(j == 0 || j == box.nd[1]/2){
                        ind2 = (size_t)((box.nd[0] - i)*box.nd[1] + j)*((size_t)box.nz2) + k;

                        deltak[ind][0] = deltak[ind2][0];
                        deltak[ind][1] = -deltak[ind2][1];
                        deltak_tmp[ind][0] = deltak[ind][0];
                        deltak_tmp[ind][1] = deltak[ind][1];

                        continue;                         
                    }
                    else if(i > box.nd[0]/2 && j < box.nd[1]/2){
                        ind2 = (size_t)((box.nd[0] - i)*box.nd[1] + (box.nd[1] - j))*((size_t)box.nz2) + k;

                        deltak[ind][0] = deltak[ind2][0];
                        deltak[ind][1] = -deltak[ind2][1];
                        deltak_tmp[ind][0] = deltak[ind][0];
                        deltak_tmp[ind][1] = deltak[ind][1];

                        continue;                        
                    }
                }
	
				kmod = (fft_real) sqrt(kx*kx + ky*ky + kz*kz);	

                /*Compute the amplitude of the field*/
                if(kmod <= k_smooth){
                    if(kmod > 0.0)
                        A = (fft_real) sqrt(gsl_spline_eval(spline, (double) kmod, acc)/2.0);             
                    else if(kmod == 0.0 && R_max < 100000.0){
                        kmod = (fft_real) pow(box.kl[0]*box.kl[1]*box.kl[2], 1.0/3.0)/4.0;
                        A = (fft_real) sqrt(gsl_spline_eval(spline, (double) kmod, acc)/2.0);
                    }
                    else
                        A = 0.0;
                }
                else
                    A = 0.0;
	
				/*Generate Gaussian random number with std*/
                if((i == 0 || i == box.nd[0]/2) && (j == 0 || j == box.nd[1]/2) && (k == 0 || k == box.nd[2]/2)){
                    if(fixed == TRUE)
                        deltak[ind][0] = sqrt(2.0)*A; 
                    else
                        deltak[ind][0] = sqrt(2.0)*A*((fft_real) gsl_ran_gaussian(rng_ptr, 1.0));;
                    deltak[ind][1] = 0.0;
                }
                else{
                    if(fixed == TRUE){
                        theta = 2.0*M_PI*((fft_real) gsl_ran_uniform(rng_ptr));
                        deltak[ind][0] = A*cos(theta); 
                        deltak[ind][1] = A*sin(theta);   
                    }
                    else{
                        deltak[ind][0] = A*((fft_real) gsl_ran_gaussian(rng_ptr, 1.0)); 
                        deltak[ind][1] = A*((fft_real) gsl_ran_gaussian(rng_ptr, 1.0));           
                    }
                }

                /*Add the constant phase*/
                if(phase != 0.0){
                    deltak[ind][0] = deltak[ind][0]*cos(phase) - deltak[ind][1]*sin(phase);
                    deltak[ind][1] = deltak[ind][0]*sin(phase) + deltak[ind][1]*cos(phase);
                }

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

/*Compute the density field to a given power*/
void Compute_Den_to_n(fft_real *delta, fft_real *delta_n, int n){
    size_t i;

    for(i=0;i<box.ng;i++)
        delta_n[i] = pow(delta[i], n);
}

/*Compute the potential field of a given density field*/
void Compute_Phi(fft_real *delta, fft_complex *deltak, fft_real *phi){
    int i, j, k, ink;
    size_t ind;
    fft_real kx, ky, kz, kmod, fact;
    FFTW(plan) p1;

    /*Create and compute deltak if it was not given*/
    if(deltak == NULL){
        fft_complex *deltak;
        ink = FALSE;

        /*Alloc the array for the density field in Fourier space*/
        deltak = (fft_complex *) FFTW(malloc)((size_t)box.nd[0]*(size_t)box.nd[1]*(size_t)box.nz2*sizeof(fft_complex));
        check_memory(deltak, "deltak")

        /*Compute the density grid in Fourier space*/
        Compute_Denk(delta, deltak);
    }
    else
        ink = TRUE;

	/*Compute the first order potential in Fourier space*/
	for(i=0;i<box.nd[0];i++){
		if(2*i<box.nd[0]) kx = i*box.kl[0];
		else kx = (i-box.nd[0])*box.kl[0];
	
		for(j=0;j<box.nd[1];j++){
			if(2*j<box.nd[1]) ky = j*box.kl[1];
			else ky = (j-box.nd[1])*box.kl[1];
	
			for(k=0;k<box.nz2;k++){
				kz = k*box.kl[2];
				if(k == box.nd[2]/2) kz = -(fft_real)box.nd[2]/2.0*box.kl[2];

				kmod = sqrt(pow(kx, 2.0) + pow(ky, 2.0) + pow(kz, 2.0));

				ind = (size_t)(i*box.nd[1] + j)*(size_t)box.nz2 + (size_t)k;
				if(kmod != 0.0){
					fact = -pow(kmod, 2.0);
					
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

	/*Compute the potential in real space*/
    p1 = FFTW(plan_dft_c2r_3d)(box.nd[0], box.nd[1], box.nd[2], deltak, phi, FFTW_ESTIMATE); 
	FFTW(execute)(p1);
    FFTW(destroy_plan)(p1);
    if(ink == FALSE)
        FFTW(free)(deltak);

    /*Normalize phi*/
    #pragma omp parallel for private(ind) 
    for(ind=0;ind<box.ng;ind++)
        phi[ind] = box.Normx*phi[ind];
}

/*Compute the tidal field*/
void Compute_Tidal(fft_real *delta, fft_complex *deltak, fft_real *tidal){
    int i, j, k, ink, ki, kj, count;
    size_t ind;
    fft_real kmod, kvec[3], fact;
    FFTW(plan) p1;
    fft_complex *deltak_tmp;

    /*Create and compute deltak if it was not given*/
    if(deltak == NULL){
        ink = FALSE;

        /*Alloc the array for the density field in Fourier space*/
        deltak = (fft_complex *) FFTW(malloc)((size_t)box.nd[0]*(size_t)box.nd[1]*(size_t)box.nz2*sizeof(fft_complex));
        check_memory(deltak, "deltak")

        /*Compute the density grid in Fourier space*/
        Compute_Denk(delta, deltak);

    }
    else
        ink = TRUE;

    /*Alloc the temporary array*/
    deltak_tmp = (fft_complex *) FFTW(malloc)((size_t)box.nd[0]*(size_t)box.nd[1]*(size_t)box.nz2*sizeof(fft_complex));
    check_memory(deltak_tmp, "deltak_tmp")

	/*Compute each component of the tidal field*/
    count = 0;
    for(ki=0;ki<3;ki++){
        for(kj=ki;kj<3;kj++){
            for(i=0;i<box.nd[0];i++){
                if(2*i<box.nd[0]) kvec[0] = i*box.kl[0];
                else kvec[0] = (i-box.nd[0])*box.kl[0];
            
                for(j=0;j<box.nd[1];j++){
                    if(2*j<box.nd[1]) kvec[1] = j*box.kl[1];
                    else kvec[1] = (j-box.nd[1])*box.kl[1];
            
                    for(k=0;k<box.nz2;k++){
                        kvec[2] = k*box.kl[2];
                        if(k == box.nd[2]/2) kvec[2] = -(fft_real)box.nd[2]/2.0*box.kl[2];

                        kmod = sqrt(pow(kvec[0], 2.0) + pow(kvec[1], 2.0) + pow(kvec[2], 2.0));
                        ind = (size_t)(i*box.nd[1] + j)*(size_t)box.nz2 + (size_t)k;

                        if(kmod != 0.0){
                            fact =  kvec[ki]*kvec[kj]/pow(kmod, 2.0);
                            
                            deltak_tmp[ind][0] = fact*deltak[ind][0];
                            deltak_tmp[ind][1] = fact*deltak[ind][1];
                        }
                        else{
                            deltak_tmp[ind][0] = 0.0;
                            deltak_tmp[ind][1] = 0.0;
                        }
                    }
                }
            }

            /*Compute the potential in real space*/
            p1 = FFTW(plan_dft_c2r_3d)(box.nd[0], box.nd[1], box.nd[2], deltak_tmp, &tidal[count*box.ng], FFTW_ESTIMATE); 
            FFTW(execute)(p1);

            /*Normalize phi*/
            #pragma omp parallel for private(ind) 
            for(ind=0;ind<box.ng;ind++)
                tidal[count*box.ng + ind] = box.Normx*tidal[count*box.ng + ind];

            count ++;
        }
    }

    /*Free the memory*/
    FFTW(destroy_plan)(p1);
    FFTW(free)(deltak_tmp);
    if(ink == FALSE)
        FFTW(free)(deltak);
}

/*Get the position in the flat array of the tidal field*/
int Get_IndK(int i, int j){
    int resp;

    if(j < i){
        resp = j;
        j = i;
        i = resp;
    }

    if(i == 0)
        resp = j;
    else if(i == 1)
        resp = 2 + j;
    else if(i == 2)
        resp = 5;

    return resp;
}

/*Compute K2 given the density field or the tidal field and the subtraction of the delta field (given by a):K2 =  K^2 - a*delta^2*/
void Compute_K2(fft_real *delta, fft_complex *deltak, fft_real *tidal, fft_real *K2, fft_real a){
    size_t ind;
    int intd;
    fft_real trace, K2_tmp;

    /*Create and compute the tidal field if it was not given*/
    if(tidal == NULL){
        intd = FALSE;

        /*Alloc the array for the tidal field*/
        tidal = (fft_real *) malloc(6*(size_t)box.nd[0]*(size_t)box.nd[1]*(size_t)box.nz2*sizeof(fft_real));
        check_memory(tidal, "tidal")

        /*Compute the density grid in Fourier space*/
        Compute_Tidal(delta, deltak, tidal);
    }
    else
        intd = TRUE;

    /*Compute K2*/
	for(ind=0;ind<box.ng;ind++){
        trace = tidal[0*box.ng + ind] + tidal[3*box.ng + ind] + tidal[5*box.ng + ind];
        K2_tmp = pow(tidal[0*box.ng + ind], 2.0) + pow(tidal[3*box.ng + ind], 2.0) + pow(tidal[5*box.ng + ind], 2.0) + 2.0*(pow(tidal[1*box.ng + ind], 2.0) + pow(tidal[2*box.ng + ind], 2.0) + pow(tidal[4*box.ng + ind], 2.0));

        K2[ind] = K2_tmp - a*pow(trace, 2.0);
    }

    /*Free the array with the tidal field if it was not give*/
    if(intd == FALSE)
        free(tidal);
}

/*Compute K3 given the density field or the tidal field*/
void Compute_K3(fft_real *delta, fft_complex *deltak, fft_real *tidal, fft_real *K3, fft_real a, fft_real b){
    size_t ind;
    int intd, i, j, k;
    fft_real trace, K2, K3_tmp;

    /*Create and compute the tidal field if it was not given*/
    if(tidal == NULL){
        intd = FALSE;

        /*Alloc the array for the tidal field*/
        tidal = (fft_real *) malloc(6*(size_t)box.nd[0]*(size_t)box.nd[1]*(size_t)box.nz2*sizeof(fft_real));
        check_memory(tidal, "tidal")

        /*Compute the density grid in Fourier space*/
        Compute_Tidal(delta, deltak, tidal);
    }
    else
        intd = TRUE;

    /*Compute K2*/
	for(ind=0;ind<box.ng;ind++){
        trace = tidal[0*box.ng + ind] + tidal[3*box.ng + ind] + tidal[5*box.ng + ind];
        K2 = pow(tidal[0*box.ng + ind], 2.0) + pow(tidal[3*box.ng + ind], 2.0) + pow(tidal[5*box.ng + ind], 2.0) + 2.0*(pow(tidal[1*box.ng + ind], 2.0) + pow(tidal[2*box.ng + ind], 2.0) + pow(tidal[4*box.ng + ind], 2.0));
        K3_tmp = pow(tidal[0*box.ng + ind], 3.0) + pow(tidal[3*box.ng + ind], 3.0) + pow(tidal[5*box.ng + ind], 3.0) + 3.0*(pow(tidal[1*box.ng + ind], 2.0)*(tidal[0*box.ng + ind] + tidal[3*box.ng + ind]) + pow(tidal[2*box.ng + ind], 2.0)*(tidal[0*box.ng + ind] + tidal[5*box.ng + ind]) + pow(tidal[4*box.ng + ind], 2.0)*(tidal[3*box.ng + ind] + tidal[5*box.ng + ind])) + 6.0*tidal[1*box.ng + ind]*tidal[2*box.ng + ind]*tidal[4*box.ng + ind];
            
        K3[ind] = K3_tmp - a*K2*trace + b*pow(trace, 3.0);
    }

    /*Free the array with the tidal field if it was not give*/
    if(intd == FALSE)
        free(tidal);
}