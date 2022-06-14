#include "find_halos.h"

/*Evaluate the square root of matter variance*/
fft_real calc_sigma(fft_real *k, fft_real *P, int Nk, fft_real R){
	int i;
	fft_real resp;

	resp = 0.0;
	for(i=0;i<Nk-1;i++)
		resp += (k[i+1] - k[i])/2.0*(P[i]*pow(k[i]*W(k[i],R), 2) + P[i+1]*pow(k[i+1]*W(k[i+1],R), 2));

	return resp/(2.0*M_PI*M_PI);
}

/*Compute sigma(M) as function of the number of cells*/
void Compute_Sig(int Nr, fft_real *R, double *M, double *Sig, fft_real *Sig_grid, fft_real *K, fft_real *P, int Nk){
    int i, Ncells;
    fft_real Rmin, Rmax;

    /*Define the values of R and M*/
    Rmin = (fft_real)pow(box.Mcell*0.9*3.0/(4.0*M_PI*cosmo.rhomz), 1.0/3.0);
    Rmax = (fft_real)pow(M_max*3.0/(4.0*M_PI*cosmo.rhomz), 1.0/3.0);
    for(i=0;i<Nr;i++){
        R[i] = pow(10, log10(Rmin) + i*(log10(Rmax) - log10(Rmin))/(Nr-1));
        M[i] = (double) 4.0/3.0*M_PI*((fft_real)cosmo.rhomz*pow(R[i], 3));
    }
    
    /*Evaluating the Sigma(R)*/
    for(i=0;i<Nr;i++)
        Sig[i] = (double) sqrt(calc_sigma(K, P, Nk, R[i]));

    /*Interpolate the Sigma(M)*/
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, Nr);
    gsl_spline_init(spline, M, Sig, Nr);

    /*Compute the Sigma as function of the number of cells in the halo*/
    Ncells = floor(M_max/box.Mcell);
    Sig_grid[0] = 1e+30;
    for(i=1;i<Ncells;i++)
        Sig_grid[i] = pow(gsl_spline_eval(spline, (double) i*box.Mcell, acc), 2.0);
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
}

/*Evaluate the mass function for a given sigma*/
fft_real fh(fft_real sigma, int model){
	fft_real resp, nu;
	fft_real B, d, e, f, g;

	//Press-Schechter
	if(model == 0){
		nu = cosmo.dc/sigma;
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

/*Compute the integral over the mass function and interpolate it*/
void Compute_nh(int model, int Nr, fft_real *R, double *M, double *Sig, gsl_spline *spline_I, gsl_spline *spline_InvI){
    double *Int;
    int i;

    /*Alloc the arrays*/
    Int = (double *)malloc(Nr*sizeof(double));
    check_memory(Int, "Int")
    Int[0] = 0.0;

    /*Compute the mass function*/
    for(i=1;i<Nr;i++){
        Int[i] = Int[i-1] - (double) (log(Sig[i]) - log(Sig[i-1]))/2.0*(fh(Sig[i], model)/pow(R[i], 3.0) + fh(Sig[i-1], model)/pow(R[i-1], 3.0));
        if(Int[i] == Int[i-1])  Int[i] = Int[i-1]*1.000001;
    }

    /*Interpolate the integral of the mass function as function of mass and its inverse*/
    gsl_spline_init(spline_I, M, Int, Nr);
    gsl_spline_init(spline_InvI, Int, M, Nr);

    /*Free the array*/
    free(Int);
}

/*Check if the current position is a peak of the density grid*/
char Check_Peak(fft_real *delta, fft_real den, int i, int j, int k){
    char resp;

    if(den > delta[(size_t)(cysum(i, 1, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k] && den > delta[(size_t)(cysum(i, -1, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k] && den > delta[(size_t)(i*box.nd[1] + cysum(j, 1, box.nd[1]))*(size_t)box.nd[2] + (size_t)k] && den > delta[(size_t)(i*box.nd[1] + cysum(j, -1, box.nd[1]))*(size_t)box.nd[2] + (size_t)k] && den > delta[(size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, 1, box.nd[2])] && den > delta[(size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, -1, box.nd[2])])
        resp = TRUE;
    else
        resp = FALSE;

    return resp;
}

/*Count the number of peaks*/
size_t Count_Peaks(fft_real *delta){
    int i, j, k; 
    size_t np, ind;
    fft_real den;

    /*Counting the number of peaks*/
    np = 0;
    for(i=0;i<box.nd[0];i++)
        for(j=0;j<box.nd[1];j++)
            for(k=0;k<box.nd[2];k++){
                ind = (size_t)(i*box.nd[1] + j)*((size_t)box.nd[2]) + (size_t)k;
                den = delta[ind];

                if(den > delta[(size_t)(cysum(i, 1, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k] && den > delta[(size_t)(cysum(i, -1, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k] && den > delta[(size_t)(i*box.nd[1] + cysum(j, 1, box.nd[1]))*(size_t)box.nd[2] + (size_t)k] && den > delta[(size_t)(i*box.nd[1] + cysum(j, -1, box.nd[1]))*(size_t)box.nd[2] + (size_t)k] && den > delta[(size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, 1, box.nd[2])] && den > delta[(size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, -1, box.nd[2])])
                    np++;	
            }

    return np;
}

/*Save the positions and density of each peak*/
void Find_Peaks(fft_real *delta, size_t np, PEAKS *peaks){
    int i, j, k; 
    size_t ind, cont;
    fft_real den;

    /*Save the position and density of each peak*/ 
    cont = 0;
    for(i=0;i<box.nd[0];i++)
        for(j=0;j<box.nd[1];j++)
            for(k=0;k<box.nd[2];k++){
                ind = (size_t)(i*box.nd[1] + j)*((size_t)box.nd[2]) + (size_t)k;
                den = delta[ind];

                if(den > delta[(size_t)(cysum(i, 1, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k] && den > delta[(size_t)(cysum(i, -1, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k] && den > delta[(size_t)(i*box.nd[1] + cysum(j, 1, box.nd[1]))*(size_t)box.nd[2] + (size_t)k] && den > delta[(size_t)(i*box.nd[1] + cysum(j, -1, box.nd[1]))*(size_t)box.nd[2] + (size_t)k] && den > delta[(size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, 1, box.nd[2])] && den > delta[(size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, -1, box.nd[2])]){
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
}

/*function to swap elements*/
void swap_peaks(PEAKS *a, PEAKS *b){
    int k;
    PEAKS t;

    t.den = (*a).den;
    (*a).den = (*b).den;
    (*b).den = t.den;

    for(k=0;k<3;k++){
        t.x[k] = (*a).x[k];
        (*a).x[k] = (*b).x[k];
        (*b).x[k] = t.x[k];
    }
}

/*Partition function for the quicksort*/
long long partition_peaks(PEAKS *array, long long low, long long high){
    fft_real pivot = array[high].den;
    long long i = (low - 1);

    for (long long j = low; j < high; j++){
        if (array[j].den <= pivot){
            i++;
            swap_peaks(&array[i], &array[j]);
        }
    }
    swap_peaks(&array[i + 1], &array[high]);
  
    return (i + 1);
}

/*The quicksort algorithm to sort the peaks list*/
void quickSort_peaks(PEAKS *array, long long low, long long high){   
    if (low < high) {
        long long pi = partition_peaks(array, low, high);
        quickSort_peaks(array, low, pi - 1);
        quickSort_peaks(array, pi + 1, high);
    }
}

/*Barrier used for the halo definition*/
fft_real Barrier(fft_real S){
	fft_real resp;

	/*The Press-Schechter barrier*/
	if(out.DO_EB == 0)
		resp = cosmo.dc;

	/*The Sheth-Tormen barrier*/
	else if(out.DO_EB == 1)
		resp = sqrt(barrier.a)*cosmo.dc*(1.0 + barrier.beta*pow(S/(barrier.a*cosmo.dc*cosmo.dc), barrier.alpha));

	return resp;
}

/*It grows the spheres around the peaks to create the halos*/
size_t Grow_Halos(size_t np, size_t *flag, fft_real *Sig_Grid, fft_real *delta, PEAKS *peaks, HALOS *halos){
    int i, j, k, m, count, count_tmp, grows, grows_tmp, tmp, Ncells = floor(M_max/box.Mcell);
    size_t l, nh, ind;
    fft_real dist, Rmax, cost, den, den_tmp, Pos[3];

    /*Run over all peaks*/
    nh = 0;
    for(l=np-1;l>0;l--){

        /*If this peak (and the nexts) are below the minimum of the barrier stop finding halos*/
        if(peaks[l].den < Barrier(Sig_Grid[Ncells - 1]))
            break;

        /*If this peak is already in a halo jump to the next one*/
        if(flag[(size_t)(peaks[l].x[0]*box.nd[1] + peaks[l].x[1])*(size_t)box.nd[2] + (size_t)peaks[l].x[2]] != box.ng)
            continue;

        /*Check if this peak is near to the slice used to construct the light cone*/
        if(out.OUT_HALOS == 2 || out.OUT_HALOS == 3){
            Rmax = (double)pow(M_max*3.0/(4.0*M_PI*cosmo.rhomz), 1.0/3.0);

            m = 1;
            for(i=-lightcone.Nrep[0];i<=lightcone.Nrep[0];i++)
                for(j=-lightcone.Nrep[1];j<=lightcone.Nrep[1];j++)
                    for(k=-lightcone.Nrep[2];k<=lightcone.Nrep[2];k++){

                        /*Compute the distance for this replic*/
                        Pos[0] = (peaks[l].x[0] + 0.5)*box.Lc + box.L[0]*i - lightcone.Pobs[0];
                        Pos[1] = (peaks[l].x[1] + 0.5)*box.Lc + box.L[1]*j - lightcone.Pobs[1];
                        Pos[2] = (peaks[l].x[2] + 0.5)*box.Lc + box.L[2]*k - lightcone.Pobs[2];
                        dist = 0.0;
                        for(m=0;m<3;m++)
                            dist += Pos[m]*Pos[m];
                        dist = sqrt(dist);

                        if(dist <= lightcone.dist_min - Rmax || dist > lightcone.dist_max + Rmax)	m = 0;

                        /*Compute the angle theta*/		
                        cost = 0.0;
                        for(m=0;m<3;m++)
                            cost += Pos[m]*lightcone.LoS[m];
                        cost = cost/dist;

                        if(lightcone.theta_max + Rmax/dist < M_PI && cost < cos(lightcone.theta_max + Rmax/dist))	m = 0;
                    }
            if(m == 0)	
                continue;
        }

        den = peaks[l].den;
        den_tmp = peaks[l].den;
        count = 0;
        count_tmp = 1;
        grows_tmp = 0;

        /*Grows the shells up to the minimum of the barrier*/
        while(den_tmp >= Barrier(Sig_Grid[Ncells - 1])){
            if(count < count_tmp)	grows = grows_tmp;
            grows_tmp ++;
            den = den_tmp;
            count = count_tmp;
            den_tmp = den*(fft_real)count;
            tmp = floor(sqrt((double) grows_tmp));
            if(tmp > box.nmin/2)	tmp = box.nmin/2;

            for(i=-tmp;i<=tmp;i++)
                for(j=-tmp;j<=tmp;j++)
                    for(k=-tmp;k<=tmp;k++)
                        if(dist2(i, j, k) == (size_t) grows_tmp){
                            ind = (size_t)(cysum(peaks[l].x[0], i, box.nd[0])*box.nd[1] + cysum(peaks[l].x[1], j, box.nd[1]))*(size_t)box.nd[2] + (size_t)cysum(peaks[l].x[2], k, box.nd[2]);

                            if(flag[ind] != box.ng)
                                den_tmp += -box.Mtot;
                            else
                                den_tmp += delta[ind];
                            count_tmp ++;						
                        }

            den_tmp = den_tmp/(fft_real)count_tmp;
        }

        /*Decrease the shells up to the correct value of the barrier*/
        while(den < Barrier(Sig_Grid[count]) && count > 0){
            den_tmp = den;
            count_tmp = count;
            den = den*(fft_real)count;
            tmp = floor(sqrt((double) grows));
            if(tmp > box.nmin/2)	tmp = box.nmin/2;

            for(i=-tmp;i<=tmp;i++)
                for(j=-tmp;j<=tmp;j++)
                    for(k=-tmp;k<=tmp;k++)
                        if(dist2(i, j, k) == grows){
                            size_t ind = (size_t)(cysum(peaks[l].x[0], i, box.nd[0])*box.nd[1] + cysum(peaks[l].x[1], j, box.nd[1]))*(size_t)box.nd[2] + (size_t)cysum(peaks[l].x[2], k, box.nd[2]);

                            den -= delta[ind];
                            count --;						
                        }

            if(count > 0)	den = den/(fft_real)count;
            if(count < count_tmp)	grows_tmp = grows;

            grows --;
        }

        if(count == 0)
            continue;

        /*Put the correct flags to the cells*/
        tmp = floor(sqrt((double) grows_tmp));
        for(i=-tmp;i<=tmp;i++)
            for(j=-tmp;j<=tmp;j++)
                for(k=-tmp;k<=tmp;k++)
                    if(dist2(i, j, k) < (size_t) grows_tmp){
                        size_t ind = (size_t)(cysum(peaks[l].x[0], i, box.nd[0])*box.nd[1] + cysum(peaks[l].x[1], j, box.nd[1]))*(size_t)box.nd[2] + (size_t)cysum(peaks[l].x[2], k, box.nd[2]);

                        if(flag[ind] != box.ng)
                            printf("(1): This flag != -1! Flag = %ld and the new one is %ld\n", flag[ind], nh);		

                        flag[ind] = nh;
                    }

        /*Save the halo information*/
        if(count >= barrier.Nmin){
            halos[nh].count = count;
            for(i=0;i<3;i++)
                halos[nh].x[i] = peaks[l].x[i];
            nh ++;
        }
        else{
            for(i=-tmp;i<=tmp;i++)
                for(j=-tmp;j<=tmp;j++)
                    for(k=-tmp;k<=tmp;k++)
                        if(dist2(i, j, k) < (size_t) grows_tmp){
                            size_t ind = (size_t)(cysum(peaks[l].x[0], i, box.nd[0])*box.nd[1] + cysum(peaks[l].x[1], j, box.nd[1]))*(size_t)box.nd[2] + (size_t)cysum(peaks[l].x[2], k, box.nd[2]);
                            flag[ind] = box.ng + 1;
                        }
        }
    }

    return nh;
}

/*Compute the number of grid cells inside each possible sphere*/
void Compute_Spheres(int Ncells, char *spheresfile){
    int i, j, k, m, l, tmp, count, *sphere;
    FILE *spheres;

    /*Copute the total number of grid cells inside each possible sphere*/
    m = 0;
    sphere = (int *)malloc(Ncells*sizeof(int));
    for(l=0;l<10000;l++){
        tmp = floor(sqrt((fft_real) l));
        count = 0;
        
        for(i=-tmp;i<=tmp;i++)
            for(j=-tmp;j<=tmp;j++)
                for(k=-tmp;k<=tmp;k++)
                    if(dist2(i, j, k) == (size_t) l)
                        count ++;
        
        if(count > 0){
            if(m > 0)	sphere[m] = sphere[m-1] + count;
            else		sphere[m] = count;

            m ++;
        }
    }

    /*Save open the output file*/
    spheres = fopen(spheresfile, "wb");
        if (spheres == NULL) {
            printf("Unable to open spheres.dat\n");
            exit(0);
        }

    /*Save the information about the spheres*/
    fwrite(&m, sizeof(int), 1, spheres);
    for(i=0;i<m;i++)
        fwrite(&sphere[i], sizeof(int), 1, spheres);
    fclose(spheres);
    free(sphere);
}

/*Read the number of grid cells inside each possible sphere*/
void Read_Spheres(int **sphere, char *spheresfile){
    int m, i;
    FILE *spheres;   

    /*Open the file*/
    spheres = fopen(spheresfile, "rb");
    if (spheres == NULL) {
        printf("Unable to open spheres.dat\n");
        exit(0);
    }

    fread(&m, sizeof(int), 1, spheres);
    *sphere = (int *)malloc(m*sizeof(int));
    for(i=0;i<m;i++)
        fread(&((*sphere)[i]), sizeof(int), 1, spheres);
    fclose(spheres);
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

/*Compute the mass of each halo*/
void Compute_Mass(size_t nh, int *sphere, HALOS *halos, gsl_interp_accel *acc, gsl_spline *spline_I, gsl_spline *spline_InvI){
    size_t i;
    int Ncells, cont;
    fft_real den_tmp;

    /*Maximum number of cells*/
    Ncells = floor(M_max/box.Mcell);

    /*Alloc quantities for GSL*/
	gsl_rng *rng_ptr;
	rng_ptr = gsl_rng_alloc (gsl_rng_taus);
	gsl_rng_set(rng_ptr, barrier.seed);

    /*Compute the mass of each halo*/
	for(i=0;i<nh;i++){
        cont = Next_Count(sphere, Ncells, halos[i].count);

		den_tmp = gsl_spline_eval(spline_I, halos[i].count*box.Mcell, acc) + (gsl_spline_eval(spline_I, sphere[cont]*box.Mcell, acc) - gsl_spline_eval(spline_I, halos[i].count*box.Mcell, acc))*gsl_rng_uniform(rng_ptr);

		halos[i].Mh = gsl_spline_eval(spline_InvI, den_tmp, acc);
	}
}

/*Find halos from a density grid*/
size_t Find_Halos(fft_real *delta, fft_real *K, fft_real *P, int Nk, size_t *flag, HALOS **halos){
    int Nr, Ncells, *spheres;
    fft_real *R, *Sig_grid;
    double *M, *Sig;
    char spheresfile[1000];
    size_t np, nh; 
    PEAKS *peaks;

    /*Alloc the variables for GSL interpolation*/
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline_I;
    gsl_spline *spline_InvI;

    /*Alloc arrays to used to compute sigma(M) and the mass integrated mass function*/
	Nr = Nk;
	Ncells = floor(M_max/box.Mcell);

	R = (fft_real *)malloc(Nr*sizeof(fft_real));
	check_memory(R, "R")
	M = (double *)malloc(Nr*sizeof(double));
	check_memory(M, "M")
	Sig = (double *)malloc(Nr*sizeof(double));
	check_memory(Sig, "Sig")
	Sig_grid = (fft_real *)malloc(Ncells*sizeof(fft_real));
	check_memory(Sig_grid, "Sig_grid")

	/*Compute and interpolate sigma(M)*/
	Compute_Sig(Nr, R, M, Sig, Sig_grid, K, P, Nk);

	/*Compute Mstar*/
	//cosmo.Mstar = Compute_Mstar(Nr, M, Sig);

	/*Compute the integral over the mass function and interpolate it*/
	spline_I = gsl_spline_alloc(gsl_interp_cspline, Nr);
	spline_InvI = gsl_spline_alloc(gsl_interp_cspline, Nr);
	Compute_nh(1, Nr, R, M, Sig, spline_I, spline_InvI);

	/*Free some arrays*/
	free(R);
	free(M);
	free(Sig);

	/*Count the number of peaks*/
	np = Count_Peaks(delta);

	/*Alloc the array with the peaks*/
	peaks = (PEAKS *)malloc(np*sizeof(PEAKS));

	/*Save the positions and density of each peak*/
	Find_Peaks(delta, np, peaks);

	/*Sort the peaks*/
	quickSort_peaks(peaks, 0, np-1);

	/*Grow the spherical halos around the density peaks*/
	*halos = (HALOS *)malloc(np*sizeof(HALOS));
	if(out.VERBOSE == 1)
		printf("There are %ld peaks\n", np);
	nh = Grow_Halos(np, flag, Sig_grid, delta, peaks, *halos);

	free(peaks);
	if(out.VERBOSE == 1)
		printf("There are %ld halos\n", nh);
	free(Sig_grid);

	/*Compute the file with the number of grid cells in each possible sphere (used in the mass computation)*/
	//Compute_Spheres(Ncells);

	/*Read the number of grid cells inside each possible sphere*/
    strcpy(spheresfile,  SPHERES_DIRC);
    strcat(spheresfile, "Spheres.dat");
	Read_Spheres(&spheres, spheresfile);

    /*Compute the mass of each halo*/
	Compute_Mass(nh, spheres, *halos, acc, spline_I, spline_InvI);

	/*Free some arrays*/
	free(spheres);
	gsl_spline_free(spline_I);
	gsl_spline_free(spline_InvI);
    gsl_interp_accel_free(acc);

    return nh;
}