#include "find_halos.h"

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

/*Partition function for the quicksort*/
size_t partition_peaks(PEAKS *a, size_t l, size_t r){
   	size_t i, j, k;
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
void quickSort_peaks(PEAKS *a, size_t l, size_t r){
	size_t j;

   	if( l < r ){
   	// divide and conquer
        j = partition_peaks( a, l, r);
       	quickSort_peaks( a, l, j-1);
      	quickSort_peaks( a, j+1, r);
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
    for(l=0;l<np;l++){

        /*If this peak is already in a halo jump to the next one*/
        if(flag[(size_t)(peaks[l].x[0]*box.nd[1] + peaks[l].x[1])*(size_t)box.nd[2] + (size_t)peaks[l].x[2]] != (size_t) -1)
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

                            if(flag[ind] != (size_t) -1)
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

                        if(flag[ind] != (size_t) -1)
                            printf("(1): This flag != -1! Flag = %ld and the new one is %ld\n", flag[ind], nh);		

                        flag[ind] = nh;
                    }

        /*Save the halo information*/
        if(count >= barrier.Nmin){
            halos[nh].cont = count;
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
                            flag[ind] = -2;
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
void Compute_Mass(size_t nh, int *sphere, HALOS *halos, gsl_interp_accel *acc, gsl_spline *spline_I, gsl_spline *spline_InvI, fft_real *Massh){
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
        cont = Next_Count(sphere, Ncells, halos[i].cont);

		den_tmp = gsl_spline_eval(spline_I, halos[i].cont*box.Mcell, acc) + (gsl_spline_eval(spline_I, sphere[cont]*box.Mcell, acc) - gsl_spline_eval(spline_I, halos[i].cont*box.Mcell, acc))*gsl_rng_uniform(rng_ptr);
		Massh[i] = gsl_spline_eval(spline_InvI, den_tmp, acc);
	}
}