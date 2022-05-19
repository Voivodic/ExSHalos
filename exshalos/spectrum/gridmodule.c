#include "gridmodule.h"

/*Define the Hubble function in units of h*/
fft_real H(fft_real Om0, fft_real z){
	return 100.0*sqrt(Om0*pow(1.0 + z, 3) + (1.0 - Om0));
}

/*Give the density to each grid using the NGP density assignment*/
long double NGP(fft_real *grid, fft_real *pos, int nd, fft_real L, fft_real mass){
	fft_real Ld = L/nd;
    int xt[3];
    
    ind(pos, xt, Ld, nd);

    grid[xt[0]*nd*nd + xt[1]*nd + xt[2]] += mass;

    return (long double) mass;
}

/*Give the density to each grid using the CIC density assignment*/
long double CIC(fft_real *grid, fft_real *pos, int nd, fft_real L, fft_real mass){

	fft_real Ld =  L/nd;
	fft_real dx[3], t[3];
	int xt[3], sign[3], i;

	ind(pos, xt, Ld, nd);

	for(i=0;i<3;i++){
		dx[i] = pos[i]/Ld - (xt[i] + 1.0/2.0);	

		if(dx[i]>=0.0){ 
			sign[i] = 1;
			t[i] = 1.0 - dx[i];
		}
		else{
			sign[i] = -1;
			dx[i] = -dx[i];
			t[i] = 1.0 - dx[i];
		}
	}

	grid[xt[0]*nd*nd + xt[1]*nd + xt[2]] += mass*t[0]*t[1]*t[2];
	grid[mod(xt[0],sign[0],nd)*nd*nd + xt[1]*nd + xt[2]] += mass*dx[0]*t[1]*t[2];
	grid[xt[0]*nd*nd + mod(xt[1],sign[1],nd)*nd + xt[2]] += mass*t[0]*dx[1]*t[2];
	grid[xt[0]*nd*nd + xt[1]*nd + mod(xt[2],sign[2],nd)] += mass*t[0]*t[1]*dx[2];
	grid[mod(xt[0],sign[0],nd)*nd*nd + mod(xt[1],sign[1],nd)*nd + xt[2]] += mass*dx[0]*dx[1]*t[2];
	grid[mod(xt[0],sign[0],nd)*nd*nd + xt[1]*nd + mod(xt[2],sign[2],nd)] += mass*dx[0]*t[1]*dx[2];
	grid[xt[0]*nd*nd + mod(xt[1],sign[1],nd)*nd + mod(xt[2],sign[2],nd)] += mass*t[0]*dx[1]*dx[2];
	grid[mod(xt[0],sign[0],nd)*nd*nd + mod(xt[1],sign[1],nd)*nd + mod(xt[2],sign[2],nd)] += mass*dx[0]*dx[1]*dx[2];

    return (long double) mass;
}

/*Give the density to each grid using a sphere*/
long double Sphere(fft_real *grid, fft_real *pos, int nd, fft_real L, fft_real R, fft_real mass){
	fft_real Ld =  L/nd;
	fft_real dx[3];
	int xt[3], i, j, k, l, times;
	long double M;

	ind(pos, xt, Ld, nd);
	times = floor(0.5 + R/Ld);
	
	M = 0.0;
	for(i=-times;i<=times;i++)
		for(j=-times;j<=times;j++)
			for(k=-times;k<=times;k++){
				dx[0] = pos[0] - (mod(xt[0], i, nd) + 1.0/2.0)*Ld;
				dx[1] = pos[1] - (mod(xt[1], j, nd) + 1.0/2.0)*Ld;
				dx[2] = pos[2] - (mod(xt[2], k, nd) + 1.0/2.0)*Ld;
	
				for(l=0;l<3;l++){	
					if(dx[l] > L/2.0)	dx[l] = L - dx[l];
					if(dx[l] < -L/2.0)	dx[l] = L + dx[l];
				}

				if(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2] <= R*R){
					grid[mod(xt[0],i,nd)*nd*nd + mod(xt[1],j,nd)*nd + mod(xt[2],k,nd)] += mass;
					M += (long double) mass;
				}
			}

	return M;
}

/*Give the density to each grid using a sphere*/
long double Exp(fft_real *grid, fft_real *pos, int nd, fft_real L, fft_real R, fft_real R_times, fft_real mass){
	fft_real Ld =  L/nd;
	fft_real dx[3], r2, w;
	int xt[3], i, j, k, l, times;
	long double M;
	
	ind(pos, xt, Ld, nd);
	times = floor(0.5 + R*R_times/Ld);
	
	M = 0.0;
	for(i=-times;i<=times;i++)
		for(j=-times;j<=times;j++)
			for(k=-times;k<=times;k++){
				dx[0] = pos[0] - (mod(xt[0], i, nd) + 1.0/2.0)*Ld;
				dx[1] = pos[1] - (mod(xt[1], j, nd) + 1.0/2.0)*Ld;
				dx[2] = pos[2] - (mod(xt[2], k, nd) + 1.0/2.0)*Ld;
	
				for(l=0;l<3;l++){	
					if(dx[l] > L/2.0)	dx[l] = L - dx[l];
					if(dx[l] < -L/2.0)	dx[l] = L + dx[l];
				}

				r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
				if(r2 <= R*R*R_times*R_times){
					w = exp(-0.5*r2/R/R);
					grid[mod(xt[0],i,nd)*nd*nd + mod(xt[1],j,nd)*nd + mod(xt[2],k,nd)] += mass*w;
					M += (long double) mass*w;
				}
			}

	return M;
}

/*Compute the density grid given a particle*/
long double Density_Grid(fft_real *grid, int nd, fft_real L, fft_real *pos, fft_real mass, int window, fft_real R, fft_real R_times){
    long double M;

    if(window == 1)
        M = NGP(grid, pos, nd, L, mass);

    else if(window == 2)
    	M = CIC(grid, pos, nd, L, mass);

    else if(window == 3)
        M = Sphere(grid, pos, nd, L, R, mass);

    else if(window == 4)
        M = Exp(grid, pos, nd, L, R, R_times, mass);

    else{
        printf("You must choose a value of the window (w) 0<=w<=3!\n\tNGP (0)\n\tCIC (1)\n\tSphere (2)\n\tExp (3)\n");
        exit(0);
    }

	return M;
}

/*Compute the density grids for each type of tracer*/
void Tracer_Grid(fft_real *grid, int nd, fft_real L, int direction, fft_real *pos, fft_real *vel, size_t np, fft_real *mass,  int *type, int ntype, int window, fft_real R, fft_real R_times, int interlacing, fft_real Om0, fft_real z){
	size_t j, ng, ind, ind2;
	int i;
    fft_real post[3], Lb;
	long double *M1, *M2;

	Lb = L/nd;		//Size of each cell
	ng = ((size_t) nd)*((size_t) nd)*((size_t) nd);	//Total number of cells

	/*Array with the total mass for each type of tracer*/
	M1 = (long double *)malloc(ntype*sizeof(long double));
	if(interlacing == TRUE)
		M2 = (long double *)malloc(ntype*sizeof(long double));
	for(i=0;i<ntype;i++){
		M1[i] = 0.0;
		if(interlacing == TRUE)
			M2[i] = 0.0;
	}

	/*Put the particles in redshift space*/
	if(direction != -1)
		for(i=0;i<np;i++)
			pos[3*i+direction] = cysumf(pos[3*i+direction], vel[3*i+direction]*pow(1.0 + z, 3.0)/H(Om0, z), L);

	/*Case with multiple types without weight (mass) between the particles and without interlacing*/
	if(mass == NULL && interlacing == FALSE){
		#pragma omp parallel private(j, i) shared(np, pos, grid, M1, nd, L, window, R, R_times)
		{
			long double *M1_private = (long double *)malloc(ntype*sizeof(long double));
			fft_real post[3];

			for(i=0;i<ntype;i++)
				M1_private[i] = 0.0;

			if(ntype == 1){
				#pragma omp for
				for(j=0;j<np;j++){
					for(i=0;i<3;i++)
						post[i] = pos[3*j+i];
					M1_private[0] += Density_Grid(grid, nd, L, post, 1.0, window, R, R_times);
				}
			}
			else{
				#pragma omp for
				for(j=0;j<np;j++){
					for(i=0;i<3;i++)
						post[i] = pos[3*j+i];
					M1_private[type[j]] += Density_Grid(&grid[((size_t) type[j])*ng], nd, L, post, 1.0, window, R, R_times);
				}			
			}

			#pragma omp critical
			{
				for(i=0;i<ntype;i++)
					M1[i] += M1_private[i];
			}
			free(M1_private);
		}

		for(i=0;i<ntype;i++){
			ind = ((size_t) i)*ng;
			#pragma omp parallel for private(j) shared(ind, grid, M1, ng)
			for(j=0;j<ng;j++)
				grid[ind + j] = grid[ind + j]/(M1[i]/ng) - 1.0;
		}
	}

	/*Case with multiple types without weight (mass) between the particles and with interlacing*/
	else if(mass == NULL && interlacing == TRUE){
		#pragma omp parallel private(j, i) shared(np, pos, grid, M1, M2, nd, L, window, R, R_times)
		{
			long double *M1_private = (long double *)malloc(ntype*sizeof(long double));
			long double *M2_private = (long double *)malloc(ntype*sizeof(long double));
			fft_real post[3];

			for(i=0;i<ntype;i++){
				M1_private[i] = 0.0;
				M2_private[i] = 0.0;
			}

			if(ntype == 1){
				#pragma omp for
				for(j=0;j<np;j++){
					for(i=0;i<3;i++)
						post[i] = pos[3*j+i];
					M1_private[0] += Density_Grid(grid, nd, L, post, 1.0, window, R, R_times);
					for(i=0;i<3;i++)
						post[i] = cysumf(post[i], -Lb/2.0, L);
					M2_private[0] += Density_Grid(&grid[ng], nd, L, post, 1.0, window, R, R_times);
				}
			}
			else{
				#pragma omp for
				for(j=0;j<np;j++){
					for(i=0;i<3;i++)
						post[i] = pos[3*j+i];
					M1_private[type[j]] += Density_Grid(&grid[2*((size_t) type[j])*ng], nd, L, post, 1.0, window, R, R_times);
					for(i=0;i<3;i++)
						post[i] = cysumf(post[i], -Lb/2.0, L);
					M2_private[type[j]] += Density_Grid(&grid[(2*((size_t) type[j]) + 1)*ng], nd, L, post, 1.0, window, R, R_times);
				}			
			}

			#pragma omp critical
			{
				for(i=0;i<ntype;i++){
					M1[i] += M1_private[i];
					M2[i] += M2_private[i];
				}
			}
			free(M1_private);
			free(M2_private);
		}

		for(i=0;i<ntype;i++){
			ind = 2*((size_t) i)*ng;
			ind2 = (2*((size_t) i) + 1)*ng;
			#pragma omp parallel for private(j) shared(ind, ind2, grid, M1, M2, ng)
			for(j=0;j<ng;j++){
				grid[ind + j] = grid[ind + j]/(M1[i]/ng) - 1.0;
				grid[ind2 + j] = grid[ind2 + j]/(M2[i]/ng) - 1.0;
			}
		}
	}

	/*Case with multiple types with weight (mass) between the particles and without interlacing*/
	else if (mass != NULL && interlacing == FALSE){
		#pragma omp parallel private(j, i) shared(np, pos, grid, M1, nd, L, window, R, R_times, mass)
		{
			long double *M1_private = (long double *)malloc(ntype*sizeof(long double));
			fft_real post[3];

			for(i=0;i<ntype;i++)
				M1_private[i] = 0.0;

			if(ntype == 1){
				#pragma omp for
				for(j=0;j<np;j++){
					for(i=0;i<3;i++)
						post[i] = pos[3*j+i];
					M1_private[0] += Density_Grid(grid, nd, L, post, mass[j], window, R, R_times);
				}
			}
			else{
				#pragma omp for
				for(j=0;j<np;j++){
					for(i=0;i<3;i++)
						post[i] = pos[3*j+i];
					M1_private[type[j]] += Density_Grid(&grid[((size_t) type[j])*ng], nd, L, post, mass[j], window, R, R_times);
				}			
			}

			#pragma omp critical
			{
				for(i=0;i<ntype;i++)
					M1[i] += M1_private[i];
			}
			free(M1_private);
		}

		for(i=0;i<ntype;i++){
			ind = ((size_t) i)*ng;
			#pragma omp parallel for private(j) shared(ind, grid, M1, ng)
			for(j=0;j<ng;j++)
				grid[ind + j] = grid[ind + j]/(M1[i]/ng) - 1.0;
		}
	}

	/*Case with multiple types with weight (mass) between the particles and with interlacing*/
	else if (mass != NULL &&  interlacing == TRUE){
		#pragma omp parallel private(j, i) shared(np, pos, grid, M1, M2, nd, L, window, R, R_times, mass)
		{
			long double *M1_private = (long double *)malloc(ntype*sizeof(long double));
			long double *M2_private = (long double *)malloc(ntype*sizeof(long double));
			fft_real post[3];

			for(i=0;i<ntype;i++){
				M1_private[i] = 0.0;
				M2_private[i] = 0.0;
			}

			if(ntype == 1){
				#pragma omp for
				for(j=0;j<np;j++){
					for(i=0;i<3;i++)
						post[i] = pos[3*j+i];
					M1_private[0] += Density_Grid(grid, nd, L, post, mass[j], window, R, R_times);
					for(i=0;i<3;i++)
						post[i] = cysumf(post[i], -Lb/2.0, L);
					M2_private[0] += Density_Grid(&grid[ng], nd, L, post, mass[j], window, R, R_times);
				}
			}
			else{
				#pragma omp for
				for(j=0;j<np;j++){
					for(i=0;i<3;i++)
						post[i] = pos[3*j+i];
					M1_private[type[j]] += Density_Grid(&grid[2*((size_t) type[j])*ng], nd, L, post, mass[j], window, R, R_times);
					for(i=0;i<3;i++)
						post[i] = cysumf(post[i], -Lb/2.0, L);
					ind = ((size_t) ntype)*ng;
					M2_private[type[j]] += Density_Grid(&grid[(2*((size_t) type[j]) + 1)*ng], nd, L, post, mass[j], window, R, R_times);
				}			
			}

			#pragma omp critical
			{
				for(i=0;i<ntype;i++){
					M1[i] += M1_private[i];
					M2[i] += M2_private[i];
				}
			}
			free(M1_private);
			free(M2_private);
		}

		for(i=0;i<ntype;i++){
			ind = 2*((size_t) i)*ng;
			ind2 = (2*((size_t) i) + 1)*ng;
			#pragma omp parallel for private(j) shared(ind, ind2, grid, M1, M2, ng)
			for(j=0;j<ng;j++){
				grid[ind + j] = grid[ind + j]/(M1[i]/ng) - 1.0;
				grid[ind2 + j] = grid[ind2 + j]/(M2[i]/ng) - 1.0;
			}
		}
	}
	free(M1);
	if(interlacing == TRUE)
		free(M2);
}