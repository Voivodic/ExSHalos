#include "lpt.h"

/*Compute the first order displacements*/
void Compute_1LPT(fft_complex *deltak, fft_real **posh, fft_real **velh, fft_real *S, fft_real *V, size_t *flag, fft_real k_smooth){
    int i, j, k;
    size_t ind, tmp;
    fft_real kx, ky, kz, factx, facty, factz, fact, kmod;
	fft_real *phi;
    FFTW(plan) p1;

	/*Compute the first order potential in Fourier space*/
	for(i=0;i<box.nd[0];i++){
		if(2*i<box.nd[0]) kx = i*box.kl[0];
		else kx = (i-box.nd[0])*box.kl[0];

		factx = 1.0/90.0*(2.0*cos(3.0*kx*box.Lc) - 27.0*cos(2.0*kx*box.Lc) + 270.0*cos(kx*box.Lc) - 245.0)/(box.Lc*box.Lc);
	
		for(j=0;j<box.nd[1];j++){
			if(2*j<box.nd[1]) ky = j*box.kl[1];
			else ky = (j-box.nd[1])*box.kl[1];

			facty = 1.0/90.0*(2.0*cos(3.0*ky*box.Lc) - 27.0*cos(2.0*ky*box.Lc) + 270.0*cos(ky*box.Lc) - 245.0)/(box.Lc*box.Lc);
	
			for(k=0;k<box.nz2;k++){
				kz = k*box.kl[2];
				if(k == box.nd[2]/2) kz = -(fft_real)box.nd[2]/2.0*box.kl[2];

				factz = 1.0/90.0*(2.0*cos(3.0*kz*box.Lc) - 27.0*cos(2.0*kz*box.Lc) + 270.0*cos(kz*box.Lc) - 245.0)/(box.Lc*box.Lc);
				kmod = sqrt(pow(kx, 2.0) + pow(ky, 2.0) + pow(kz, 2.0));

				ind = (size_t)(i*box.nd[1] + j)*(size_t)box.nz2 + (size_t)k;
				if((kx != 0.0 || ky != 0.0 || kz != 0.0) && (kmod <= k_smooth)){
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

	/*Alloc the grid for the potential*/
	phi = (fft_real *) malloc(((size_t) box.nd[0])*((size_t) box.nd[1])*((size_t) box.nd[2])*sizeof(fft_real));

	/*Compute the potential at first order in real space*/
    p1 = FFTW(plan_dft_c2r_3d)(box.nd[0], box.nd[1], box.nd[2], deltak, phi, FFTW_ESTIMATE); 
	FFTW(execute)(p1);
    FFTW(destroy_plan)(p1);

	/*Compute the first order displacements, save it in the S arrays, and update the position and velocity of each halo*/
	tmp = -1;
	for(i=0;i<box.nd[0];i++)
		for(j=0;j<box.nd[1];j++)
			for(k=0;k<box.nd[2];k++){
				ind = (size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k;
				if(out.OUT_HALOS != FALSE) tmp = flag[ind];

				if(tmp < 0 && out.OUT_LPT == FALSE && out.DO_2LPT == FALSE)	continue;
	
				kx = -(1.0*phi[(size_t)(cysum(i, 3, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k] - 9.0*phi[(size_t)(cysum(i, 2, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k] + 45.0*phi[(size_t)(cysum(i, 1, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k] - 45.0*phi[(size_t)(cysum(i, -1, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k] + 9.0*phi[(size_t)(cysum(i, -2, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k] - 1.0*phi[(size_t)(cysum(i, -3, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k])*box.Normx/(60.0*box.Lc);

				ky = -(1.0*phi[(size_t)(i*box.nd[1] + cysum(j, 3, box.nd[1]))*(size_t)box.nd[2] + (size_t)k] - 9.0*phi[(size_t)(i*box.nd[1] + cysum(j, 2, box.nd[1]))*(size_t)box.nd[2] + (size_t)k] + 45.0*phi[(size_t)(i*box.nd[1] + cysum(j, 1, box.nd[1]))*(size_t)box.nd[2] + (size_t)k] - 45.0*phi[(size_t)(i*box.nd[0] + cysum(j, -1, box.nd[1]))*(size_t)box.nd[2] + (size_t)k] + 9.0*phi[(size_t)(i*box.nd[1] + cysum(j, -2, box.nd[1]))*(size_t)box.nd[2] + (size_t)k] - 1.0*phi[(size_t)(i*box.nd[1] + cysum(j, -3, box.nd[1]))*(size_t)box.nd[2] + (size_t)k])*box.Normx/(60.0*box.Lc);

				kz = -(1.0*phi[(size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, 3, box.nd[2])] - 9.0*phi[(size_t)(i*box.nd[1] + j)*box.nd[2] + (size_t)cysum(k, 2, box.nd[2])] + 45.0*phi[(size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, 1, box.nd[2])] - 45.0*phi[(size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + cysum(k, -1, box.nd[2])] + 9.0*phi[(size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, -2, box.nd[2])] - 1.0*phi[(size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, -3, box.nd[2])])*box.Normx/(60.0*box.Lc);

				if(out.OUT_HALOS != FALSE && tmp >= 0){
					posh[tmp][0] += kx;
					posh[tmp][1] += ky;
					posh[tmp][2] += kz;
	
					velh[tmp][0] += pow(cosmo.Omz, 0.5454)*cosmo.Hz/(1.0 + cosmo.redshift)*kx;
					velh[tmp][1] += pow(cosmo.Omz, 0.5454)*cosmo.Hz/(1.0 + cosmo.redshift)*ky;
					velh[tmp][2] += pow(cosmo.Omz, 0.5454)*cosmo.Hz/(1.0 + cosmo.redshift)*kz;
				}

				if(out.OUT_LPT == TRUE || out.DO_2LPT == TRUE){
					S[3*ind] = kx;
					S[3*ind+1] = ky;
					S[3*ind+2] = kz;

					if(out.OUT_LPT == TRUE && out.OUT_VEL == TRUE){
						V[3*ind] = pow(cosmo.Omz, 0.5454)*cosmo.Hz/(1.0 + cosmo.redshift)*kx;
						V[3*ind+1] = pow(cosmo.Omz, 0.5454)*cosmo.Hz/(1.0 + cosmo.redshift)*ky;
						V[3*ind+2] = pow(cosmo.Omz, 0.5454)*cosmo.Hz/(1.0 + cosmo.redshift)*kz;
					}
				}
			}

	free(phi);
}

/*Compute the second order displacements*/
void Compute_2LPT(fft_real **posh, fft_real **velh, fft_real *S, fft_real *V, size_t *flag, fft_real k_smooth){
    int i, j, k;
    size_t ind, tmp;
    fft_real kx, ky, kz, kmod, factx, facty, factz, fact, phixx, phixy, phixz, phiyy, phiyz, phizz;
	fft_real *phi;
	fft_complex *phik;
    FFTW(plan) p1, p2;

	/*Alloc the arrays for the potential and its Fourier transform*/
	phi = (fft_real *) malloc(((size_t) box.nd[0])*((size_t) box.nd[1])*((size_t) box.nd[2])*sizeof(fft_real));
	phik = (fft_complex *) FFTW(malloc)(((size_t) box.nd[0])*((size_t) box.nd[1])*((size_t) box.nz2)*sizeof(fft_complex));

	/*Prepare the FFTW plans*/
    p1 = FFTW(plan_dft_c2r_3d)(box.nd[0], box.nd[1], box.nd[2], phik, phi, FFTW_ESTIMATE); 
	p2 = FFTW(plan_dft_r2c_3d)(box.nd[0], box.nd[1], box.nd[2], phi, phik, FFTW_ESTIMATE); 

	/*Compute the second order "density"*/
	for(i=0;i<box.nd[0];i++)
	    for(j=0;j<box.nd[1];j++)
			for(k=0;k<box.nd[2];k++){
                ind = (size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k;

				phixx = (1.0*S[3*((size_t)(cysum(i, 3, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k)] - 9.0*S[3*((size_t)(cysum(i, 2, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k)] + 45.0*S[3*((size_t)(cysum(i, 1, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k)] - 45.0*S[3*((size_t)(cysum(i, -1, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k)] + 9.0*S[3*((size_t)(cysum(i, -2, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k)] - 1.0*S[3*((size_t)(cysum(i, -3, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k)])/(60.0*box.Lc);

				phixy = (1.0*S[3*((size_t)(i*box.nd[1] + cysum(j, 3, box.nd[1]))*(size_t)box.nd[2] + (size_t)k)] - 9.0*S[3*((size_t)(i*box.nd[1] + cysum(j, 2, box.nd[1]))*(size_t)box.nd[2] + (size_t)k)] + 45.0*S[3*((size_t)(i*box.nd[1] + cysum(j, 1, box.nd[1]))*(size_t)box.nd[2] + (size_t)k)] - 45.0*S[3*((size_t)(i*box.nd[0] + cysum(j, -1, box.nd[1]))*(size_t)box.nd[2] + (size_t)k)] + 9.0*S[3*((size_t)(i*box.nd[1] + cysum(j, -2, box.nd[1]))*(size_t)box.nd[2] + (size_t)k)] - 1.0*S[3*((size_t)(i*box.nd[1] + cysum(j, -3, box.nd[1]))*(size_t)box.nd[2] + (size_t)k)])/(60.0*box.Lc);

				phixz = (1.0*S[3*((size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, 3, box.nd[2]))] - 9.0*S[3*((size_t)(i*box.nd[1] + j)*box.nd[2] + (size_t)cysum(k, 2, box.nd[2]))] + 45.0*S[3*((size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, 1, box.nd[2]))] - 45.0*S[3*((size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + cysum(k, -1, box.nd[2]))] + 9.0*S[3*((size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, -2, box.nd[2]))] - 1.0*S[3*((size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, -3, box.nd[2]))])/(60.0*box.Lc);

				phiyy = (1.0*S[3*((size_t)(i*box.nd[1] + cysum(j, 3, box.nd[1]))*(size_t)box.nd[2] + (size_t)k) + 1] - 9.0*S[3*((size_t)(i*box.nd[1] + cysum(j, 2, box.nd[1]))*(size_t)box.nd[2] + (size_t)k) + 1] + 45.0*S[3*((size_t)(i*box.nd[1] + cysum(j, 1, box.nd[1]))*(size_t)box.nd[2] + (size_t)k) + 1] - 45.0*S[3*((size_t)(i*box.nd[0] + cysum(j, -1, box.nd[1]))*(size_t)box.nd[2] + (size_t)k) + 1] + 9.0*S[3*((size_t)(i*box.nd[1] + cysum(j, -2, box.nd[1]))*(size_t)box.nd[2] + (size_t)k) + 1] - 1.0*S[3*((size_t)(i*box.nd[1] + cysum(j, -3, box.nd[1]))*(size_t)box.nd[2] + (size_t)k) + 1])/(60.0*box.Lc);

				phiyz = (1.0*S[3*((size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, 3, box.nd[2])) + 1] - 9.0*S[3*((size_t)(i*box.nd[1] + j)*box.nd[2] + (size_t)cysum(k, 2, box.nd[2])) + 1] + 45.0*S[3*((size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, 1, box.nd[2])) + 1] - 45.0*S[3*((size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + cysum(k, -1, box.nd[2])) + 1] + 9.0*S[3*((size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, -2, box.nd[2])) + 1] - 1.0*S[3*((size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, -3, box.nd[2])) + 1])/(60.0*box.Lc);

				phizz = (1.0*S[3*((size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, 3, box.nd[2])) + 2] - 9.0*S[3*((size_t)(i*box.nd[1] + j)*box.nd[2] + (size_t)cysum(k, 2, box.nd[2])) + 2] + 45.0*S[3*((size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, 1, box.nd[2])) + 2] - 45.0*S[3*((size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + cysum(k, -1, box.nd[2])) + 2] + 9.0*S[3*((size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, -2, box.nd[2])) + 2] - 1.0*S[3*((size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, -3, box.nd[2])) + 2])/(60.0*box.Lc);

				phi[ind] = (phixx*phiyy + phixx*phizz + phiyy*phizz - pow(phixy, 2.0) - pow(phixz, 2.0) - pow(phiyz, 2.0));
			}

	/*Go to fourier space to solve the poisson equation*/
	FFTW(execute)(p2);
    FFTW(destroy_plan)(p2);

	/*Compute the second order potential in Fourier space*/
	for(i=0;i<box.nd[0];i++){
		if(2*i<box.nd[0]) kx = i*box.kl[0];
		else kx = (i-box.nd[0])*box.kl[0];
	
		factx = 1.0/90.0*(2.0*cos(3.0*kx*box.Lc) - 27.0*cos(2.0*kx*box.Lc) + 270.0*cos(kx*box.Lc) - 245.0)/(box.Lc*box.Lc);

		for(j=0;j<box.nd[1];j++){
			if(2*j<box.nd[1]) ky = j*box.kl[1];
			else ky = (j-box.nd[1])*box.kl[1];

			facty = 1.0/90.0*(2.0*cos(3.0*ky*box.Lc) - 27.0*cos(2.0*ky*box.Lc) + 270.0*cos(ky*box.Lc) - 245.0)/(box.Lc*box.Lc);
	
			for(k=0;k<box.nz2;k++){
				kz = k*box.kl[2];
				if(k == box.nd[2]/2) kz = -(fft_real)box.nd[2]/2.0*box.kl[2];

				factz = 1.0/90.0*(2.0*cos(3.0*kz*box.Lc) - 27.0*cos(2.0*kz*box.Lc) + 270.0*cos(kz*box.Lc) - 245.0)/(box.Lc*box.Lc);
				kmod = sqrt(pow(kx, 2.0) + pow(ky, 2.0) + pow(kz, 2.0));

				ind = (size_t)(i*box.nd[1] + j)*(size_t)box.nz2 + (size_t)k;
				if((kx != 0.0 || ky != 0.0 || kz != 0.0) && (kmod <= k_smooth)){
					fact = factx + facty + factz;

					phik[ind][0] = phik[ind][0]/fact*box.Normk;
					phik[ind][1] = phik[ind][1]/fact*box.Normk;
				}
				else{
					phik[ind][0] = 0.0;
					phik[ind][1] = 0.0;
				}
			}
		}
	}

	/*Compute the second order potential in real space*/
	FFTW(execute)(p1);
    FFTW(destroy_plan)(p1);
	FFTW(free)(phik);

	/*Compute the second order displacements and velocities*/
	tmp = -1;
	for(i=0;i<box.nd[0];i++)
		for(j=0;j<box.nd[1];j++)
			for(k=0;k<box.nd[2];k++){
				ind = (size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k;
				if(out.OUT_HALOS != FALSE) tmp = flag[ind];

				if(tmp < 0 && out.OUT_LPT == FALSE)	continue;

				kx = (1.0*phi[(size_t)(cysum(i, 3, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k] - 9.0*phi[(size_t)(cysum(i, 2, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k] + 45.0*phi[(size_t)(cysum(i, 1, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k] - 45.0*phi[(size_t)(cysum(i, -1, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k] + 9.0*phi[(size_t)(cysum(i, -2, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k] - 1.0*phi[(size_t)(cysum(i, -3, box.nd[0])*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k])*box.Normx/(60.0*box.Lc);

				ky = (1.0*phi[(size_t)(i*box.nd[1] + cysum(j, 3, box.nd[1]))*(size_t)box.nd[2] + (size_t)k] - 9.0*phi[(size_t)(i*box.nd[1] + cysum(j, 2, box.nd[1]))*(size_t)box.nd[2] + (size_t)k] + 45.0*phi[(size_t)(i*box.nd[1] + cysum(j, 1, box.nd[1]))*(size_t)box.nd[2] + (size_t)k] - 45.0*phi[(size_t)(i*box.nd[0] + cysum(j, -1, box.nd[1]))*(size_t)box.nd[2] + (size_t)k] + 9.0*phi[(size_t)(i*box.nd[1] + cysum(j, -2, box.nd[1]))*(size_t)box.nd[2] + (size_t)k] - 1.0*phi[(size_t)(i*box.nd[1] + cysum(j, -3, box.nd[1]))*(size_t)box.nd[2] + (size_t)k])*box.Normx/(60.0*box.Lc);

				kz = (1.0*phi[(size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, 3, box.nd[2])] - 9.0*phi[(size_t)(i*box.nd[1] + j)*box.nd[2] + (size_t)cysum(k, 2, box.nd[2])] + 45.0*phi[(size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, 1, box.nd[2])] - 45.0*phi[(size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + cysum(k, -1, box.nd[2])] + 9.0*phi[(size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, -2, box.nd[2])] - 1.0*phi[(size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)cysum(k, -3, box.nd[2])])*box.Normx/(60.0*box.Lc);

				kx = -3.0/7.0*pow(cosmo.Omz, -1.0/143)*kx;
				ky = -3.0/7.0*pow(cosmo.Omz, -1.0/143)*ky;
				kz = -3.0/7.0*pow(cosmo.Omz, -1.0/143)*kz;

				if(out.OUT_HALOS != FALSE && tmp >= 0){	
					posh[tmp][0] += kx;
					posh[tmp][1] += ky;
					posh[tmp][2] += kz;
	
					velh[tmp][0] += 2.0*pow(cosmo.Omz, 6.0/11.0)*cosmo.Hz/(1.0 + cosmo.redshift)*kx;
					velh[tmp][1] += 2.0*pow(cosmo.Omz, 6.0/11.0)*cosmo.Hz/(1.0 + cosmo.redshift)*ky;
					velh[tmp][2] += 2.0*pow(cosmo.Omz, 6.0/11.0)*cosmo.Hz/(1.0 + cosmo.redshift)*kz;
				}

				if(out.OUT_LPT == TRUE){
					S[3*ind] += kx;
					S[3*ind+1] += ky;
					S[3*ind+2] += kz;

					if(out.OUT_VEL == TRUE){
						V[3*ind] += 2.0*pow(cosmo.Omz, 6.0/11.0)*cosmo.Hz/(1.0 + cosmo.redshift)*kx;
						V[3*ind+1] += 2.0*pow(cosmo.Omz, 6.0/11.0)*cosmo.Hz/(1.0 + cosmo.redshift)*ky;
						V[3*ind+2] += 2.0*pow(cosmo.Omz, 6.0/11.0)*cosmo.Hz/(1.0 + cosmo.redshift)*kz;
					}
				}						
			}

	free(phi);
}

/*Compute the final position of each particle*/
void Compute_Pos(fft_real *S){
	size_t ind;
    int i, j, k;

	for(i=0;i<box.nd[0];i++)
		for(j=0;j<box.nd[1];j++)
			for(k=0;k<box.nd[2];k++){
				ind = (size_t)(i*box.nd[1] + j)*(size_t)box.nd[2] + (size_t)k;

				S[3*ind] = cysumf(i*box.Lc + box.Lc/2.0, S[3*ind], box.L[0]);
				S[3*ind+1] = cysumf(j*box.Lc + box.Lc/2.0, S[3*ind+1], box.L[1]);
				S[3*ind+2] = cysumf(k*box.Lc + box.Lc/2.0, S[3*ind+2], box.L[2]);
			}
}

/*Compute the final position and velocity of each halo*/
void Compute_Posh(HALOS *halos, fft_real **posh, fft_real **velh, size_t nh){
    int i;

    /*Compute the mean position and velocity of each halo*/
    for(i=0;i<nh;i++){
		posh[i][0] = cysumf(halos[i].x[0]*box.Lc + box.Lc/2.0, posh[i][0]/halos[i].count, box.L[0]);
		posh[i][1] = cysumf(halos[i].x[1]*box.Lc + box.Lc/2.0, posh[i][1]/halos[i].count, box.L[1]);
		posh[i][2] = cysumf(halos[i].x[2]*box.Lc + box.Lc/2.0, posh[i][2]/halos[i].count, box.L[2]);

		velh[i][0] = velh[i][0]/halos[i].count;
		velh[i][1] = velh[i][1]/halos[i].count;
		velh[i][2] = velh[i][2]/halos[i].count;
	}
}
