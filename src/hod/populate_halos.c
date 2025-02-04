#include "populate_halos.h"

/*Mean value of central galaxies*/
fft_real Ncentral_tot(fft_real M){
	return 0.5*(1.0 + erf((log10(M) - hodp.logMmin)/hodp.siglogM));
}

/*Mean value of satellite galaxies*/
fft_real Nsatellite_tot(fft_real M){
	fft_real resp;

    if(M >= pow(10.0, hodp.logM0))
	    resp = pow((M - pow(10.0, hodp.logM0))/pow(10.0, hodp.logM1), hodp.alpha)*Ncentral_tot(M);
    else    
        resp = 0.0;

	return resp;
}

/*Halo concentration*/
fft_real f_c(fft_real Mv){
    fft_real resp;

    resp = 9.0/(1.0 + cosmo.redshift)*pow(Mv/cosmo.Mstar, -0.13);

	return resp;
}

/*Unnormalized density profile used to populate the halos*/
fft_real Profile(fft_real x, fft_real c){
	fft_real resp;

	resp = 1.0/(c*x*pow(1.0 + c*x, 2.0))/(1.0 + pow(x, 2.0/hodp.sigma));

	return resp;
}

/*Compute construct the interpolations used to generate the radial position of the galaxies*/
void Interpolate_r_Eps(fft_real cmin, fft_real cmax, gsl_spline **spline_r, gsl_interp_accel *acc, fft_real M_frac){
	size_t i, j;
	fft_real c, A, rtmp;
	double x[NRs], Int[NRs], Eps[Neps], r[Neps], Mh, rmax;

	/*Alloc the GSL stuff for interpolations*/
    gsl_spline *spline_I;
	spline_I = gsl_spline_alloc(gsl_interp_cspline, NRs);
	for(i=0;i<NRs;i++)
		x[i] = pow(10.0, log10(R_MIN) + (log10(R_MAX) - log10(R_MIN))*i/(NRs - 1));

	/*Define the arrays of Eps used to generate the r*/
	for(i=0;i<Neps;i++)
		Eps[i] = ((double) i)/(Neps - 1);

	/*Run over all concentrarion bins to compute the interpolation*/
	for(i=0;i<NCs;i++){
		c = pow(10.0, log10(cmin) + (log10(cmax) - log10(cmin))*i/(NCs - 1));

		/*Compute the integral of the density profile and interpolate it*/
		Int[0] = 0.0;
        Mh = 0.0;
		for(j=1;j<NRs;j++){
			Int[j] = Int[j-1] + (Profile(x[j], c)*x[j]*x[j] + Profile(x[j-1], c)*x[j-1]*x[j-1])*(x[j] - x[j-1])/2.0;
            if(Mh == 0.0 && x[j] >= 1.0)
                Mh = int[j];
            if(Mh > 0.0 && Int[j] >= ((double) M_frac)*Mh){
                rmax = x[j];
                break;
            }
        }
		gsl_spline_init(spline_I, x, Int, NRs);
		A = gsl_spline_eval(spline_I, rmax, acc);

		/*Run over different values of epsilon*/
		r[0] = R_MIN;
		for(j=1;j<Neps-1;j++){
            // Set the guess to the last solution
			r[j] = r[j-1];
			rtmp = r[j] + 1.0;

			/*Find the solution of the equation F^{-1}(r) = Eps*/
			while(fabs(r[j] - rtmp) > Tot){
				rtmp = r[j];
				r[j] = r[j] - (gsl_spline_eval(spline_I, r[j], acc)/A - Eps[j])/(Profile(r[j], c)*r[j]*r[j]/A);

				if(r[j] >= rmax || r[j] <= R_MIN)	r[j] = Eps[j];
			}
		}
		r[Neps-1] = rmax;

		/*Interpolate the r(Eps) relation*/
		gsl_spline_init(spline_r[i], Eps, r, Neps);
	}
}

/*Compute the radial distance given epsilon*/
fft_real Generate_r(size_t ind_h, fft_real w1, fft_real w2, gsl_spline **spline_r, gsl_interp_accel *acc, fft_real Eps){
	fft_real r, r1, r2;

	/*Compute the value of the radial coordinate for the two bin ends*/
	r1 = gsl_spline_eval(spline_r[ind_h], Eps, acc);
	r2 = gsl_spline_eval(spline_r[ind_h+1], Eps, acc);

	/*Interpolate between the two values*/
	r = r1*w1 + r2*w2;

	return r;
}

/*Compute the number of galaxies and their positions and velocities in each halo*/
size_t Populate_total(size_t nh, fft_real *posh, fft_real *velh, fft_real *Massh, fft_real *Ch, fft_real *posg, fft_real *velg, long *gal_type, gsl_rng *rng_ptr){
    int j, Ncen, Nsat, Ngals;
    size_t i, count, ind_h;
    fft_real Rv, phi, theta, r, cmin, cmax, w1, w2;

	/*Compute the concentrations*/
	if(out.IN_C == FALSE)
		for(i=0;i<nh;i++)
			Ch[i] = f_c(Massh[i]);

	/*Find the minimum and maximum concentration*/
	cmin = Ch[0];
	cmax = Ch[0];
	for(i=1;i<nh;i++){
		if(Ch[i] < cmin)	cmin = Ch[i];
		if(Ch[i] > cmax)	cmax = Ch[i];
	}

	/*Construct the interpolator used to generate the radial coordinate*/
	gsl_interp_accel *acc = gsl_interp_accel_alloc();
	gsl_spline **spline_r;
	spline_r = (gsl_spline **) malloc(NCs*sizeof(spline_r));

	for(i=0;i<NCs;i++)
		spline_r[i] = gsl_spline_alloc(gsl_interp_cspline, Neps);

	Interpolate_r_Eps(cmin, cmax, spline_r, acc, 100.0);

	/*Populate all halos*/
    count = 0;
	for(i=0;i<nh;i++){
		/*Compute the number of central and satellite galaxies*/
		if(Ncentral_tot(Massh[i]) >= gsl_rng_uniform(rng_ptr))
			Ncen = 1;
		else
			Ncen = 0;		
		Nsat = gsl_ran_poisson(rng_ptr, (double) Nsatellite_tot(Massh[i]));
		Ngals = Ncen + Nsat;
		if(Ngals == 0) continue;

		/*Save the central galaxy*/
		if(Ncen == 1){
            posg[3*count] = posh[3*i];
            posg[3*count+1] = posh[3*i+1];
            posg[3*count+2] = posh[3*i+2];
			if(out.OUT_VEL == TRUE){
				velg[3*count] = velh[3*i];
				velg[3*count+1] = velh[3*i+1];
				velg[3*count+2] = velh[3*i+2];
			}
            if (out.OUT_TYPE == TRUE)
                gal_type[count] = - (long) i;

			count ++;
		}
		if(Nsat == 0)	continue;

		/*Compute the radius of this halo and its position in the concentration bins*/
		ind_h = floor((log10(Ch[i]) - log10(cmin))/(log10(cmax) - log10(cmin)));
		if(ind_h == NCs)	ind_h = ind_h - 1;
		w1 = Ch[i] - pow(10.0, log10(cmin) + (log10(cmax) - log10(cmin))*ind_h/(NCs - 1));
		w2 = pow(10.0, log10(cmin) + (log10(cmax) - log10(cmin))*(ind_h+1)/(NCs - 1)) - Ch[i];
		Rv = w1 + w2;
		w1 = w1/Rv;
		w2 = w2/Rv;
		Rv = pow(3.0*Massh[i]/(4.0*M_PI*cosmo.Dv*cosmo.rhomz), 1.0/3.0);

        /*Put each satellite galaxy following the given profile*/
		for(j=0;j<Nsat;j++){
			phi = 2.0*M_PI*gsl_rng_uniform(rng_ptr);	
			theta = M_PI*gsl_rng_uniform(rng_ptr);

			r = Generate_r(ind_h, w1, w2, spline_r, acc, (fft_real) gsl_rng_uniform(rng_ptr));

			posg[3*count] = cysumf(posh[3*i], r*sin(theta)*cos(phi), box.L[0]); 
			posg[3*count+1] = cysumf(posh[3*i+1], r*sin(theta)*sin(phi), box.L[1]);
			posg[3*count+2] = cysumf(posh[3*i+2], r*cos(theta), box.L[2]);
			if(out.OUT_VEL == TRUE){
				velg[3*count] = velh[3*i];
				velg[3*count+1] = velh[3*i+1];
				velg[3*count+2] = velh[3*i+2];
			}
            if (out.OUT_TYPE == TRUE)
                gal_type[count] = (long) i;

			count ++;
		}

		if(count > Ng_max*nh){
			printf("Maximum number of galaxies reached! Increase Ng_max in hod_h.h!\n");
			exit(0);
		}
	}
	for(i=0;i<NCs;i++)
		gsl_spline_free(spline_r[i]);	
	free(spline_r);
	gsl_interp_accel_free(acc);

    return count;
}

// Populate the halos with particles
size_t Populate_Particles(size_t nh, fft_real *posh, fft_real *velh, fft_real *Massh, fft_real *Ch, fft_real *posg, fft_real *velg, long *gal_type, fft_real massp, fft_real M_frac, size_t np, gsl_rng *rng_ptr){
    int j, Ncen, Nsat, Ngals;
    size_t i, count, ind_h;
    fft_real Rv, phi, theta, r, cmin, cmax, w1, w2, np;

	/*Compute the concentrations*/
	if(out.IN_C == FALSE)
		for(i=0;i<nh;i++)
			Ch[i] = f_c(Massh[i]);

	/*Find the minimum and maximum concentration*/
	cmin = Ch[0];
	cmax = Ch[0];
	for(i=1;i<nh;i++){
		if(Ch[i] < cmin)	cmin = Ch[i];
		if(Ch[i] > cmax)	cmax = Ch[i];
	}

	/*Construct the interpolator used to generate the radial coordinate*/
	gsl_interp_accel *acc = gsl_interp_accel_alloc();
	gsl_spline **spline_r;
	spline_r = (gsl_spline **) malloc(NCs*sizeof(spline_r));

	for(i=0;i<NCs;i++)
		spline_r[i] = gsl_spline_alloc(gsl_interp_cspline, Neps);

    // Interpolate the r(eps) function used in the computation of the radial coordinate
	Interpolate_r_Eps(cmin, cmax, spline_r, acc, M_frac);

	/*Populate all halos*/
    count = 0;
	for(i=0;i<nh;i++){
        // Compute the number of particles in this halo
        np = Massh[i]/massp;

		/*Compute the number of central and satellite galaxies*/
		if(np >= gsl_rng_uniform(rng_ptr))
			Ncen = 1;
		else
			Ncen = 0;		
		Nsat = gsl_ran_poisson(rng_ptr, (double)(np - Ncen));
		Ngals = Ncen + Nsat;
		if(Ngals == 0) continue;

		/*Save the central galaxy*/
		if(Ncen == 1){
            posg[3*count] = posh[3*i];
            posg[3*count+1] = posh[3*i+1];
            posg[3*count+2] = posh[3*i+2];
			if(out.OUT_VEL == TRUE){
				velg[3*count] = velh[3*i];
				velg[3*count+1] = velh[3*i+1];
				velg[3*count+2] = velh[3*i+2];
			}
            if (out.OUT_TYPE == TRUE)
                gal_type[count] = - (long) i;

			count ++;
		}
		if(Nsat == 0)	continue;

		/*Compute the radius of this halo and its position in the concentration bins*/
		ind_h = floor((log10(Ch[i]) - log10(cmin))/(log10(cmax) - log10(cmin)));
		if(ind_h == NCs)	ind_h = ind_h - 1;
		w1 = Ch[i] - pow(10.0, log10(cmin) + (log10(cmax) - log10(cmin))*ind_h/(NCs - 1));
		w2 = pow(10.0, log10(cmin) + (log10(cmax) - log10(cmin))*(ind_h+1)/(NCs - 1)) - Ch[i];
		Rv = w1 + w2;
		w1 = w1/Rv;
		w2 = w2/Rv;
		Rv = pow(3.0*Massh[i]/(4.0*M_PI*cosmo.Dv*cosmo.rhomz), 1.0/3.0);

        /*Put each satellite galaxy following the given profile*/
		for(j=0;j<Nsat;j++){
			phi = 2.0*M_PI*gsl_rng_uniform(rng_ptr);	
			theta = M_PI*gsl_rng_uniform(rng_ptr);

			r = Generate_r(ind_h, w1, w2, spline_r, acc, (fft_real) gsl_rng_uniform(rng_ptr));

			posg[3*count] = cysumf(posh[3*i], r*sin(theta)*cos(phi), box.L[0]); 
			posg[3*count+1] = cysumf(posh[3*i+1], r*sin(theta)*sin(phi), box.L[1]);
			posg[3*count+2] = cysumf(posh[3*i+2], r*cos(theta), box.L[2]);
			if(out.OUT_VEL == TRUE){
				velg[3*count] = velh[3*i];
				velg[3*count+1] = velh[3*i+1];
				velg[3*count+2] = velh[3*i+2];
			}
            if (out.OUT_TYPE == TRUE)
                gal_type[count] = (long) i;

			count ++;
		}

        // Stop if the number of particles was reached
        if(count == np)
            break;
	}

    // Put some extra particles to get np particles
    while(count < np){
        posg[3*count] = ((fft_real) gsl_rng_uniform(rng_ptr))*box.L[0]; 
        posg[3*count+1] = ((fft_real) gsl_rng_uniform(rng_ptr))*box.L[1];
        posg[3*count+2] = ((fft_real) gsl_rng_uniform(rng_ptr))*box.L[2];
        if(out.OUT_VEL == TRUE){
            velg[3*count] = 0.0;
            velg[3*count+1] = 0.0;
            velg[3*count+2] = 0.0;
        }
        if (out.OUT_TYPE == TRUE)
            gal_type[count] = (long) -np;
        count ++;
    }

    // Free memory used by the interpolators
	for(i=0;i<NCs;i++)
		gsl_spline_free(spline_r[i]);	
	free(spline_r);
	gsl_interp_accel_free(acc);

    return count;
}
