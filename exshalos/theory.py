from tkinter.messagebox import NO
import numpy as np
from scipy.integrate import simps, odeint
from scipy.special import binom
import exshalos

#Get the mass of each cell given its size
def Get_Mcell(Om0 = 0.31, Lc = 2.0):
	return 2.775e+11*Om0*np.power(Lc, 3.0)

#Get the size of each cell given its mass
def Get_Lc(Om0 = 0.31, Mcell = 8.5e+10):
	return np.power(Mcell/(2.775e+11*Om0), 1.0/3.0)

#Return the value of the matter overdensity in a given redshift
def Get_Omz(z = 0.0, Om0 = 0.31):
	return Om0*pow(1.0 + z, 3.0)/(Om0*pow(1.0 + z, 3.0) + (1.0 - Om0))

#Return the the value of deltac following a fit
def Get_deltac(z = 0.0, Om0 = 0.31):
	return 1.686*pow(Get_Omz(z, Om0), 0.0055)

#Return the Hubble function in units of 100*h
def Get_Hz(z = 0.0, Om0 = 0.31):
	return np.sqrt(Om0*np.power(1.0 + z, 3.0) + (1.0 - Om0))

#Return the Hubble function in units of 100*h
def Get_Ha(a = 1.0, Om0 = 0.31):
	return np.sqrt(Om0*np.power(a, -3.0) + (1.0 - Om0)) 

#Return the derivative of the Hubble's function in units of 100*h
def Get_dHa(a = 1.0, Om0 = 0.31):
        return -1.5*Om0*np.power(a, -4.0)/Get_Ha(a, Om0)

#Return the growth rate
def Get_fz(z = 0.0, Om0 = 0.31):
	return np.power(Get_Omz(z, Om0), 0.5454)

#Define the system of differential equations used to compute the growth function
def Growth_eq(y, a, Om0):
        d, v = y

        return np.array([v, -(3.0/a + Get_dHa(a, Om0)/Get_Ha(a, Om0))*v + 3.0/2.0*Om0/(np.power(Get_Ha(a, Om0), 2.0)*np.power(a, 5.0))*d])

#Return the growth function
def Get_Dz(Om0 = 0.31, zmax = 1000, zmin = -0.5, nzs = 1000):
	resp = {}

	#Set the initial conditions
	a = np.logspace(np.log10(1.0/(zmax + 1.0)), np.log10(1.0/(zmin + 1.0)), nzs)
	resp['z'] = 1.0/a - 1.0
	resp['a'] = a
	d0 = a[0]
	dd0 = 1.0
	y0 = [d0, dd0]

	#Solve the Growth function equation
	sol = odeint(Growth_eq, y0, a, args=(Om0,))
	resp['Dz'] = sol[:,0]
	resp['dDz'] = sol[:,1]

	return resp

#Window function in Fourier space
def W(k, R):
	resp = 3.0/(np.power(k*R,2))*(np.sin(k*R)/(k*R) - np.cos(k*R))

	return resp

#Compute the variance of the linear density field
def Compute_sigma(k, P, R = None, M = None, Om0 = 0.31, z = 0.0):

	#Compute R(M)
	if(R is None):
		if(M is None):
			raise ValueError("You have to give the mass or radius array!")
		else:
			R = np.power(3.0*M/(4.0*np.pi*2.775e+11*Om0*np.power(1+z, 3.0)), 1.0/3.0)

	#Evaluate sigma
	Nr = len(R)
	sigma = np.zeros(Nr)
	for j in range(Nr):
		kt = k[k<=2.0*np.pi/R[j]]
		Pt = P[k<=2.0*np.pi/R[j]]
		
		sigma[j] = np.sqrt(simps(Pt*kt*kt*np.power(W(kt, R[j]), 2.0), kt)/(2.0*np.pi*np.pi))

	return sigma


#Compute the derivative of sigma with respect to M
def dlnsdlnm(M, sigma):
	resp = np.zeros(len(M))

	resp[0] = (np.log(sigma[1]) - np.log(sigma[0]))/(np.log(M[1]) - np.log(M[0]))
	resp[1:-1] = (np.log(sigma[2:]) - np.log(sigma[:-2]))/(np.log(M[2:]) - np.log(M[:-2]))
	resp[-1] = (np.log(sigma[-1]) - np.log(sigma[-2]))/(np.log(M[-1]) - np.log(M[-2]))

	return resp

#Multiplicity function
def fh(s, model = 0, theta = None, delta_c = -1, Om0 = 0.31, z = 0.0):
	if(delta_c < 0.0):
		delta_c = Get_deltac(Get_Omz(Om0 = Om0, z = z))
	nu = delta_c/s
	resp = np.zeros(len(s))

	#Press-Schechter
	if(model == 0 or model == "ps" or model == "PS" or model == "1SB"):	
		resp = np.sqrt(2.0/np.pi)*nu*np.exp(-nu*nu/2)

	#Sheth-Tormen
	elif(model == 1 or model == "ST" or model == "st" or model == "elliptical"):
		if(theta != None):
			a, b, p = theta
		else:
			a, b, p = np.array([0.7, 0.4, 0.6])

		B = np.sqrt(a)*delta_c*(1.0 + b*np.power(a*nu*nu, -p))
		A = 0.0
		for i in range(6):
			A += np.power(-1, i)*binom(p, i)

		resp = np.sqrt(2.0*a/np.pi)*nu*np.exp(-B*B/(2.0*s*s))*(1.0 + b*A*np.power(a*nu*nu, -p))
	
	#Tinker
	elif(model == 2 or model == "Tinker" or model == "tinker" or model == "TINKER"):
		if(theta != None):
			Delta = theta
		else:
			Delta = 300

		#Tinker Delta = 200
		if(Delta == 200):
			B = 0.482
			d = 1.97
			e = 1.0
			f = 0.51
			g = 1.228
	
		#Tinker Delta = 300
		elif(Delta == 300):
			B = 0.466
			d = 2.06
			e = 0.99
			f = 0.48
			g = 1.310
	
		#Tinker Delta = 400
		elif(Delta == 400):
			B = 0.494
			d = 2.30
			e = 0.93
			f = 0.48
			g = 1.403
	
		resp = B*(np.power(s/e, -d) + np.power(s, -f))*np.exp(-g/(s*s))

	#Linear difusive barrier
	elif(model == 3 or model == "2LDB"):
		if(theta != None):
			b, D, dv, J_max = theta
		else:
			b, D, dv, J_max = np.array([0.0, 0.0, 2.71, 20])

		resp = np.zeros(len(s))
		dt = delta_c + dv

		for n in range(1,J_max+1):
			resp += 2.0*(1.0+D)*np.exp(-b*b*s*s/(2.0*(1.0+D)))*np.exp(-b*delta_c/(1.0+D))*(n*np.pi/(dt*dt))*s*s*np.sin(n*np.pi*delta_c/dt)*np.exp(-n*n*np.pi*np.pi*s*s*(1.0+D)/(2.0*dt*dt))

	return resp

#Halo mass function
def dlnndlnm(M, sigma = None, model = 0, theta = None, delta_c = -1, Om0 = 0.31, z = 0.0, k = None, P = None):
	rhoc = 2.775e+11
	rhom = Om0*rhoc*np.power(1+z, 3)

	if(sigma is None):
		sigma = Compute_sigma(k, P, M = M, Om0 = Om0, z = z)

	return -fh(sigma, model, theta, delta_c, Om0, z)*rhom/M*dlnsdlnm(M, sigma)

#Halo bias of first order
def bh1(M, s = None, model = 0, theta = None, delta_c = -1, Om0 = 0.31, z = 0.0, k = None, P = None, Lagrangian = False):
	if(s is None):
		s = Compute_sigma(k, P, M = M, Om0 = Om0, z = z)

	if(delta_c < 0.0):
		delta_c = Get_deltac(Get_Omz(Om0 = Om0, z = z))
	nu = delta_c/s
	resp = np.zeros(len(s))

	#Press-Schechter
	if(model == 0 or model == "ps" or model == "PS" or model == "1SB"):	
		resp = 1.0 + (nu*nu - 1.0)/delta_c

	#Sheth-Tormen
	elif(model == 1 or model == "ST" or model == "st" or model == "elliptical"):
		if(theta != None):
			a, b, p = theta
		else:
			a, b, p = np.array([0.7, 0.4, 0.6])

		A = 0.0
		for i in range(6):
			A += np.power(-1, i)*binom(p, i)

		resp = 1.0 + np.sqrt(a)*nu*nu/delta_c*(1.0 + b*np.power(a*nu*nu, -p)) - 1.0/(np.sqrt(a)*delta_c*(1.0 + A*np.power(a*nu*nu, -p)))

	#Tinker
	elif(model == 2 or model == "Tinker" or model == "tinker" or model == "TINKER"):
		if(theta != None):
			Del = theta
		else:
			Del = 300

		y = np.log10(Del)
		A = 1.0 + 0.24*y*np.exp(-(4.0/y)**4)
		a = 0.44*y - 0.88
		B = 0.183
		b = 1.5
		C = 0.019 + 0.107*y + 0.19*np.exp(-(4.0/y)**4)
		c = 2.4

		resp = 1.0 - A*np.power(nu,a)/(np.power(nu,a) + pow(delta_c,a)) + B*np.power(nu,b) + C*np.power(nu,c)

	#Linear difusive barrier
	elif(model == 3 or model == "2LDB"):
		if(theta != None):
			b, D, dv, J_max = theta
		else:
			b, D, dv, J_max = np.array([0.0, 0.0, 2.71, 20])

		resp = np.zeros(len(s))
		tmp = np.zeros(len(s))
		dt = delta_c + dv

		#Halos
		for n in range(1,J_max+1):
			resp -= (n*np.pi/(dt*dt))*np.sin(n*np.pi*delta_c/dt)*np.exp(-n*n*np.pi*np.pi*s*s*(1.0+D)/(2.0*dt*dt))*(np.power(np.tan(n*np.pi*delta_c/dt), -1.0)*(n*np.pi/dt) - b/(1.0 + D))

		for n in range(1,J_max+1):
			tmp += (n*np.pi/(dt*dt))*np.sin(n*np.pi*delta_c/dt)*np.exp(-n*n*np.pi*np.pi*s*s*(1.0+D)/(2.0*dt*dt))

		resp = np.ones(len(s)) + resp/tmp

	#COnvert to the Lagrangian bias
	if(Lagrangian == True):
		resp -= 1.0

	return resp

#Halo bias of second order
def bh2(M, s = None, model = 0, theta = None, delta_c = -1, Om0 = 0.31, z = 0.0, k = None, P = None, Lagrangian = False, b1 = None):
	if(s is None):
		s = Compute_sigma(k, P, M = M, Om0 = Om0, z = z)

	if(delta_c < 0.0):
		delta_c = Get_deltac(Get_Omz(Om0 = Om0, z = z))
	nu = delta_c/s
	S = s**2
	resp = np.zeros(len(s))

	#Press-Schechter
	if(model == 0 or model == "ps" or model == "PS" or model == "1SB"):	
		resp = np.power(nu*nu/delta_c, 2.0) - 3.0*np.power(nu/delta_c, 2.0)

	#Sheth-Tormen
	elif(model == 1 or model == "ST" or model == "st" or model == "elliptical"):
		if(theta != None):
			a, b, p = theta
		else:
			a, b, p = np.array([0.7, 0.4, 0.6])

		A = 0.0
		for i in range(6):
			A += np.power(-1, i)*binom(p, i)

		B = np.sqrt(a)*delta_c*(1.0 + b*np.power(a*nu*nu, -p))
		BP = np.sqrt(a)*delta_c*(1.0 + A*np.power(a*nu*nu, -p))

		resp = np.power(B/S, 2.0) - 1.0/S - 2.0*B/(S*BP)

	#Matteo
	elif(model == 2 or model == "matteo" or model == "Matteo"):
		if(b1 is None):
			b1 = bh1(M, s = s, model = 2, theta = 330, delta_c = delta_c, Om0 = Om0, z = z, k = k, P = P, Lagrangian = True)
		resp = -0.09143*b1**3 + 0.7093*b1**2 - 0.2607*b1 - 0.3469

	#Lazeyras
	elif(model == 3 or model == "Lazeyras" or model == "lazeyras"):
		if(b1 is None):
			b1 = bh1(M, s = s, model = 2, theta = 330, delta_c = delta_c, Om0 = Om0, z = z, k = k, P = P, Lagrangian = True)
		resp = 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3
		#resp = 2.0*resp - 4.0/21.0*b1 	

	if(Lagrangian == False):
		if(b1 is None):
			b1 = bh1(M, s = s, model = model, theta = theta, delta_c = delta_c, Om0 = Om0, z = z, k = k, P = P, Lagrangian = True)
		resp = 4.0/21.0*b1 + 1.0/2.0*resp

	return resp

#Halo bias of third order
def bh3(M, s = None, model = 0, theta = None, delta_c = -1, Om0 = 0.31, z = 0.0, k = None, P = None, Lagrangian = False, bs2 = 0.0):
	if(s is None):
		s = Compute_sigma(k, P, M = M, Om0 = Om0, z = z)

	if(delta_c < 0.0):
		delta_c = Get_deltac(Get_Omz(Om0 = Om0, z = z))
	nu = delta_c/s
	S = s**2
	resp = np.zeros(len(s))

	#Press-Schechter
	if(model == 0 or model == "ps" or model == "PS" or model == "1SB"):	
		resp = np.power(delta_c/S, 3.0) - 6.0*delta_c/np.power(S, 2.0)  + 3.0/S/delta_c

	#Sheth-Tormen
	elif(model == 1 or model == "ST" or model == "st" or model == "elliptical"):
		if(theta != None):
			a, b, p = theta
		else:
			a, b, p = np.array([0.7, 0.4, 0.6])

		A = 0.0
		for i in range(6):
			A += np.power(-1, i)*binom(p, i)

		B = np.sqrt(a)*delta_c*(1.0 + b*np.power(a*nu*nu, -p))
		BP = np.sqrt(a)*delta_c*(1.0 + A*np.power(a*nu*nu, -p))

		resp = np.power(B/S, 3.0) - 3.0*B/np.power(S, 2.0) - 3.0*B*B/(S*S*BP) + 3.0/(S*BP)

	if(Lagrangian == False):
		b2 = bh2(M, s = s, model = model, theta = theta, delta_c = delta_c, Om0 = Om0, z = z, k = k, P = P, Lagrangian = True)

		resp = - 1.0/2.0*b2 + 1.0/6.0*resp - 2.0/3.0*bs2 

	return resp

#Compute the power spectra using CLPT at first order
def CLPT_Powers(k, P, Lambda = 0.7, kmax = 0.7, nmin = 5, nmax = 10, verbose = False):
    """
    k: Wavebumber of the power spectrum | 1D numpy array
    P: Linear power spectrum | 1D numpy array
	Lambda: Scale to be used to smooth the power spectrum | float
	kmax: Maximum wavenumber of the outputs | float
	nmin: Maximum order used in the full computation of the terms of the expansion | int
	nmax: Maximum order used in the Limber approximation of the terms of the expansion | int
    verbose: Output or do not output information in the c code | boolean

    return: The power spectra of the operators | Dictonaty with 3 arrays. "k": wavenumbers, "Plin": linear power spectrum used as input, "P11": Result for the 11 power spectrum
   	"""

    x = exshalos.analytical.analytical.clpt_compute(k.astype("float64"), P.astype("float64"), np.float64(Lambda), np.float64(kmax), np.int32(nmin), np.int32(nmax), np.int32(verbose))

    return x

#Compute the generalized corraletion functions (Xi_lm)
def Xi_lm(r, k, P, Lambda = 0.7, l = 0, mk = 2, mr = 0, K = 11, alpha = 4.0, Rmax = 1.0, verbose = False):
	"""
	r: Radial distances for the output | 1D numpy array
    k: Wavebumber of the power spectrum | 1D numpy array
    P: Linear power spectrum | 1D numpy array
	Lambda: Scale to be used to smooth the power spectrum | float
	l: Order of the spherical Bessel's function | int
	mk: Power of k in the integral | int
	mr: Power of r in the integral | int
	K: Number of points used by the Gaussian smooth | int
	alpha: Value of alpha used by the Gaussian smooth | float
	verbose: Output or do not output information in the c code | boolean
	
	return: The generalized correlation function xi_lm = \int dk k^mk r^mr P(k) j_l(kr) | 1D numpy array
	"""

	x = exshalos.analytical.analytical.xilm_compute(r.astype("float64"), k.astype("float64"), P.astype("float64"), np.float64(Lambda), np.int32(l), np.int32(mk), np.int32(mr), np.int32(K), np.float64(alpha), np.float64(Rmax), np.int32(verbose))

	return x

#Compute the 1-loop matter or galaxy power spectrum using classPT
def Pgg_EFTofLSS(k = None, parameters = {}, b = None, cs = None, c = None, IR_resummation = True, cb = True, RSD = True, AP = False, Om_fid = 0.31, z = 0.0, ls = [0, 2, 4], pk_mult = None, fz = None, OUT_MULT = False, h_units = True):
	"""
	parameters: Cosmological parameters used by class | dictionary
	b: Values of the bias parameters (b1, b2, bG2, bGamma3, b4)| 1D or 2D (multitracers) array
	IR_resummation: Option to do the IR resummation of the spectrum | boolean
	cb: Option to add baryions | boolean
	RSD: Option to give the power spectrum in redshift space | boolean
	AP: Optino to use the AP | boolean
	Om_fid: Omega matter fiducial for the AP correction | float
	z: Redshift of the power spectrum | float
	ls: The multipoles to be computed [0, 2, 4] | list or int

	return:
	"""

	#Compute the power spectra using classPT
	if(pk_mult == None):
		from classy import Class

		#Set the parameters
		M = Class()
		params = {'A_s':2.089e-9, 'n_s':0.9649, 'tau_reio':0.052, 'omega_b':0.02237, 'omega_cdm':0.12, 'h':0.6736, 'YHe':0.2425, 'N_ur':2.0328, 'N_ncdm':1, 'm_ncdm':0.06}
		for key in parameters.keys():
			params[key] = parameters[key]
		params['z_pk'] = z
		M.set(params)
		if(cb == True):
			cb = "Yes"
		else:
			cb = "No"
		if(AP == True):
			AP = "Yes"
		else:
			AP = "No"
		if(IR_resummation == True):
			IR_resummation = "Yes"
		else:
			IR_resummation = "No"
		if(RSD == True):
			M.set({'output':'mPk', 'non linear':'PT', 'IR resummation':IR_resummation, 'Bias tracers':'Yes', 'cb':cb, 'RSD':'Yes', 'AP':AP, 'Omfid':Om_fid})
		else:
			M.set({'output':'mPk', 'non linear':'PT', 'IR resummation':IR_resummation, 'Bias tracers':'Yes', 'cb':cb, 'RSD':'No'})
		M.compute()

		#Compute the spectra of the basis
		if(k is None):
			raise TypeError("You have to give an array of k where to compute the power spectrum")
		h = M.h()
		kh = k*h
		fz = M.scale_independent_growth_factor_f(z)
		M_mult = M.get_pk_mult(kh, z, len(kh))
       
		#Save a dictionary with the spectra
		pk_mult = {}
		spectra_label = ["Id2d2", "Id2", "IG2", "Id2G2", "IG2G2", "FG2", "ctr", "lin", "1loop",]
		spectra_ind = [1, 2, 3, 4, 5, 6, 10, 14, 0]
		for i in range(len(spectra_label)):
			pk_mult[spectra_label[i]] = M_mult[spectra_ind[i]]
		if(RSD == True):
			spectra_label = ["FG2_0b1", "FG2_0", "FG2_2", "ctr_0", "ctr_2", "ctr_4"]
			spectra_ind = [7, 8 , 9, 11, 12, 13]
			for i in range(len(spectra_label)):
				pk_mult[spectra_label[i]] = M_mult[spectra_ind[i]]	
			spectra_label = ["lin_0_vv", "lin_0_vd", "lin_0_dd", "lin_2_vv", "lin_2_vd", "lin_4_vv", "1loop_0_vv", "1loop_0_vd", "1loop_0_dd", "1loop_2_vv", "1loop_2_vd", "1loop_2_dd", "1loop_4_vv", "1loop_4_vd", "1loop_4_dd", "Idd2_0", "Id2_0", "IdG2_0", "IG2_0", "Idd2_2", "Id2_2", "IdG2_2", "IG2_2", "Id2_4", "IG2_4"]
			for i in range(len(spectra_label)):
				pk_mult[spectra_label[i]] = M_mult[15+i]	

	else:
		h = 1
		if(fz == None):
			fz = pow((Om_fid*pow(1+z, 3.0))/(Om_fid*pow(1+z, 3.0) + 1.0 - Om_fid), 0.5454)
		if(RSD == True and len(pk_mult.keys()) < 10):
			raise ValueError("There are not all spectra needed for the computations in redshift space")

	#Get the number of tracers
	if(b is None):
		raise TypeError("You have to give an array with the values of the bias parameters")
	if(len(b.shape) == 1):
		Ntracers = 1
		b = np.array(b).reshape([1, len(b)])
		cs = np.array([cs])
		if(RSD == True):
			c = np.array(c).reshape([1, c.shape[0], c.shape[1]])
		else:
			c = np.array(c).reshape([1, c.shape[0]])
	else: 
		Ntracers = b.shape[0]

	#Set all combinations of the bias parameters 
	#(b1, b1^2, b2^2, b1*b2, b2, b1*bG2, bG2, b2*bG2, bG^2, b1*bGamma3, bGamma3, b4, b1*b4, b1^2*b4)
	if(RSD == True):
		bias = np.zeros([int(Ntracers*(Ntracers+1)/2), 14])
		ctrs = np.zeros([int(Ntracers*(Ntracers+1)/2), 3])
		count = 0
		for i in range(Ntracers):
			for j in range(i):
				bias[count, :] = np.array([(b[i,0] + b[j,0])/2.0, b[i,0]*b[j,0], b[i,1]*b[j,1], (b[i,0]*b[j,1] + b[i,1]*b[j,0])/2.0, (b[i,1] + b[j,1])/2.0, (b[i,0]*b[j,2] + b[i,2]*b[j,0])/2.0, (b[i,2] + b[j,2])/2.0, (b[i,1]*b[j,2] + b[i,2]*b[j,1])/2.0, b[i,2]*b[j,2], (b[i,0]*b[j,3] + b[i,3]*b[j,0])/2.0, (b[i,3] + b[j,3])/2.0, (b[i,4] + b[j,4])/2.0, (b[i,0]*b[j,4] + b[i,4]*b[j,0])/2.0, (b[i,0]**2*b[j,4] + b[i,4]*b[j,0]**2)/2.0])
				for l in range(3):
					ctrs[count, l] = (cs[i, l]*b[j,0] + cs[j, l]*b[i,0])/2.0
				count += 1
			bias[count, :] = np.array([b[i,0], b[i,0]**2, b[i,1]**2, b[i,0]*b[i,1], b[i,1], b[i,0]*b[i,2], b[i,2], b[i,1]*b[i,2], b[i,2]**2, b[i,0]*b[i,3], b[i,3], b[i,4], b[i,0]*b[i,4], b[i,0]**2*b[i,4]])
			for l in range(3):
				ctrs[count, l] = cs[i, l]*b[i,0]
			count += 1		
	#(b1^2, b1*b2, b1*bG2, b1*bGamma3, b2^2, bG2^2, b2*bG2)
	else:
		bias = np.zeros([int(Ntracers*(Ntracers+1)/2), 7])
		ctrs = np.zeros(int(Ntracers*(Ntracers+1)/2))
		count = 0
		for i in range(Ntracers):
			for j in range(i):
				bias[count, :] = np.array([b[i,0]*b[j,0], (b[i,0]*b[j,1] + b[i,1]*b[j,0])/2.0, (b[i,0]*b[j,2] + b[i,2]*b[j,0])/2.0, (b[i,0]*b[j,3] + b[i,3]*b[j,0])/2.0, b[i,1]*b[j,1], b[i,2]*b[j,2], (b[i,1]*b[j,2] + b[i,2]*b[j,1])/2.0])
				ctrs[count] = (cs[i]*b[j,0] + cs[j]*b[i,0])/2.0
				count += 1
			bias[count, :] = np.array([b[i,0]**2, b[i,0]*b[i,1], b[i,0]*b[i,2], b[i,0]*b[i,3], b[i,1]**2, b[i,2]**2, b[i,1]*b[i,2]])
			ctrs[count] = cs[i]*b[i,0]
			count += 1

				
	#Define the functions to compute each power spectra
	#Compute Pgg in real space
	def Pgg(ind):
		resp = (bias[ind,0]*(pk_mult["lin"] + pk_mult["1loop"]) + bias[ind,1]*pk_mult["Id2"] + 2.0*bias[ind,2]*pk_mult["IG2"] + 2.0*bias[ind, 2]*pk_mult["FG2"] + 0.8*bias[ind, 3]*pk_mult["FG2"] + 0.25*bias[ind, 4]*pk_mult["Id2d2"] + bias[ind, 5]*pk_mult["IG2G2"] + bias[ind, 6]*pk_mult["Id2G2"])*pow(h, 3.0*h_units) + 2.0*ctrs[ind]*pk_mult["ctr"]*pow(h, h_units)
		for i in range(len(c[ind,:])):
			resp += c[ind,i]*np.power(k, 2*i)

		return resp

	#Compute the monopole of the power spectrum
	def Pgg_l0(ind):
		resp =  (pk_mult["lin_0_vv"] + pk_mult["1loop_0_vv"] + bias[ind,0]*(pk_mult["lin_0_vd"] + pk_mult["1loop_0_vd"]) + bias[ind,1]*(pk_mult["lin_0_dd"] + pk_mult["1loop_0_dd"]) + 0.25*bias[ind,2]*pk_mult["Id2d2"] + bias[ind,3]*pk_mult["Idd2_0"] + bias[ind,4]*pk_mult["Id2_0"] + bias[ind,5]*pk_mult["IdG2_0"] + bias[ind,6]*pk_mult["IG2_0"] + bias[ind,7]*pk_mult["Id2G2"] + bias[ind,8]*pk_mult["IG2G2"] + 2.0*bias[ind,5]*pk_mult["FG2_0b1"] + 2.0*bias[ind,6]*pk_mult["FG2_0"] + 0.8*bias[ind,9]*pk_mult["FG2_0b1"] + 0.8*bias[ind,10]*pk_mult["FG2_0"])*pow(h, 3.0*h_units) + 2.0*ctrs[ind,0]*pk_mult["ctr_0"]*pow(h, h_units) + fz**2*np.power(k, 2.0)*35/8.0*pk_mult["ctr_4"]*(1.0/9.0*bias[ind,11]*fz**2 + 2.0/7.0*fz*bias[ind,12] + 1.0/5.0*bias[ind,13])*pow(h, h_units)
		for i in range(len(c[ind,:,0])):
			resp += c[ind,i,0]*np.power(k, 2*i)

		return resp

	#Compute the quadrupole of the power spectrum
	def Pgg_l2(ind):
		resp = (pk_mult["lin_2_vv"] + pk_mult["1loop_2_vv"] + bias[ind,0]*(pk_mult["lin_2_vd"] + pk_mult["1loop_2_vd"]) + bias[ind,1]*pk_mult["1loop_2_dd"] + bias[ind,3]*pk_mult["Idd2_2"] + bias[ind,4]*pk_mult["Id2_2"] + bias[ind,5]*pk_mult["IdG2_2"] + bias[ind,6]*pk_mult["IG2_2"] + (2.0*bias[ind,6] + 0.8*bias[ind,10])*pk_mult["FG2_2"])*pow(h, 3.0*h_units) + 2.0*ctrs[ind,1]*pk_mult["ctr_2"]*pow(h, h_units) + fz**2*np.power(k, 2.0)*35/8.0*pk_mult["ctr_4"]*(70.0*bias[ind,11]*fz**2 + 165.0*fz*bias[ind,12] + 99.0*bias[ind,13])*(4.0/693.0)*pow(h, h_units) + c[ind,0,1] + c[ind,1,1]*np.power(k, 2.0)
		for i in range(len(c[ind,:,1])):
			resp += c[ind,i,1]*np.power(k, 2*i)

		return resp

	#Compute the hexadecapole of the power spectrum
	def Pgg_l4(ind):
		resp = (pk_mult["lin_4_vv"] + pk_mult["1loop_4_vv"] + bias[ind,0]*pk_mult["1loop_4_vd"] + bias[ind,1]*pk_mult["1loop_4_dd"] + bias[ind,4]*pk_mult["Id2_4"] + bias[ind,6]*pk_mult["IG2_4"])*pow(h, 3.0*h_units) + 2.0*ctrs[ind,2]*pk_mult["ctr_4"]*pow(h, h_units) + fz**2*np.power(k, 2.0)*35/8.0*pk_mult["ctr_4"]*(210.0*bias[ind,11]*fz**2 + 390.0*fz*bias[ind,12] + 143.0*bias[ind,13])*(8.0/5005.0)*pow(h, h_units)
		for i in range(len(c[ind,:,2])):
			resp += c[ind,i,2]*np.power(k, 2*i)

		return resp

	#Compute the spectra and save in the dictionary
	x = {}
	if(RSD == True):
		if(0 in ls):
			P = np.zeros([int(Ntracers*(Ntracers+1)/2), len(k)])
			ind = 0
			for i in range(Ntracers):
				for j in range(i+1):
					P[ind,:] = Pgg_l0(ind)
					ind += 1
			x["Pgg_l0"] = P	
		if(2 in ls):
			P = np.zeros([int(Ntracers*(Ntracers+1)/2), len(k)])
			ind = 0
			for i in range(Ntracers):
				for j in range(i+1):
					P[ind,:] = Pgg_l2(ind)
					ind += 1
			x["Pgg_l2"] = P	
		if(4 in ls):
			P = np.zeros([int(Ntracers*(Ntracers+1)/2), len(k)])
			ind = 0
			for i in range(Ntracers):
				for j in range(i+1):
					P[ind,:] = Pgg_l4(ind)
					ind += 1
			x["Pgg_l4"] = P	
	else:
		P = np.zeros([int(Ntracers*(Ntracers+1)/2), len(k)])
		ind = 0
		for i in range(Ntracers):
			for j in range(i+1):
				P[ind,:] = Pgg(ind)
				ind += 1
		x["Pgg"] = P

	#Output the spectra
	if(OUT_MULT == True):
		for key in pk_mult.keys():
			if(key == "ctr" or key == "ctr_0" or key == "ctr_2" or key == "ctr_4"):
				x[key] = pk_mult[key]*pow(h, h_units)
			else:
				x[key] = pk_mult[key]*pow(h, 3.0*h_units)

	return x