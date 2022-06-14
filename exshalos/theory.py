import numpy as np
from scipy.integrate import simps, odeint
from scipy.special import binom

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
def f(s, model = 0, theta = None, delta_c = -1, Om0 = 0.31, z = 0.0):
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

	return -f(sigma, model, theta, delta_c, Om0, z)*rhom/M*dlnsdlnm(M, sigma)

#Halo bias
def bh(M, s = None, model = 0, theta = None, delta_c = -1, Om0 = 0.31, z = 0.0, k = None, P = None):
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

	return resp