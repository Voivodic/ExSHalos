import numpy as np
from scipy.integrate import simps
from scipy.special import binom
import pechuga

#Window function in Fourier space
def W(k, R):
	resp = 3.0/(np.power(k*R,2))*(np.sin(k*R)/(k*R) - np.cos(k*R))

	return resp

#Compute the variance of the linear density field
def Compute_sigma(k, P, R):

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
def f(cosmo, s, model = 0, theta = None):
	nu = cosmo.dc/s
	resp = np.zeros(len(s))

	#Press-Schechter
	if(model == 0):	
		resp = np.sqrt(2.0/np.pi)*nu*np.exp(-nu*nu/2)

	#Sheth-Tormen
	elif(model == 1):
		if(theta != None):
			a, b, p = theta
		else:
			a, b, p = np.array([0.7, 0.4, 0.6])

		B = np.sqrt(a)*cosmo.dc*(1.0 + b*np.power(a*nu*nu, -p))
		A = 0.0
		for i in range(6):
			A += np.power(-1, i)*binom(p, i)

		resp = np.sqrt(2.0*a/np.pi)*nu*np.exp(-B*B/(2.0*s*s))*(1.0 + b*A*np.power(a*nu*nu, -p))
	
	#Tinker
	elif(model == 2):
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
	elif(model == 3):
		if(theta != None):
			b, D, dv, J_max = theta
		else:
			b, D, dv, J_max = np.array([0.0, 0.0, 2.71, 20])

		resp = np.zeros(len(s))
		dt = cosmo.dc + dv

		for n in range(1,J_max+1):
			resp += 2.0*(1.0+D)*np.exp(-b*b*s*s/(2.0*(1.0+D)))*np.exp(-b*cosmo.dc/(1.0+D))*(n*np.pi/(dt*dt))*s*s*np.sin(n*np.pi*cosmo.dc/dt)*np.exp(-n*n*np.pi*np.pi*s*s*(1.0+D)/(2.0*dt*dt))

	return resp

#Halo mass function
def dlnndlnm(M, sigma, model=0, theta=None, cosmo=ExSHalos.Run_ExSHalos.Cosmology()):
	return -f(cosmo, sigma, model, theta)*cosmo.rhomz/M*dlnsdlnm(M, sigma)

#Halo bias
def bh(s, model=0, theta=None, cosmo=ExSHalos.Run_ExSHalos.Cosmology()):
	nu = cosmo.dc/s
	resp = np.zeros(len(s))

	#Press-Schechter
	if(model == 0):	
		resp = 1.0 + (nu*nu - 1.0)/cosmo.dc

	#Sheth-Tormen
	elif(model == 1):
		if(theta != None):
			a, b, p = theta
		else:
			a, b, p = np.array([0.7, 0.4, 0.6])

		A = 0.0
		for i in range(6):
			A += np.power(-1, i)*binom(p, i)

		resp = 1.0 + np.sqrt(a)*nu*nu/cosmo.dc*(1.0 + b*np.power(a*nu*nu, -p)) - 1.0/(np.sqrt(a)*cosmo.dc*(1.0 + A*np.power(a*nu*nu, -p)))

	#Tinker
	elif(model == 2):
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

		resp = 1.0 - A*np.power(nu,a)/(np.power(nu,a) + pow(cosmo.dc,a)) + B*np.power(nu,b) + C*np.power(nu,c)

	#Linear difusive barrier
	elif(model == 3):
		if(theta != None):
			b, D, dv, J_max = theta
		else:
			b, D, dv, J_max = np.array([0.0, 0.0, 2.71, 20])

		resp = np.zeros(len(s))
		tmp = np.zeros(len(s))
		dt = cosmo.dc + dv

		#Halos
		for n in range(1,J_max+1):
			resp -= (n*np.pi/(dt*dt))*np.sin(n*np.pi*cosmo.dc/dt)*np.exp(-n*n*np.pi*np.pi*s*s*(1.0+D)/(2.0*dt*dt))*(np.power(np.tan(n*np.pi*cosmo.dc/dt), -1.0)*(n*np.pi/dt) - b/(1.0 + D))

		for n in range(1,J_max+1):
			tmp += (n*np.pi/(dt*dt))*np.sin(n*np.pi*cosmo.dc/dt)*np.exp(-n*n*np.pi*np.pi*s*s*(1.0+D)/(2.0*dt*dt))

		resp = np.ones(len(s)) + resp/tmp

	return resp