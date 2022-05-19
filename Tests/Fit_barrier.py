import numpy as np 
import exshalos
import pylab as pl
from scipy.interpolate import interp1d
import time
import h5py

#Parameters
Om0 = 0.31
z = 0.0
Lc = 4.0
Nd = 256
L = Lc*Nd
Nk = 64
Window = "CIC"

#Open the linear power spectrum
k, P = np.loadtxt("MDPL2_z00_matterpower.dat", unpack = True)

#Compute the tinker mass function
Mt = np.logspace(10, 16, 61)
sigma = exshalos.theory.Compute_sigma(k, P, M = Mt, Om0 = Om0, z = z)
dnt = exshalos.theory.dlnndlnm(Mt, sigma = sigma, model = 2, theta = 300, delta_c = -1, Om0 = Om0, z = z)
fdn = interp1d(Mt, dnt)

#Fit the barrier
'''print("Fitting the barrier")
barrier = None#np.array([0.54296115, 0.3974207,  0.69304439])
x = exshalos.utils.Fit_Barrier(k, P, Mt, dnt, grid = None, R_max = 100000.0, Mmin = -1.0, Mmax = -1.0, Nm = 25, nd = Nd, Lc = Lc, Om0 = Om0, z = z, delta_c = -1.0, Nmin = 1, seed = 12345, x0 = barrier, verbose = False, nthreads = 1, tol = 0.1)

print(x.x)

#Generate a halo catologue with this barrier
print("Generating the halo catalogues")
barrier = np.array(x.x)'''
barrier = [0.67376829, 0.35238359, 0.44854763]

print("First catalogue")
x = exshalos.mock.Generate_Halos_Box_from_Pk(k, P, nd = Nd*2, Lc = Lc, Om0 = Om0, z = z, k_smooth = 0.4, Nmin = 1, a = barrier[0], beta = barrier[1], alpha = barrier[2], seed = 12345, OUT_DEN = False, OUT_LPT = False, OUT_VEL = False, DO_2LPT = False, OUT_FLAG = False, verbose = False, nthreads = 1)
dnh1 = exshalos.simulation.Compute_Abundance(x["Mh"], Lc = Lc, nd = Nd*2)

print("Second catalogue")
x = exshalos.mock.Generate_Halos_Box_from_Pk(k, P, nd = Nd*2, Lc = Lc, Om0 = Om0, z = z, k_smooth = 0.4, Nmin = 1, seed = 12345, OUT_DEN = False, OUT_LPT = False, OUT_VEL = False, DO_2LPT = False, OUT_FLAG = False, verbose = False, nthreads = 1)
dnh2 = exshalos.simulation.Compute_Abundance(x["Mh"], Lc = Lc, nd = Nd*2)

#Plot the mass functions
pl.clf()

mask1 = dnh1["dn"] > 0.0
mask2 = dnh2["dn"] > 0.0

#pl.plot(Mt, dnt, linestyle = "-", linewidth = 3, marker = "")
pl.errorbar(dnh1["Mh"][mask1], dnh1["dn"][mask1]/fdn(dnh1["Mh"][mask1]) - 1.0, yerr = dnh1["dn_err"][mask1]/fdn(dnh1["Mh"][mask1]), linestyle = "", linewidth = 3, marker = "o", markersize = 5)
pl.errorbar(dnh2["Mh"][mask2], dnh2["dn"][mask2]/fdn(dnh2["Mh"][mask2]) - 1.0, yerr = dnh2["dn_err"][mask2]/fdn(dnh2["Mh"][mask2]), linestyle = "", linewidth = 3, marker = "s", markersize = 5)

pl.xscale("log")
pl.yscale("linear")
pl.ylim(-0.5, 0.5)
pl.grid(True)

pl.savefig("Abundance.pdf")