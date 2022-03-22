import numpy as np 
import exshalos
import pylab as pl
import time

#Open the linear power spectrum
k, P = np.loadtxt("MDPL2_z00_matterpower.dat", unpack = True)

#Compute the correlation function
R, Xi = exshalos.simulation.Compute_Correlation(k, P, verbose = True)
k2, P2 = exshalos.simulation.Compute_Correlation(R, Xi, direction = -1, verbose = True)

#Plot the correlation
pl.loglog(k, P, "-")
pl.loglog(k2, P2, "--")
pl.savefig("Correlation.pdf")

#Compute the density grid
grid = exshalos.utils.Generate_Density_Grid(k, P, Lc = 8.0, nd = 256)
print(grid.shape)

ks, Ps, Nks = exshalos.simulation.Compute_Power_Spectrum(grid, L = 2048.0, window = 0, Nk = 64, nthreads = 1, l_max = 0)

print(Ps.shape)

pl.loglog(k, P, "-")
pl.loglog(ks, Ps[0,0,:], "o")
pl.ylim(1e+2, 5e+4)
pl.xlim(5e-3, 1e+0)
pl.savefig("Power_sim.pdf")