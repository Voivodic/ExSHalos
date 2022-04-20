import numpy as np 
import exshalos
import pylab as pl
import time

#Parameters
Lc = 2.0
Nd = 256
L = Lc*Nd
Nk = 64
Window = "CIC"

#Open the linear power spectrum
k, P = np.loadtxt("MDPL2_z00_matterpower.dat", unpack = True)

#Compute the correlation function
print("Computing the correlation")
R, Xi = exshalos.simulation.Compute_Correlation(k, P, verbose = True)
k2, P2 = exshalos.simulation.Compute_Correlation(R, Xi, direction = -1, verbose = True)

#Plot the correlation
pl.clf()
pl.loglog(k, P, "-")
pl.loglog(k2, P2, "--")
pl.savefig("Correlation.pdf")

#Compute the density grid
print("Generating the density grid")
grid = exshalos.utils.Generate_Density_Grid(k, P, Lc = Lc, nd = Nd)
print(grid.shape)

ks, Ps, Nks = exshalos.simulation.Compute_Power_Spectrum(grid, L = L, window = 0, Nk = Nk, nthreads = 1, l_max = 0)

print(Ps.shape)

pl.clf()
pl.loglog(k, P, "-")
pl.loglog(ks, Ps, "o")
pl.ylim(1e+2, 5e+4)
pl.xlim(5e-3, 1e+0)
pl.savefig("Power_sim.pdf")

#Find the halos in the grid above
print("Finding the halos")
posh, Mh = exshalos.utils.Find_Halos_from_Grid(grid, k, P, Lc = Lc, Nmin = 1)
print(posh.shape, Mh.shape)

print(np.min(posh), np.max(posh))
print("%e %e" %(np.min(Mh), np.max(Mh)))

gridh = exshalos.simulation.Compute_Density_Grid(posh, nd = Nd, L = L, window = Window, interlacing = True, nthreads = 1, verbose = False)
kh, Ph, Nkh = exshalos.simulation.Compute_Power_Spectrum(gridh, L = L, window = Window, Nk = Nk, nthreads = 1, l_max = 0)

pl.clf()
pl.loglog(kh, Ph , "s")
pl.loglog(ks, Ps, "o")
#pl.ylim(1e+2, 5e+4)
#pl.xlim(5e-3, 1e+0)
pl.savefig("Power_halos.pdf")

pl.clf()
pl.hist(Mh, bins = np.logspace(11, 15, 41), log=True)
pl.xscale("log")
pl.savefig("Hist_Mh.pdf")

#Compute the LPT using the same grid
print("Doing LPT")

pl.clf()
#pl.plot(ks, Ps, "o", label = "Gaussian")
k_smooth = [0.1, 0.15, 0.2, 0.3, 1.0]
cores = ["blue", "red", "darkgreen", "purple", "black"]
for ks, cor in zip(k_smooth, cores):
    pos = exshalos.utils.Displace_LPT(grid, Lc = Lc, k_smooth = ks, DO_2LPT = False)

    gridp = exshalos.simulation.Compute_Density_Grid(pos, nd = Nd, L = L, window = Window, interlacing = True, nthreads = 1)
    kp, Pp, Nkp = exshalos.simulation.Compute_Power_Spectrum(gridp, L = L, window = Window, Nk = Nk, nthreads = 1, l_max = 0)

    pl.plot(kp, Pp/Ps - 1.0, "s", color = cor, label = r"$k_{\rm smooth} = %.2f$" %(ks))

    pos = exshalos.utils.Displace_LPT(grid, Lc = Lc, k_smooth = ks, DO_2LPT = True)

    gridp = exshalos.simulation.Compute_Density_Grid(pos, nd = Nd, L = L, window = Window, interlacing = True, nthreads = 1)
    kp, Pp, Nkp = exshalos.simulation.Compute_Power_Spectrum(gridp, L = L, window = Window, Nk = Nk, nthreads = 1, l_max = 0)

    pl.plot(kp, Pp/Ps - 1.0, "^", color = cor)   

pl.grid(True)
pl.yscale("linear")
pl.ylim(-0.5, 0.25)
pl.xlim(0.0, 0.6)
pl.legend(loc = "best")
pl.savefig("Power_LPT.pdf")

print(np.mean(grid), np.std(grid))
print(np.mean(gridp[0]), np.std(gridp[0]))
print(np.mean(gridh[0]), np.std(gridh[0]))

pl.clf()
pl.subplot(131)
pl.imshow(grid[128,:,:])
pl.subplot(132)
pl.imshow(gridp[0,128,:,:])
pl.subplot(133)
pl.imshow(gridh[0,128,:,:])
pl.savefig("Grids.pdf")