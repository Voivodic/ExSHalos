import numpy as np 
import exshalos
import pylab as pl
import time
import h5py

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
posh, Mh = exshalos.utils.Find_Halos_from_Grid(grid, k, P, Lc = Lc, Nmin = 10)
print(posh.shape, Mh.shape)

print(np.min(posh), np.max(posh))
print("%e %e" %(np.min(Mh), np.max(Mh)))

#Compute the mass function
t1 = time.time()
M, dn, dn_err = exshalos.simulation.Compute_Abundance(Mh, Mmin = 1e+13, Mmax = 1e+15, Nm = 20, Lc = Lc, nd = Nd)
t1 = time.time() - t1

t2 = time.time()
Mbin = np.logspace(13, 15, 21)
n = np.histogram(Mh, bins = Mbin)[0]

dn2 = n/(np.log(Mbin[1:]) - np.log(Mbin[:-1]))/(Lc**3*Nd**3)
dn2_err = np.sqrt(n)/(np.log(Mbin[1:]) - np.log(Mbin[:-1]))/(Lc**3*Nd**3)
t2 = time.time() - t2

print(t1, t2)

#COmpute the spectrum
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
pos, vel = exshalos.utils.Displace_LPT(grid, Lc = Lc, k_smooth = 0.4, DO_2LPT = False, OUT_VEL = True, OUT_POS = False)

#gridp = exshalos.simulation.Compute_Density_Grid(pos, nd = Nd, L = L, window = Window, interlacing = True, nthreads = 1)
#kp, Pp, Nkp = exshalos.simulation.Compute_Power_Spectrum(gridp, L = L, window = Window, Nk = Nk, nthreads = 1, l_max = 0)

'''

pl.clf()
#pl.plot(ks, Ps, "o", label = "Gaussian")
k_smooth = [1.0]
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
'''
#Generate a halo catalogue from a given power spectrum
print("Generating the halo catalogue")
'''
pl.clf()
k_smooth = [0.2, 0.4, 0.5, 0.6, 2.0]
cores = ["blue", "red", "darkgreen", "purple", "black"]
for ks, cor in zip(k_smooth, cores):
    print(ks)

    posh2, Mh2 = exshalos.Generate_Halos_Box_from_Pk(k, P, nd = Nd, Lc = Lc, Om0 = 0.31, z = 0.0, k_smooth = ks, Nmin = 1, a = 1.0, beta = 0.0, alpha = 0.0, OUT_DEN = False, OUT_LPT = False, OUT_VEL = False, DO_2LPT = False, OUT_FLAG = False)

    gridh2 = exshalos.simulation.Compute_Density_Grid(posh2, nd = Nd, L = L, window = Window, interlacing = True, nthreads = 1, verbose = False)
    kh2, Ph2, Nkh2 = exshalos.simulation.Compute_Power_Spectrum(gridh2, L = L, window = Window, Nk = Nk, nthreads = 1, l_max = 0)

    pl.plot(kh2, Ph2 , "o", color = cor, label = r"$k_{\rm smooth} = %.2f$" %(ks))

    posh2, Mh2 = exshalos.Generate_Halos_Box_from_Pk(k, P, nd = Nd, Lc = Lc, Om0 = 0.31, z = 0.0, k_smooth = ks, Nmin = 1, a = 1.0, beta = 0.0, alpha = 0.0, OUT_DEN = False, OUT_LPT = False, OUT_VEL = False, DO_2LPT = True, OUT_FLAG = False)

    gridh2 = exshalos.simulation.Compute_Density_Grid(posh2, nd = Nd, L = L, window = Window, interlacing = True, nthreads = 1, verbose = False)
    kh2, Ph2, Nkh2 = exshalos.simulation.Compute_Power_Spectrum(gridh2, L = L, window = Window, Nk = Nk, nthreads = 1, l_max = 0)

    pl.plot(kh2, Ph2 , "^", color = cor)


pl.axhline(y = pow(L, 3.0)/len(Mh2), color = 'black', linestyle = '-')
pl.grid(True)
pl.yscale("log")
pl.xscale("log")
#pl.ylim(-0.5, 0.25)
#pl.xlim(0.0, 0.6)
pl.legend(loc = "best")
pl.savefig("Power_halos1.pdf")
'''

start = time.time()
posh2, Mh2 = exshalos.mock.Generate_Halos_Box_from_Pk(k, P, nd = Nd, Lc = Lc, Om0 = 0.31, z = 0.0, k_smooth = 0.4, Nmin = 1, a = 1.0, beta = 0.0, alpha = 0.0, OUT_DEN = False, OUT_LPT = False, OUT_VEL = False, DO_2LPT = False, OUT_FLAG = False)
end = time.time()
run1 = end - start

start = time.time()
posh3, Mh3= exshalos.mock.Generate_Halos_Box_from_Grid(grid = grid, k = k, P = P, Lc = Lc, Om0 = 0.31, z = 0.0, k_smooth = 0.4, Nmin = 1, a = 1.0, beta = 0.0, alpha = 0.0, OUT_LPT = False, OUT_VEL = False, DO_2LPT = False, OUT_FLAG = False)
end = time.time()
run2 = end - start

start = time.time()
posh4, Mh4= exshalos.mock.Generate_Halos_Box_from_Grid(grid = grid, k = k, P = P, S = pos, Lc = Lc, Om0 = 0.31, z = 0.0, k_smooth = 0.4, Nmin = 1, a = 1.0, beta = 0.0, alpha = 0.0, OUT_LPT = False, OUT_VEL = False, DO_2LPT = False, OUT_FLAG = False)
end = time.time()
run3 = end - start

print(posh2.shape, posh3.shape, posh4.shape)
print(run1, run2, run3)

print("Generating the galaxy catalogue")
print(posh2.shape, Mh2.shape)
star = time.time()
posg = exshalos.mock.Generate_Galaxies_from_Halos(posh2, Mh2, nd = Nd, Lc = Lc, Om0 = 0.31, z = 0.0, OUT_VEL = False)
end = time.time()
rung = end - star

gridh = exshalos.simulation.Compute_Density_Grid(posh2, nd = Nd, L = L, window = Window, interlacing = True, nthreads = 1, verbose = False)
kh, Ph, Nkh = exshalos.simulation.Compute_Power_Spectrum(gridh, L = L, window = Window, Nk = Nk, nthreads = 1, l_max = 0)

gridg = exshalos.simulation.Compute_Density_Grid(posg, nd = Nd, L = L, window = Window, interlacing = True, nthreads = 1, verbose = False)
kg, Pg, Nkg = exshalos.simulation.Compute_Power_Spectrum(gridg, L = L, window = Window, Nk = Nk, nthreads = 1, l_max = 0)

pl.clf()
pl.plot(kg, Pg , "o", color = "blue")
pl.plot(kh, Ph , "o", color = "red")
pl.plot(ks, Ps, "o", color = "black")

pl.grid(True)
pl.xscale("log")
pl.yscale("log")
pl.savefig("Power_gal.pdf")

print(posg.shape)
print(rung)


print("Populating multidark")
f = h5py.File("/home/voivodic/Documents/Multi_EFT/MD4_M13.2_xyz_Vxyz_TF.hdf5", 'r')
x = f['catalog'][:]
nh = len(x)
f.close()

x1 = np.loadtxt("/home/voivodic/Documents/Multi_EFT/HODs/Power_gals.dat")
pl.clf()
pl.plot(x1[:,0], x1[:,-2], "-", color = "black")

for i in range(1):
    print(i)

    posg = exshalos.mock.Generate_Galaxies_from_Halos(x[:,1:4], pow(10.0, x[:,0]), nd = 4000, Lc = 1.0, Om0 = 0.31, z = 0.0, OUT_VEL = False, seed = i*642)
    end = time.time()

    gridg = exshalos.simulation.Compute_Density_Grid(posg, nd = 512, L = 4000.0, window = Window, interlacing = True, nthreads = 1, verbose = False)
    kg, Pg, Nkg = exshalos.simulation.Compute_Power_Spectrum(gridg, L = 4000.0, window = Window, Nk = 128, nthreads = 1, l_max = 0)

    pl.plot(x1[:,0], Pg , "--")

pl.grid(True)
pl.xscale("log")
pl.yscale("log")
pl.savefig("Power_gal_MD.pdf")

print(posg.shape)
print(rung)