import numpy as np 
import h5py
import exshalos
import pylab as pl
import time

#Generate the fake data
L = 4000.0
Nd = 256
ntypes = 1
interlacing = True
window = "CIC"
nthreads = 4
Nk = 5
verbose = False

#Read the halo catalogue
f = h5py.File('/home/voivodic/Documents/Multi_EFT/MD4_M13.2_xyz_Vxyz_TF.hdf5','r')  
logM = np.array(f['catalog'][:,0])
pos = np.array(f['catalog'][:,1:4])
f.close()

print(logM.shape, pos.shape)

types = np.zeros(len(logM))
if(ntypes == 2):
    types[logM > 13.5] = 1
else:
    types = None

#Compute the density grid
print("Computing the density grid")
start = time.time()
g1 = exshalos.simulation.Compute_Density_Grid(pos, nd = Nd, type = types, L = L, window = window, interlacing = interlacing, nthreads = nthreads, verbose = verbose)
end = time.time()

print("Time took = %f" %(end - start))
print(g1.shape)
print(np.mean(g1[0]), np.std(g1[0]))

#Compute the power spectra
print("Computing the power spectrum")
start = time.time()
k, P, Nmodes = exshalos.simulation.Compute_Power_Spectrum(g1, L = L, window = window, Nk = Nk, nthreads = nthreads, verbose = verbose, ntype = ntypes)
end = time.time()

print("Time took = %f" %(end - start))
print(k.shape, P.shape, Nmodes.shape)

print("Computing the bispectrum")
start = time.time()
kP2, P2, Nmodes2, kB, B, Ntri = exshalos.simulation.Compute_BiSpectrum(g1, L = L, window = window, Nk = Nk, nthreads = nthreads, verbose = verbose, ntype = ntypes)
end = time.time()

print("Time took = %f" %(end - start))
print(kB.shape, B.shape, Ntri.shape, kP2.shape, P2.shape, Nmodes2.shape)

print("Computing the trispectrum")
start = time.time()
kP3, P3, Nmodes3, kT, T, Tu, Nsq = exshalos.simulation.Compute_TriSpectrum(g1, L = L, window = window, Nk = Nk, nthreads = nthreads, verbose = verbose, ntype = ntypes)
end = time.time()

print("Time took = %f" %(end - start))
print(kT.shape, T.shape, Tu.shape, Nsq.shape, kP3.shape, P3.shape, Nmodes3.shape)

pl.clf()
if(ntypes == 2):
    pl.loglog(k, P[0,:], "-", color = "red")
    pl.loglog(k, P[1,:], "-", color = "red")
    pl.loglog(k, P[2,:], "-", color = "red")

    pl.loglog(kP2, P2[0,:], "--", color = "blue")
    pl.loglog(kP2, P2[1,:], "--", color = "blue")
    pl.loglog(kP2, P2[2,:], "--", color = "blue")

    pl.loglog(kP3, P3[0,:], ":", color = "darkgreen")
    pl.loglog(kP3, P3[1,:], ":", color = "darkgreen")
    pl.loglog(kP3, P3[2,:], ":", color = "darkgreen")

else:
    pl.loglog(k, P, "-", color = "red")
    pl.loglog(kP2, P2, "--", color = "blue")
    pl.loglog(kP3, P3, ":", color = "darkgreen")

pl.savefig("Test_power.pdf")

print("Done!")