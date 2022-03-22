import numpy as np 
import h5py
import exshalos
import pylab as pl
import time

#Generate the fake data
L = 4000.0
Nd = 256

#Read the halo catalogue
f = h5py.File('/home/voivodic/Documents/Multi_EFT/MD4_M13.2_xyz_Vxyz_TF.hdf5','r')  
logM = np.array(f['catalog'][:,0])
pos = np.array(f['catalog'][:,1:4])
f.close()

print(logM.shape, pos.shape)

types = np.zeros(len(logM))
types[logM > 13.5] = 1

#Compute the density grid
print("Computing the density grid")
start = time.time()
g1 = exshalos.simulation.Compute_Density_Grid(pos, nd = Nd, type = types, L = L, window = "NGP", interlacing = True, nthreads = 1, verbose = False)
end = time.time()

print("Time took = %f" %(end - start))
print(g1.shape)

#Compute the power spectra
print("Computing the power spectrum")
start = time.time()
k, P, Nk = exshalos.simulation.Compute_Power_Spectrum(g1, ntype = 2, nd = Nd, L = L, window = "NGP", interlacing = True, Nk = 5, nthreads = 1, verbose = False)
end = time.time()

print("Time took = %f" %(end - start))
print(k.shape, P.shape, Nk.shape)

print("Computing the bispectrum")
start = time.time()
kP2, P2, Nk2, kB, B, Ntri = exshalos.simulation.Compute_BiSpectrum(g1, ntype = 2, nd = Nd, L = L, window = "NGP", interlacing = True, Nk = 5, nthreads = 4, verbose = False)
end = time.time()

print("Time took = %f" %(end - start))
print(kB.shape, B.shape, Ntri.shape, kP2.shape, P2.shape, Nk2.shape)

print("Computing the trispectrum")
start = time.time()
kP3, P3, Nk3, kT, T, Tu, Nsq = exshalos.simulation.Compute_TriSpectrum(g1, ntype = 2, nd = Nd, L = L, window = "NGP", interlacing = True, Nk = 5, nthreads = 4, verbose = False)
end = time.time()

print("Time took = %f" %(end - start))
print(kT.shape, T.shape, Tu.shape, Nsq.shape, kP3.shape, P3.shape, Nk3.shape)
print(np.log10(T), np.log10(Tu), np.log10(T - Tu))

#pl.clf()
pl.loglog(k, P[0,:], "-", color = "red")
pl.loglog(k, P[1,:], "-", color = "red")
pl.loglog(k, P[2,:], "-", color = "red")

pl.loglog(kP2, P2[0,:], "--", color = "blue")
pl.loglog(kP2, P2[1,:], "--", color = "blue")
pl.loglog(kP2, P2[2,:], "--", color = "blue")

pl.loglog(kP3, P3[0,:], ":", color = "darkgreen")
pl.loglog(kP3, P3[1,:], ":", color = "darkgreen")
pl.loglog(kP3, P3[2,:], ":", color = "darkgreen")

pl.savefig("Test_power.pdf")

print("Done!")