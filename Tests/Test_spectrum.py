import numpy as np 
import h5py
import exshalos
import pylab as pl
import time
import bacco

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
vel = np.array(f['catalog'][:,4:7])
f.close()

print(logM.shape, pos.shape, vel.shape)

types = np.zeros(len(logM))
if(ntypes == 2):
    types[logM > 13.5] = 1
else:
    types = None

#Compute the density grid
print("Computing the density grid")
start = time.time()
g1 = exshalos.simulation.Compute_Density_Grid(pos, vel = vel, nd = Nd, type = types, direction = "z", L = L, window = window, interlacing = interlacing, nthreads = nthreads, verbose = verbose)
end = time.time()

print("Time took = %f" %(end - start))
print(g1.shape)
print(np.mean(g1[0]), np.std(g1[0]))

#Compute the power spectra
print("Computing the power spectrum")
start = time.time()
P = exshalos.simulation.Compute_Power_Spectrum(g1, L = L, window = window, Nk = Nk, nthreads = nthreads, verbose = verbose, ntype = ntypes)
end = time.time()

print("Time took = %f" %(end - start))
print(P['k'].shape, P["Pk"].shape, P["Nk"].shape)

print("Computing the bispectrum")
start = time.time()
B = exshalos.simulation.Compute_BiSpectrum(g1, L = L, window = window, Nk = Nk, nthreads = nthreads, verbose = verbose, ntype = ntypes)
end = time.time()

print("Time took = %f" %(end - start))
print(B["kB"].shape, B["Bk"].shape, B["Ntri"].shape, B["kP"].shape, B["Pk"].shape, B["Nk"].shape)

print("Computing the trispectrum")
start = time.time()
T= exshalos.simulation.Compute_TriSpectrum(g1, L = L, window = window, Nk = Nk, nthreads = nthreads, verbose = verbose, ntype = ntypes)
end = time.time()

print("Time took = %f" %(end - start))
print(T["kT"].shape, T["Tk"].shape, T["Tuk"].shape, T["Nsq"].shape, T["kP"].shape, T["Pk"].shape, T["Nk"].shape)

pl.clf()
if(ntypes == 2):
    pl.loglog(P["k"], P["Pk"][0,:], "-", color = "red")
    pl.loglog(P["k"], P["Pk"][1,:], "-", color = "red")
    pl.loglog(P["k"], P["Pk"][2,:], "-", color = "red")

    pl.loglog(B["kP"], B["Pk"][0,:], "--", color = "blue")
    pl.loglog(B["kP"], B["Pk"][1,:], "--", color = "blue")
    pl.loglog(B["kP"], B["Pk"][2,:], "--", color = "blue")

    pl.loglog(T["kP"], T["Pk"][0,:], ":", color = "darkgreen")
    pl.loglog(T["kP"], T["Pk"][1,:], ":", color = "darkgreen")
    pl.loglog(T["kP"], T["Pk"][2,:], ":", color = "darkgreen")

else:
    pl.loglog(P["k"], P["Pk"], "-", color = "red")
    pl.loglog(B["kP"], B["Pk"], "--", color = "blue")
    pl.loglog(T["kP"], T["Pk"], ":", color = "darkgreen")

pl.savefig("Test_power.pdf")

print("Done!")