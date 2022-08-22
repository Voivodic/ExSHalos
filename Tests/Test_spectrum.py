import numpy as np 
import h5py
import exshalos
import pylab as pl
import time
import bacco

#Generate the fake data
L = 4000.0
Nd = 128
ntypes = 1
interlacing = True
window = "cic"
nthreads = 4
Nk = 10
verbose = False
kmin = 2.0*np.pi/L
kmax = np.pi/L*512

#Read the halo catalogue
f = h5py.File('/home/voivodic/Documents/Multi_EFT/MD4_M13.2_xyz_Vxyz_TF.hdf5','r')  
logM = np.array(f['catalog'][:,0])
pos = np.array(f['catalog'][:,1:4])
vel = np.array(f['catalog'][:,4:7])
f.close()

mask = logM > 13.5
logM = logM[mask]
pos = pos[mask,:]
vel = vel[mask,:]

nbar = len(logM)/L**3

print(logM.shape, pos.shape, vel.shape)

types = np.zeros(len(logM))
if(ntypes == 2):
    types[logM > 13.5] = 1
else:
    types = None
'''
#Compute the density grid
print("Computing the density grid")
start = time.time()
g1 = exshalos.simulation.Compute_Density_Grid(pos, vel = vel, nd = Nd, types = types, direction = "z", L = L, window = window, interlacing = interlacing, nthreads = nthreads, verbose = verbose)
end = time.time()

print("Time took = %f" %(end - start))
print(g1.shape)
print(np.mean(g1[0]), np.std(g1[0]))

#Compute the power spectra
print("Computing the power spectrum")
start = time.time()
P = exshalos.simulation.Compute_Power_Spectrum(g1, L = L, window = window, Nk = Nk, nthreads = 2, verbose = verbose, ntypes = ntypes, l_max = 0)
end = time.time()

print("Time took = %f" %(end - start))
print(P['k'].shape, P["Pk"].shape, P["Nk"].shape)
'''

#Computing the power spectrum for different nunbers of foldings

pl.clf()
nd = 256
folds = 1
for folds in [1,2,3,4]:
    print("Doing for Nd = %d" %(folds))

    #Compute the density grid
    g1 = exshalos.simulation.Compute_Density_Grid(pos, nd = nd, types = types, L = L, window = window, interlacing = interlacing, folds = folds, nthreads = nthreads, verbose = verbose)

    #Compute the power spectra
    kN = 2.0/3.0*np.pi/L*nd*folds
    B = exshalos.simulation.Compute_BiSpectrum(g1, L = L, window = window, Nk = Nk, k_min = kmin, k_max = kN, folds = folds, nthreads = nthreads, verbose = verbose)

    mask = (B["kB"][0,:] == B["kB"][1,:])*(B["kB"][1,:] == B["kB"][2,:])
    pl.plot(B["kB"][0,mask], B["Bk"][mask]*pow(folds, 6.0), linestyle = "-", linewidth = 3, marker = "", markersize = 5, label = "folds = %d" %(folds))


print("Doing for full")

#Compute the density grid
'''g1 = exshalos.simulation.Compute_Density_Grid(pos, nd = 512, types = types, L = L, window = window, interlacing = interlacing, folds = 1, nthreads = nthreads, verbose = verbose)

#Compute the power spectra
kN = np.pi/L*512
B = exshalos.simulation.Compute_BiSpectrum(g1, L = L, window = window, Nk = Nk, k_min = kmin, k_max = kmax, folds = 1, nthreads = nthreads, verbose = verbose)

mask = (B["kB"][0,:] == B["kB"][1,:])*(B["kB"][1,:] == B["kB"][2,:])
pl.plot(B["kB"][0,mask], B["Bk"][mask], linestyle = "-", linewidth = 3, marker = "", markersize = 5, label = "Nd = %d" %(512))
'''

pl.xscale("linear")
pl.yscale("log")
pl.legend(loc="best")

pl.savefig("Test_B_folds.pdf")

'''
print("Computing the bispectrum")
start = time.time()
B = exshalos.simulation.Compute_BiSpectrum(g1, L = L, window = window, Nk = Nk, nthreads = nthreads, verbose = verbose, ntypes = ntypes)
end = time.time()

print("Time took = %f" %(end - start))
print(B["kB"].shape, B["Bk"].shape, B["Ntri"].shape, B["kP"].shape, B["Pk"].shape, B["Nk"].shape)

print("Computing the trispectrum")
start = time.time()
T= exshalos.simulation.Compute_TriSpectrum(g1, L = L, window = window, Nk = Nk, nthreads = nthreads, verbose = verbose, ntypes = ntypes)
end = time.time()

print("Time took = %f" %(end - start))
print(T["kT"].shape, T["Tk"].shape, T["Tuk"].shape, T["Nsq"].shape, T["kP"].shape, T["Pk"].shape, T["Nk"].shape)

pl.clf()
if(ntypes == 2):
    #pl.plot(P["k"], P["Pk"][0,:], "-", color = "red")
    #pl.plot(P["k"], P["Pk"][1,:], "-", color = "red")
    #pl.plot(P["k"], P["Pk"][2,:], "-", color = "red")

    pl.plot(B["kP"], B["Pk"][0,:]/P["Pk"][0,:], "--", color = "blue")
    pl.plot(B["kP"], B["Pk"][1,:]/P["Pk"][1,:], "--", color = "blue")
    pl.plot(B["kP"], B["Pk"][2,:]/P["Pk"][2,:], "--", color = "blue")

    pl.plot(T["kP"], T["Pk"][0,:]/P["Pk"][0,:], ":", color = "darkgreen")
    pl.plot(T["kP"], T["Pk"][1,:]/P["Pk"][1,:], ":", color = "darkgreen")
    pl.plot(T["kP"], T["Pk"][2,:]/P["Pk"][2,:], ":", color = "darkgreen")

else:
    print(B["Pk"]/P["Pk"])
    print(T["Pk"]/P["Pk"])

    #pl.plot(P["k"], P["Pk"], "-", color = "red")
    pl.plot(B["kP"], B["Pk"]/P["Pk"], "--", color = "blue")
    pl.plot(T["kP"], T["Pk"]/P["Pk"], ":", color = "darkgreen")

pl.xscale("linear")
pl.yscale("linear")

pl.savefig("Test_power.pdf")

print("Done!")
'''