import numpy as np 
import exshalos
from scipy.interpolate import interp1d
import pylab as pl
import time 

#Parameters of the box
Nd = 256    #Number of cells in each direction
Lc = 2.0    #Size of each cell in Mpc/h
L = Lc*Nd   #Total size of the box in Mpc/h

#Cosmological parameters
Om0 = 0.31  #Value of the matter overdensity today
z = 0.0     #Redshift for the outputs

#Parameters of the ellipsoidal barrier (B = a*delta_c*(1 + beta*S^alpha))
a = 1.0     #Parameter a of the ellipsoidal barrier (a = 1 for spherical collapse)
beta = 0.0  #Parameter beta of the ellipsoidal barrier (beta = 0 for spherical collapse)
alpha = 0.0 #Parameter alpha of the ellipsoidal barrier (alpha = 1 for spherical collapse)

#Some options of ExSHalos
R_max = 10000.0     #Typical size of your box/survey in Mpc/h. Used to add super sample covariance
k_smooth = 0.4      #Scale used to smooth the LPT displacements in h/Mpc
Nmin = 1            #Minimum number of cells/particles per halo to be saved
seed = 12345        #Seed used for the random number generators
nthreads = 1        #Number of threads to be used
OUT_DEN = False     #Set to True to output the linear density field
OUT_LPT = True     #Set to True to output the positions (and velocities if OUT_VEL = True) of the particles
OUT_VEL = False     #Set to True to output the velocities of the halos
DO_2LPT = False     #Set to True to use the second order LPT to compute the displacements and velocities (more precise but more time and memory)
OUT_FLAG = True    #Set to True to output the flag with the host halo of each particle
OUT_PROF = False    #Set to True to output the density profile of the Lagrangian halos

#Open the linear power spectrum (it must be at the redshift 0 and in units WITH h)
k, P = np.loadtxt("MDPL2_z00_matterpower.dat", unpack = True)

#Compute the growth function and normalize the linear power spectrum
Dz = exshalos.theory.Get_Dz(Om0 = 0.31, zmax = 1000, zmin = 0.0, nzs = 1000)
fDz = interp1d(Dz["z"], Dz["Dz"])
P = fDz(z)/fDz(0.0)*P      #Change the second number by the redshift of your linear power spectrum

###Run ExSHalos###
#The full output is: halo positions, halo velocities , halo masses , particle positions , particle velocities , particle flags, density grid and density profiles
start = time.time()
x = exshalos.mock.Generate_Halos_Box_from_Pk(k, P, R_max = R_max, nd = Nd, Lc = Lc, Om0 = Om0, z = z, k_smooth = k_smooth, Nmin = Nmin, a = a, beta = beta, alpha = alpha, seed = seed, OUT_DEN = OUT_DEN, OUT_LPT = OUT_LPT, OUT_VEL = OUT_VEL, DO_2LPT = DO_2LPT, OUT_FLAG = OUT_FLAG, OUT_PROF = OUT_PROF, nthreads = nthreads)
end = time.time()

nh = len(x["Mh"])
print("ExSHalos took %f seconds to generate %d halos!" %(end - start, nh))

#Parameters used to measured the power spectra
Ntracers = 2        #Number of tracers to be used to split the halo catalogue
window = "CIC"      #Window to be used to assign the halos to the grid
kmin = 0.0          #Minimum k used to measure the spectrum in h/Mpc
kmax = None         #Maximum k used to measure the spectrum in h/Mpc
Nk = 64             #Number of k-bins
l_max = 0           #Maximum multipole measured
R = 4.0             #Radius of the sphere used to smooth the particles in Mpc/h (only used if window = EXPONENTIAL or Spherical)
R_times = 5         #Radius (in units of R) where particles are considered to construct the densidy grid (only used if window = EXPONENTIAL)
direction = None    #Direction used to put the halos in redshift space
interlacing = True  #Set to True to used interlacing to measure the spectra
Nd = 256            #Number of cells in each direction (we can use a different number of cells to compute the spectrum)

#Split the halos into multi tracers. It is just a simple example that split them (more or less) using their masses
if(Ntracers > 1):
    types = np.zeros(nh)
    for i in range(Ntracers):
        types[i*int(nh/Ntracers):] = i
else:
    types = None

###Measure the spectra###
start = time.time()
#Compute the density grid
if(OUT_VEL == False):
    grid = exshalos.simulation.Compute_Density_Grid(x["posh"], vel = None, mass = None, types = types, nd = Nd, L = L, direction = None, window = window, R = R, R_times = R_times, interlacing = interlacing, nthreads = nthreads)
else:
    grid = exshalos.simulation.Compute_Density_Grid(x["posh"], vel = x["velh"], mass = None, types = types, nd = Nd, L = L, direction = direction, window = window, R = R, R_times = R_times, interlacing = interlacing, nthreads = nthreads)

#Compute the power spectra
xh = exshalos.simulation.Compute_Power_Spectrum(grid, L = L, window = window, R = R, Nk = Nk, k_min = kmin, k_max = kmax, l_max = l_max, nthreads = 1, ntypes = Ntracers)
end = time.time()

print("ExSHalos took %f seconds to measure the power spectrum!" %(end - start))

start = time.time()
#Compute the density grid
if(OUT_VEL == False):
    gridp = exshalos.simulation.Compute_Density_Grid(x["pos"], vel = None, mass = None, nd = Nd, L = L, direction = None, window = window, R = R, R_times = R_times, interlacing = interlacing, nthreads = nthreads)
else:
    gridp = exshalos.simulation.Compute_Density_Grid(x["pos"], vel = x["vel"], mass = None, nd = Nd, L = L, direction = direction, window = window, R = R, R_times = R_times, interlacing = interlacing, nthreads = nthreads)

print(x["pos"].shape)

#Compute the power spectra
xp = exshalos.simulation.Compute_Power_Spectrum(gridp, L = L, window = window, R = R, Nk = Nk, k_min = kmin, k_max = kmax, l_max = l_max, nthreads = 1, ntypes = 1)
end = time.time()

print("ExSHalos took %f seconds to measure the power spectrum!" %(end - start))

###Plot the power spectra###
pl.clf()

if(Ntracers > 1):
    for i in range(len(xh["Pk"])):
        pl.errorbar(xh["k"], xh["Pk"][i], yerr = xh["Pk"][i]/np.sqrt(xh["Nk"]), linestyle = "", marker = "o", markersize = 5, label = r"$P_{%d}$" %(i))
else:
    pl.errorbar(xh["k"], xh["Pk"], yerr = xh["Pk"]/np.sqrt(xh["Nk"]), linestyle = "", marker = "o", markersize = 8, label = r"$P_{hh}$")
pl.errorbar(xp["k"], xp["Pk"], yerr = xp["Pk"]/np.sqrt(xp["Nk"]), linestyle = "", marker = "o", markersize = 5, label = r"$P_{\rm LPT}$")

pl.xscale("log")
pl.yscale("log")
pl.xlabel("k [$h/$Mpc]", fontsize = 15)
pl.ylabel("P(k)  [Mpc/$h/$]$^{3}$", fontsize = 15)
pl.legend(loc = "best", fontsize = 15)
pl.grid(True)

pl.savefig("Ph_example.pdf")