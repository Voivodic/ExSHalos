# Import the libraries used in this tutotial
import numpy as np
import pylab as pl

import pyexshalos as exh

# Set parameters for the halo catalogue
Om0 = 0.307115
z = 0.0
nd = 256
Lc = 4.0
L = Lc * nd
Nmin = 1
seed = 12345
x0 = np.array([0.62, 0.45, 0.54])
Max_iter = 100
verbose = True

# Load the linear matter power spectrum from the MDPL2 simulation
k, P = np.loadtxt("MDPL2_z00_matterpower.dat", unpack=True)

# Compute the theoretical mass function (used as the target mass function here)
Mh = np.logspace(10, 16, 60)
dn_theory = exh.theory.Get_dndlnm(
    M=Mh, model="Tinker", theta=300, Om0=Om0, z=0.0, k=k, P=P
)

# Fit the parameters of the barrier to reproduce the mass function computed above, for a particular seed
print("Fitting the barrier")
"""params = exh.utils.Fit_Barrier(
    k=k,
    P=P,
    M=Mh,
    dndlnM=dn_theory,
    Lc=Lc,
    seed=seed,
    Nmin=Nmin,
    verbose=verbose,
    x0=x0,
    Max_iter=Max_iter,
)"""
params = [0.803958, 0.288991, 0.525464]

# Generate a halo catalogue with the barrier found above
print("Generating the halo catalogue")

halos = exh.mock.Generate_Halos_Box_from_Pk(
    k=k,
    P=P,
    nd=nd,
    Lc=Lc,
    Om0=Om0,
    z=z,
    Nmin=Nmin,
    a=params[0],
    beta=params[1],
    alpha=params[2],
    seed=int(seed*23/17),
    verbose=verbose,
)

# Measure the abundance of the halos
print("Measuring and plotting the abundance")

dn_sim = exh.simulation.Compute_Abundance(halos["Mh"], Nm=14, Lc=Lc, nd=nd)

# Plot the halo mass function
pl.clf()

pl.plot(
    Mh, dn_theory, linestyle="-", linewidth=3, marker="", color="black", label="Theory"
)
pl.errorbar(
    dn_sim["Mh"],
    dn_sim["dn"],
    yerr=dn_sim["dn_err"],
    linestyle="",
    marker="o",
    markersize=8,
    color="red",
    label="Simulation",
)

pl.xlim(np.min(halos["Mh"]), np.max(halos["Mh"]))
pl.ylim(np.min(dn_sim["dn"][dn_sim["dn"] > 0.0])
        * 0.5, 2.0 * np.max(dn_sim["dn"]))
pl.xscale("log")
pl.yscale("log")
pl.xlabel(r"$M_{h}$ $[M_{\odot}/h]$", fontsize=12)
pl.ylabel(r"$d\, n_{\rm h}/d\, ln M_{\rm h}$ $[h/{\rm Mpc}]^{3}$", fontsize=12)
pl.legend(loc="best", fontsize=12)

pl.savefig("Abundance.png")

# Define the mass bins to measure the power spectrum
Nh_bins = 9
Mh_bins = np.logspace(
    np.log10(np.min(halos["Mh"])) *
    0.99, np.log10(np.max(halos["Mh"])) * 1.01, Nh_bins
)
Mh_mean = np.zeros(Nh_bins - 1)
for i in range(Nh_bins - 1):
    mask = (halos["Mh"] > Mh_bins[i]) * (halos["Mh"] < Mh_bins[i + 1])
    Mh_mean[i] = np.mean(halos["Mh"][mask])

# Define the types of halos using the mass bins
types = (np.log10(halos["Mh"]) - np.log10(Mh_bins[0])) // (
    np.log10(Mh_bins[1]) - np.log10(Mh_bins[0])
)

# Measure the Nh_bins - 1  density grids
print("Measuring the density grids")

nd = 128
window = "CIC"
interlacing = True
grids = exh.simulation.Compute_Density_Grid(
    pos=halos["posh"],
    types=types,
    nd=nd,
    L=L,
    window=window,
    interlacing=interlacing,
    verbose=verbose,
)

# Measure the N(N+1)/2 power spectra
print("Measuring the power spectra")

Nk = 32
k_min = 0.0
k_max = 0.3
Ph = exh.simulation.Compute_Power_Spectrum(
    grid=grids,
    L=L,
    window=window,
    Nk=Nk,
    k_min=k_min,
    k_max=k_max,
    verbose=verbose,
    ntypes=Nh_bins - 1,
)
