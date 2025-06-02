"""
Tutorial for the halo catalogue generation
"""

# Import the libraries used in this tutotial
import numpy as np
import pylab as pl
import pyexshalos as exh

# Set parameters for the halo catalogue
OM0 = 0.307115
Z = 0.0
ND = 256
LC = 4.0
L = LC * ND
N_MIN = 1
SEED = 12345
x0 = np.array([0.62, 0.45, 0.54])
MAX_ITER = 100
VERBOSE = True

# Load the linear matter power spectrum from the MDPL2 simulation
k, P = np.loadtxt("MDPL2_z00_matterpower.dat", unpack=True)

# Compute the theoretical mass function (used as the target mass function here)
Mh = np.logspace(10, 16, 60)
dn_theory = exh.theory.Get_dndlnm(
    M=Mh, model="Tinker", theta=300, Om0=OM0, z=0.0, k=k, P=P
)

# Fit the parameters of the barrier to reproduce the mass function computed above
print("Fitting the barrier")
params = exh.utils.Fit_Barrier(
    k=k,
    P=P,
    M=Mh,
    dndlnM=dn_theory,
    Lc=LC,
    seed=SEED,
    Nmin=N_MIN,
    verbose=VERBOSE,
    x0=x0,
    Max_iter=MAX_ITER,
)

# Generate a halo catalogue with the barrier found above
print("Generating the halo catalogue")
halos = exh.mock.Generate_Halos_Box_from_Pk(
    k=k,
    P=P,
    nd=ND,
    Lc=LC,
    Om0=OM0,
    z=Z,
    Nmin=N_MIN,
    a=params[0],
    beta=params[1],
    alpha=params[2],
    seed=int(SEED * 23 / 17),
    verbose=VERBOSE,
)

# Measure the abundance of the halos
print("Measuring and plotting the abundance")
dn_sim = exh.simulation.Compute_Abundance(halos["Mh"], Nm=14, Lc=LC, nd=ND)

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
pl.ylim(np.min(dn_sim["dn"][dn_sim["dn"] > 0.0]) * 0.5, 2.0 * np.max(dn_sim["dn"]))
pl.xscale("log")
pl.yscale("log")
pl.xlabel(r"$M_{h}$ $[M_{\odot}/h]$", fontsize=12)
pl.ylabel(r"$d\, n_{\rm h}/d\, ln M_{\rm h}$ $[h/{\rm Mpc}]^{3}$", fontsize=12)
pl.legend(loc="best", fontsize=12)

pl.savefig("Abundance.png")
