"""
Tutorial for creating galaxy catalogues with two types of galaxies
"""

# Import the libraries used in this tutorial
import numpy as np
import pylab as pl
import pyexshalos as exh

# Set parameters for the halo catalogue
Om0 = 0.307115
z = 0.0
Nd = 256
Lc = 1.0
L = Lc * Nd
N_MIN = 1
SEED = 12345
VERBOSE = True

# Load the linear matter power spectrum from the MDPL2 simulation
klin, Plin = np.loadtxt("MDPL2_z00_matterpower.dat", unpack=True)

# Best fit parameters found with pyexshalos.utils.Fit_Barrier
PARAMS = [0.803958, 0.288991, 0.525464]

# Generate a halo catalogue with the barrier define above
print("Generating the halo catalogue")
halos = exh.mock.Generate_Halos_Box_from_Pk(
    k=klin,
    P=Plin,
    nd=Nd,
    Lc=Lc,
    Om0=Om0,
    z=z,
    Nmin=N_MIN,
    a=PARAMS[0],
    beta=PARAMS[1],
    alpha=PARAMS[2],
    OUT_LPT=False,
    seed=SEED,
    verbose=VERBOSE,
)

# Populate the halos with galaxies
print("Populating the halos with galaxies")
gals = exh.mock.Generate_Galaxies_from_Halos(
    posh=halos["posh"],
    Mh=halos["Mh"],
    nd=Nd,
    Lc=Lc,
    Om0=Om0,
    z=z,
    logMmin=13.25424743,
    siglogM=0.26461332,
    logM0=13.28383025,
    logM1=14.32465146,
    alpha=1.00811277,
    sigma=0.5,
    seed=SEED,
    OUT_VEL=False,
    OUT_FLAG=True,
    verbose=VERBOSE,
)

# Split the galaxies into two populations
print("Splitting the galaxies into two populations")
gals_types = exh.mock.Split_Galaxies(
    Mh=halos["Mh"],
    Flag=gals["flag"],
    params_cen = np.array([37.10265321, -5.07596644, 0.17497771]),
    params_sat = np.array([19.84341938, -2.8352781, 0.10443049]),
    seed = SEED,
    verbose = VERBOSE,
)

# Compute the density grids
print("Computing the density grids")
WINDOW = "CIC"
INTERLACING = True
grids = exh.simulation.Compute_Density_Grid(
    pos=gals["posg"],
    types=np.abs(gals_types),
    nd=Nd,
    L=L,
    window=WINDOW,
    interlacing=INTERLACING,
    verbose=VERBOSE,
)

# Measure the power spectra
print("Measuring the power spectra")
NK = 32
K_MIN = 0.0
K_MAX = 0.3
P_sim = exh.simulation.Compute_Power_Spectrum(
    grid=grids,
    L=L,
    window=WINDOW,
    Nk=NK,
    k_min=K_MIN,
    k_max=K_MAX,
    verbose=VERBOSE,
    ntypes=2,
)

# Plot the power spectra
pl.clf()

pl.errorbar(P_sim["k"],
            P_sim["Pk"][0],
            yerr=P_sim["Pk"][0]/P_sim["Nk"],
            linestyle="",
            marker="o",
            markersize=6,
            label=r"$P_{11}(k)$",
            )
pl.errorbar(P_sim["k"],
            P_sim["Pk"][1],
            yerr=P_sim["Pk"][1]/P_sim["Nk"],
            linestyle="",
            marker="o",
            markersize=6,
            label=r"$P_{12}(k)$",
            )
pl.errorbar(P_sim["k"],
            P_sim["Pk"][2],
            yerr=P_sim["Pk"][2]/P_sim["Nk"],
            linestyle="",
            marker="o",
            markersize=6,
            label=r"$P_{22}(k)$",
            )

pl.xscale("linear")
pl.yscale("log")
pl.xlabel(r"$k$ [$h/$Mpc]", fontsize=12)
pl.ylabel(r"$P(k)$ [Mpc$/h]^{3}$", fontsize=12)
pl.legend(loc="best", fontsize=12)

pl.tight_layout()
pl.savefig("Multi_hod.png")
