.. ExSHalos documentation master file, created by
   sphinx-quickstart on Fri Jan 10 19:47:31 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ExSHalos
========

This library implements `ExSHalos <https://arxiv.org/abs/1906.06630>`_ and many other utilities for cosmological analysis:

- Measurement of the power/bi/tri-spectra for any number of tracers in a box;
- Population of halos with galaxies using a HOD and split of galaxies for the creation of multi-tracer catalogues;
- Creation of density grid and catalogue of particles created by LPT;
- Computation of shifted Lagrangian operators;
- Computation of EFTofLSS, for multi-tracers, using `CLASS-PT <https://github.com/Michalychforever/CLASS-PT>`_;
- Computation of simple theoretical quantities (e.g. growth function, growth rate, mass function, ...).

At this point, only a python library is exported, but all code is written in C.

This library follows the functional programing paradigm! There are no classes, random operators nor custom types.

Installation
------------

You can download the package with:

.. code-block:: sh

   git clone https://github.com/Voivodic/ExSHalos.git

Once you have all depences, installing ExSHalos is as simple as:

.. code-block:: sh

   cd ExSHalos/
   pip install . 

For more detailed installation instructions check :doc:`installation`

Quick start
-----------

Using the python interface for ExSHalos is as simple as:

.. code-block:: python

   import numpy as np
   import pyexshalos as exh
   import pylab as pl

   # Load a linear matter power spectrum
   k, Pk = np.loadtxt("matter_power_spectrum.dat", unpack = True)

   # Create the halo catalogue using exshalos
   halos = exh.mock.Generate_Halos_Box_from_Pk(k, Pk, nd = 256, Lc = 4.0, Om0 = 0.31)

   # Measure the power spectrum
   grid = exh.simulation.Compute_Density_Grid(halos["posh"], nd = 256, L = 1024.0)
   Ph = exh.simulation.Compute_Power_Spectrum(grid, L = 1024.0)

    # Plot the power spectrum
    pl.errorbar(Ph["k"], Ph["Pk"], yerr = Ph["Pk"]/np.sqrt(Ph["Nk"]), lw = 3)

    pl.xscale("log")
    pl.yscale("log")
    pl.xlabel("k [$h/$Mpc]", fontsize=15)
    pl.ylabel("P(k)  [Mpc/$h/$]$^{3}$", fontsize=15)

    pl.savefig("Ph_example.pdf")

.. toctree::
   :hidden:
  
   Installation
   Examples
   Python_API
   TODO
