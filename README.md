# ExSHalos

This is a python package that implements [ExSHalos](https://arxiv.org/abs/1906.06630) and many other utilities for cosmological analysis:
- Measurement of the power/bi/tri-spectra for any number of tracers in a box;
- Population of halos with galaxies using a HOD and split of galaxies for the creation of multi-tracer catalogues;
- Creation of density grid and catalogue of particles created by LPT;
- Computation of shifted Lagrangian operators;
- Computation of EFTofLSS, for multi-tracers, using [CLASS-PT](https://github.com/Michalychforever/CLASS-PT);
- Computation of simple theoretical quantities (e.g. growth function, growth rate, mass function, ...).

This library follows the functional programing paradigm! There are no classes, random operators nor custom types.

## Installation 

To install [pyExSHalos](https://github.com/Voivodic/ExSHalos) a few libraries are required:

### C requirements:
- [FFTW3](https://www.fftw.org/) (required for installation)
- [GSL/CBLAS](https://www.gnu.org/software/gsl/) (required for installation)
- [OpenMP](https://www.openmp.org/) (required for installation)
- [pip](https://pypi.org/project/pip/) (optional for building)

### Python requirements:
- [Numpy](https://numpy.org/) (required for installation)
- [setuptools](https://setuptools.pypa.io/en/latest/) (required for installation)
- [scipy](https://scipy.org/) (required for running)

After all libraries were installed, you just need to
```bash
pip install .
```
in the root folder of the project.

## Quick start

Using the python interface for ExSHalos is as simple as:

```python
   import numpy as np
   import pyexshalos as exh
   import pylab as pl

   # Load a linear matter power spectrum
   k, Pk = np.loadtxt("", unpack = True)

   # Create the halo catalogue using exshalos
   halos = exh.mock.Generate_Halos_Box_from_Pk(k, Pk, nd = 256, Lc = 4.0, Om0 = 0.31)

   # Measure the power spectrum
   grid = exh.simulation.Compute_Density_Grid(x["posh"], nd = 256, L = 1024.0)
   Ph = exh.simulation.Compute_Power_Spectrum(grid, L = 1024.0)

    # Plot the power spectrum
    pl.errorbar(Ph["k"], Ph["Pk"], yerr = Ph["Pk"]/np.sqrt(Ph["Nk"]), lw = 3)

    pl.xscale("log")
    pl.yscale("log")
    pl.xlabel("k [$h/$Mpc]", fontsize=15)
    pl.ylabel("P(k)  [Mpc/$h/$]$^{3}$", fontsize=15)

    pl.savefig("Ph_example.pdf")
```

## Documentation

More information about the installation, the API and some examples can be found at the [documentation](https://voivodic.github.io/ExSHalos/).

## TODO
Not necessarily in the priority order

### Code

- [ ] **Creation of more examples**
- [ ] **Completion of the documentation**
- [ ] **Integration with the zig's building system**
- [ ] **Compilation of C/C++ libraries**
- [ ] **Creation of zig wrappers**
- [ ] **Creation of a Dockerfile, apptainer's definition file and nix's shell.nix file**

### Physics

- [ ] **Functions for the halo/void finder (simulation module)**
- [ ] **Function for the halo/void profile (simulation module)**
- [ ] **Computation of the growth factor in modified gravity (theory momdule)**
- [ ] **Computation of the spherical collapse in modified gravity (theory module)**
- [ ] **Fit of the split of galaxies for the creation of multi-tracer catalogues (utils module)**
- [ ] **Creation of halo catalogues in the lightcone (mock module)**
