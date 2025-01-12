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

## Documentation

More information about the installation, the API and some examples can be found at the [documentation](voivodic.github.io/ExSHalos/).

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
