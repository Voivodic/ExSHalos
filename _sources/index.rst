.. ExSHalos documentation master file, created by
   sphinx-quickstart on Fri Jan 10 19:47:31 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ExSHalos
========

This is a python package that implements `ExSHalos <https://arxiv.org/abs/1906.06630>`_ and many other utilities for cosmological analysis:

- Measurement of the power/bi/tri-spectra for any number of tracers in a box;
- Population of halos with galaxies using a HOD and split of galaxies for the creation of multi-tracer catalogues;
- Creation of density grid and catalogue of particles created by LPT;
- Computation of shifted Lagrangian operators;
- Computation of EFTofLSS, for multi-tracers, using `CLASS-PT <https://github.com/Michalychforever/CLASS-PT>`_;
- Computation of simple theoretical quantities (e.g. growth function, growth rate, mass function, ...).

This library follows the functional programing paradigm! There are no classes, random operators nor custom types.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pyexshalos
