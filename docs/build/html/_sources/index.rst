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

Quick installation
------------------

You can download the package with:

.. code-block:: sh

   git clone https://github.com/Voivodic/ExSHalos.git

Once you have all depences, installing ExSHalos is as simple as:

.. code-block:: sh

   cd ExSHalos/
   pip install . 

For more detailed installatino instruction check :doc:`installation`

.. toctree::
   :hidden:
  
   installation
   Examples
   Python_API
   C_API
   Zig_API
   TODO.rst
