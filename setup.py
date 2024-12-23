import os

import numpy
from setuptools import Extension, find_packages, setup

# Set the paths to the include dirs
include_dirs = [
    numpy.get_include(),
    "/usr/local/include",
]

# Set the paths to the library dirs
library_dirs = [
    "/usr/local/lib",
]

# Set the extra link argumments
extra_link_args = ["-lgomp"]

# Set the extra compile argumments
extra_compile_args = [
    "-g",
    "-O3",
    "-funroll-loops",
    "-Wall",
    "-fPIC",
    "-fopenmp",
]

# Set the fftw3 libraries to float or double
DOUBLE_PRECISION = False
if DOUBLE_PRECISION:
    fftw3_libs = ["fftw3", "fftw3_omp"]
else:
    fftw3_libs = ["fftw3f", "fftw3f_omp"]

# Define the C extension modules
extensions = [
    # Module that compute the spectra and related quantities in simulated data
    Extension(
        "exshalos.spectrum.spectrum",
        sources=[
            "exshalos/spectrum/spectrum_h.c",
            "exshalos/spectrum/abundance.c",
            "exshalos/spectrum/gridmodule.c",
            "exshalos/spectrum/powermodule.c",
            "exshalos/spectrum/bimodule.c",
            "exshalos/spectrum/trimodule.c",
            "exshalos/spectrum/bias.c",
            "exshalos/spectrum/spectrum.c",
        ],
        language="c",
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["m", "gsl", "gslcblas"] + fftw3_libs,
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args,
    ),
    # Module that runs the ExSHalos
    Extension(
        "exshalos.exshalos.exshalos",
        sources=[
            "exshalos/exshalos/fftlog.c",
            "exshalos/exshalos/exshalos_h.c",
            "exshalos/exshalos/density_grid.c",
            "exshalos/exshalos/find_halos.c",
            "exshalos/exshalos/lpt.c",
            "exshalos/exshalos/box.c",
            "exshalos/exshalos/exshalos.c",
        ],
        language="c",
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["m", "fftw3", "gsl", "gslcblas"] + fftw3_libs,
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args
        + ['-DSPHERES_DIRC="%s/exshalos/exshalos/"' % (os.getcwd())],
    ),
    # Module that populate the halos using a HOD
    Extension(
        "exshalos.hod.hod",
        sources=[
            "exshalos/hod/hod_h.c",
            "exshalos/hod/populate_halos.c",
            "exshalos/hod/split_galaxies.c",
            "exshalos/hod/hod.c",
        ],
        language="c",
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["m", "gsl", "gslcblas"],
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args,
    ),
    # Module that find voids and halos
    Extension(
        "exshalos.finder.finder",
        sources=[
            "exshalos/finder/finder_h.c",
            "exshalos/finder/finder.cpp",
        ],
        language="c++",
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=["m"],
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args,
    ),
]

# Run the setup script to build the C extensions and install the package
setup(
    name="exshalos",
    version="0.1.0",
    packages=find_packages(),
    # packages=[
    #    "exshalos/",
    #    "exshalos/spectrum/",
    #    "exshalos/exshalos/",
    #    "exshalos/hod/",
    #    "exshalos/finder/",
    # ],
    ext_modules=extensions,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "h5py",
        "setuptools",
    ],
    setup_requires=["numpy"],
)
