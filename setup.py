"""
Setup script for the pyexshalos package
"""
# Libraries used for the buildind process
import os
import numpy
from setuptools import Extension, setup

# Set the compiler to be used
os.environ["CC"] = "gcc"

# Set the paths to the include dirs
include_dirs = [
    numpy.get_include(),
    "/usr/local/include",
    "/usr/include",
]

# Set the paths to the library dirs
library_dirs = [
    "/usr/local/lib",
    "/usr/lib",
]

# Set the extra link argumments
extra_link_args = ["-lgomp"]

# Set the extra compile argumments
extra_compile_args = [
    "-g",
    "-funroll-loops",
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
        "pyexshalos.lib.spectrum",
        sources=[
            "src/spectrum/spectrum_h.c",
            "src/spectrum/abundance.c",
            "src/spectrum/gridmodule.c",
            "src/spectrum/powermodule.c",
            "src/spectrum/bimodule.c",
            "src/spectrum/trimodule.c",
            "src/spectrum/bias.c",
            "src/spectrum/spectrum.c",
        ],
        language="c",
        include_dirs=include_dirs + ["include/spectrum"],
        library_dirs=library_dirs,
        libraries=["m", "gsl", "gslcblas"] + fftw3_libs,
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args,
    ),
    # Module that runs the ExSHalos
    Extension(
        "pyexshalos.lib.exshalos",
        sources=[
            "src/exshalos/fftlog.c",
            "src/exshalos/exshalos_h.c",
            "src/exshalos/density_grid.c",
            "src/exshalos/find_halos.c",
            "src/exshalos/cells_in_spheres.c",
            "src/exshalos/lpt.c",
            "src/exshalos/box.c",
            "src/exshalos/exshalos.c",
        ],
        language="c",
        include_dirs=include_dirs + ["include/exshalos"],
        library_dirs=library_dirs,
        libraries=["m", "fftw3", "gsl", "gslcblas"] + fftw3_libs,
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args
    ),
    # Module that populate the halos using a HOD
    Extension(
        "pyexshalos.lib.hod",
        sources=[
            "src/hod/hod_h.c",
            "src/hod/populate_halos.c",
            "src/hod/split_galaxies.c",
            "src/hod/hod.c",
        ],
        language="c",
        include_dirs=include_dirs + ["include/hod"],
        library_dirs=library_dirs,
        libraries=["m", "gsl", "gslcblas"],
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args,
    ),
    # Module that computes some analytical quantities
    Extension(
        "pyexshalos.lib.analytical",
        sources=[
            "src/analytical/fftlog.c",
            "src/analytical/analytical_h.c",
            "src/analytical/clpt.c",
            "src/analytical/analytical.c",
        ],
        language="c",
        include_dirs=include_dirs + ["include/analytical"],
        library_dirs=library_dirs,
        libraries=["m", "fftw3", "gsl", "gslcblas"],
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args
    ),
]

# Run the setup script to build the C extensions and install the package
setup(
    name="pyexshalos",
    version="1.0.0",
    ext_modules=extensions,
)
