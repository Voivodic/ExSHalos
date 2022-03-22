import os 
import numpy
try:
    from setuptools import setup, Extension, sysconfig
    setup
except ImportError:
    from distutils.core import setup, Extension
    from distutils import sysconfig
    setup
    
def get_include_dirs():
    try:
        import numpy
    except:
        raise ValueError("You should install numpy -- pip install numpy")
    from pkg_resources import get_build_platform

    include_dirs = [os.path.join(os.getcwd(), 'include'),
                    numpy.get_include(), "/usr/","$HOME/lib/", "$HOME/local/include/", "/usr/lib/x86_64-linux-gnu/openmpi/include/"]

    return include_dirs

#Append a include or library from the eviroment
def append_from_env(envvar, array, subdir="include"):
    libhome = os.environ.get(envvar, sysconfig.get_config_var(envvar))

    if libhome is not None:
        array.append("%s/%s" % (libhome, subdir))
        print("appended dir %s/%s" % (libhome, subdir))
    else:
        print("env variable %s not found" % envvar)

#Set the arrays with the directories of includes and libs 
include_dirs = get_include_dirs()
library_dirs =["$HOME/lib/lib", "$HOME/local/lib"]
libraries = ['m', 'gsl', 'gslcblas', 'dl']
extra_compile_args = ['-g', '-ggdb3', '-Wall', '-fPIC', '-fopenmp']
                      
#Check the double precision argument and set the FFTW3 flags
double_precision = False
if double_precision is True:
    extra_compile_args.append('-DDOUBLEPRECISION_FFTW')
    libraries.append('fftw3')
    libraries.append('fftw3_omp')
else:
    libraries.append('fftw3f')
    libraries.append('fftw3f_omp')
include_dirs.append("/usr/lib/x86_64-linux-gnu/openmpi/include/")

#Change the time of files to recompile them
def recompile_c_modules(dprecision):
    try:
        import exshalos
        if dprecision is True: #double precision
            if (exshalos.spectrum.grid.check_precision() == 8):
                return False
        else:
            if (exshalos.spectrum.grid.check_precision() == 4):
                return False
        return True
    except:
        return True #The code will be recompiled anyway

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

if(recompile_c_modules(double_precision)):
    touch(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exshalos/spectrum", "spectrum_h.c"))
    touch(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exshalos/spectrum", "spectrum.c"))
    touch(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exshalos/spectrum", "gridmodule.c"))
    touch(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exshalos/spectrum", "powermodule.c"))
    touch(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exshalos/spectrum", "bimodule.c"))
    touch(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exshalos/spectrum", "trimodule.c"))

    touch(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exshalos/exshalos", "exshalos_h.c"))
    touch(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exshalos/exshalos", "fftlog.c"))
    touch(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exshalos/exshalos", "exshalos.c")) 
    touch(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exshalos/exshalos", "density_grid.c")) 
    touch(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exshalos/exshalos", "find_halos.c")) 

#Add the environmental path to FFTW3 and GSL
append_from_env("FFTW_HOME", include_dirs, "include")
append_from_env("GSL_HOME", include_dirs, "include")
append_from_env("FFTW_HOME", library_dirs, "lib")
append_from_env("GSL_HOME", library_dirs, "lib")

#Define the extra modules to be used by the library (files .c)

spectrum = Extension("exshalos.spectrum.spectrum",
                          sources = ["exshalos/spectrum/spectrum.c", "exshalos/spectrum/spectrum_h.c", "exshalos/spectrum/gridmodule.c", "exshalos/spectrum/powermodule.c", "exshalos/spectrum/bimodule.c", "exshalos/spectrum/trimodule.c"],
                          extra_compile_args = extra_compile_args,
                          extra_link_args=['-lgomp'],
                          include_dirs = include_dirs,
                          library_dirs = library_dirs,
                          libraries = libraries)

exshalos = Extension("exshalos.exshalos.exshalos",
                          sources = ["exshalos/exshalos/exshalos.c", "exshalos/exshalos/fftlog.c", "exshalos/exshalos/exshalos_h.c", "exshalos/exshalos/density_grid.c", "exshalos/exshalos/find_halos.c"],
                          extra_compile_args = extra_compile_args,
                          extra_link_args=['-lgomp'],
                          include_dirs = include_dirs,
                          library_dirs = library_dirs,
                          libraries = [*libraries, 'fftw3'])                         

ext_modules = [spectrum, exshalos]

#Define the setup to be run
setup(
    name = 'ExSHalos',
    packages = ['exshalos/', 'exshalos/spectrum/', 'exshalos/exshalos/'],
    version = '0.1',
    description = 'Cosmology',
    author = 'Rodrigo Voivodic',
    author_email = 'rodrigo.voivodic@usp.br',
    ext_modules = ext_modules,
    install_requires = ["scipy", "matplotlib", "h5py", "cython", "setuptools", "camb", "pyfftw"],
    license = 'MIT'
)
