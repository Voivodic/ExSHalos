Installation
============

ExSHalos is written in C/C++ and need some external libraries to work.

Even with the dependence list being small and composed of standard libraries (likely present in clusters), the installation instructions, for each library, is given below.

C/C++ dependencies:
-------------------

`FFTW3 <https://www.fftw.org/>`_ (required for installation) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FFTW3 is used for the computation of discret Fourier transform of (mainly) the grid.

It can be installed from the `source <https://www.fftw.org/>`_ or from your favorite package manager.

- From source:

Download and extract the `latest version <https://www.fftw.org/download.html>`_. Then, install for single (if used) and doble precision with openmp support:

.. code-block:: sh

   tar -zxvf fftw-{$VERSION}.tar.gz
   cd fftw-{$VERSION}
   ./configure --prefix={$PREFIX} --enable-openmp
   make install
   ./configure --prefix={$PREFIX} --enable-openmp --enable-float
   make install

Where {$VERSION} is the version of your fftw and {#PREFIX} is the path for the installation directory of fftw. 

ExSHalos need the double precision for fftlog and the single precision if it is choosen to be used in the final installation.

- Debian:

.. code-block:: sh
  
    sudo apt install libfftw3-dev

- Fedora:

.. code-block:: sh

    sudo dnf install fftw-devel

- Arch:

.. code-block:: sh

    sudo pacman -S fftw

- Nix:

.. code-block:: sh

    nix-env -iA nixpkgs.fftw

`GSL/CBLAS <https://www.gnu.org/software/gsl/>`_ (required for installation) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GLS is used for the generation of random numbers, interpolations and computation of some special functions.

It can be installed from `source <https://www.gnu.org/software/gsl/>`_ or from your favorite package manager.

- From source: 

Download and extract the `latest version <https://mirror.ibcp.fr/pub/gnu/gsl/gsl-latest.tar.gz>`_. Then, install it:

.. code-block:: sh

   tar -zxvf gsl-latest.tar.gz
   cd gsl-latest
   ./configure --prefix={$PREFIX}
   make install

Where {$PREFIX} is the path for the installation.

- Debian:

.. code-block:: sh

    sudo apt install libgsl-dev libgslcblas0

- Fedora:

.. code-block:: sh

    sudo dnf install gsl-devel gsl-cblas

- Arch:

.. code-block:: sh

    sudo pacman -S gsl gsl-cblas

- Nix:

.. code-block:: sh

    nix-env -iA nixpkgs.gsl nixpkgs.gslcblas

`OpenMP <https://www.openmp.org/>`_ (required for installation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Openmp is usually pre-installed in the c/c++ compiler. If it is not the case, you can install in your system with standard commands.

- Debian:

.. code-block:: sh

    sudo apt install libomp-dev

- Fedora:

.. code-block:: sh

    sudo dnf install llvm-omp

- Arch:

.. code-block:: sh

    sudo pacman -S gsl openmp

- Nix:

.. code-block:: sh

    nix-env -iA nixpkgs.llvmPackages.openmp

`pip <https://pypi.org/project/pip/>`_ (optional for building)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pip is the easiest way to install ExSHalos globally. However, it can also be installed with:

    python setup.py install

In the case you want to install pip, it can be done with the commands:

- Debian:

.. code-block:: sh

    sudo apt install python3-pip

- Fedora:

.. code-block:: sh

    sudo dnf install python3-pip

- Arch:

.. code-block:: sh

    sudo pacman -S python-pip

- Nix:

.. code-block:: sh

    nix-env -iA nixpkgs.python3Packages.pip

Python dependencies
-------------------

- `Numpy <https://numpy.org/>`_ (required for installation) 
- `setuptools <https://setuptools.pypa.io/en/latest/>`_ (required for installation) 
- `scipy <https://scipy.org/>`_ (required for running

Setuptools is the library used to compile the C modules and link them to the python package (throgh the .so files).

Numpy.array is the fundamental object (sorry for this OOP word) in ExSHalos. Therefore, it needs to be installed, at compilation time, because of the C/python interface. 

Scipy is used in some modules for simple interpolations, optimizations and computation of special function.

Once you have pip (or conda), these libraries can be installed with:

.. code-block:: sh

    {pip/conda} install numpy setuptools scipy

Where you have to choose between pip or conda denpending of your prefered python package manager.

Package installation
--------------------

Once all dependencies are installed, ExSHalos can be install with:

.. code-block:: sh

    git clone https://github.com/Voivodic/ExSHalos.git
    cd ExSHalos
    pip install .

Automatic and isolated installation
-----------------------------------

If you want a more isolated and segure installation, and do not want to handle the dependencies manually, ExSHalos makes available three options:

`Docker <https://www.docker.com/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a Docker image you only need to

.. code-block:: sh
   
    git clone https://github.com/Voivodic/ExSHalos.git
    cd ExSHalos
    docker build -t your_image_name .

Then, to create a Docker container and enter into its shell

.. code-block:: sh

    docker run -it --name your_container_name your_image_name

`Apptainer <https://apptainer.org/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An opensource alternative to Docker (usually used in scientific clusters) is Apptainer. You can create similar images doing

.. code-block:: sh

    git clone https://github.com/Voivodic/ExSHalos.git
    cd ExSHalos
    apptainer build your_image_name.sif exshalos.def

Then, to enter in an isolated shell

.. code-block:: sh

    apptainer shell your_image_name.sif

`Nix <https://nixos.org/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Last but not least, you can also create an ephemeral shell using Nix with flakes. For this, you only need to run

.. code-block:: sh

    git clone https://github.com/Voivodic/ExSHalos.git
    cd ExSHalos
    nix develop
