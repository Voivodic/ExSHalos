# Installing ExSHalos

You have three main options to install ExSHalos. For the case you do not want to handle the dependencies manually and want an isolated working space (strongly recommended).
All options fetch the package from the [GitHub repository](https://github.com/Voivodic/ExSHalos) and not the local copy.

## [Docker](https://www.docker.com/)

To create a Docker image you only need to
```bash
git clone https://github.com/Voivodic/ExSHalos.git
cd ExSHalos
docker build -t your_image_name -f .
```

Then, to create a Docker container and enter into its shell
```bash
docker run -it --name your_container_name your_image_name
```

## [Apptainer](https://apptainer.org/)

An open source alternative to Docker (usually used in scientific clusters) is Apptainer. You can create similar images doing:
```bash
git clone https://github.com/Voivodic/ExSHalos.git
cd ExSHalos
apptainer build your_image_name.sif exshalos.def
```

Then, to enter in an isolated shell
```bash
apptainer shell your_image_name.sif
```

## [Nix](https://nixos.org/)

Last but not least, you can also create an ephemeral shell using Nix with flakes. For this, you only need to run:
```bash
git clone https://github.com/Voivodic/ExSHalos.git
cd ExSHalos
nix develop . 
```

The package can also be installed from my [GitHub repository](https://github.com/Voivodic/nix-derivations) of nix packages.
