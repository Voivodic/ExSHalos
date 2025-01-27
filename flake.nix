{
    description = "A very basic flake";

    inputs = {
        nixpkgs.url = github:NixOS/nixpkgs/nixos-24.11;
    };

    outputs = { self, nixpkgs, ... } @ inputs: 
    let
        # Set the system and the pkgs used
        system = "x86_64-linux";
        pkgs = import nixpkgs { inherit system; };

        # Install voro++ library
        voroPP = pkgs.stdenv.mkDerivation {
            name = "voro++";

            src = pkgs.fetchurl {
                url = "https://github.com/chr1shr/voro/archive/refs/heads/master.zip";
                sha256 = "sha256-UBHCMmz0o7tQQWAlQnuX4qD1ycenxPgsvyXyBJtz9Wg";
            };
            
            buildInputs = [ 
                pkgs.unzip 
                pkgs.gcc14
            ];

            configurePhase = ''
                sed -i '14s#.*#CFLAGS+=-Wall -ansi -pedantic -O3 -fPIC#' config.mk 
            '';

            buildPhase = ''
                make
            '';

            installPhase = ''
                make install PREFIX=$out
            '';

            meta = {
                description = "A three-dimensional Voronoi cell library in C++";
                homepage = "https://math.lbl.gov/voro++/";
            };
        };

        # Install pyexshalos
        pyexshalos =  pkgs.python312Packages.buildPythonPackage {
            pname = "pyexshalos";
            version = "0.1.0";

            src = pkgs.fetchFromGitHub{
                owner = "Voivodic";
                repo = "ExSHalos";
                rev = "main";
                sha256 = "sha256-DpGuwkk2/zg7I8ycAHx+0tP/jqP7et8z+ky5vy2T8ks";
            };

            propagatedBuildInputs = [
                pkgs.python312Packages.numpy
                pkgs.python312Packages.scipy
                pkgs.fftw
                pkgs.fftwFloat
                pkgs.gsl
            ];

            meta = {
                description = "Python interface to ExSHalos";
                homepage = "https://github.com/Voivodic/ExSHalos";
            };
        };
    in
    {
        # Instructions for the creation of the shell
        devShells.${system}.default = pkgs.mkShell{
            buildInputs = [
                pkgs.python312
                pyexshalos
            ];

            shellHook = ''
            '';
        };
    };
}
