# This file should be located at the root of your project: app/flake.nix
{
    description = "A development environment and runnable app for the ExSHalos Python library";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    };

    outputs = { self, nixpkgs, ... } @ inputs:
        let
            # Helper to generate outputs for all common systems
            forAllSystems = nixpkgs.lib.genAttrs nixpkgs.lib.systems.flakeExposed;

            # Create an attribute set for each system containing our common packages
            perSystem = forAllSystems (system:
                let
                    pkgs = import nixpkgs { inherit system; };
                    python = pkgs.python313;

                    # Define the package derivation for ExSHHalos
                    pyexshalos = python.pkgs.buildPythonPackage {
                        pname = "pyexshalos";
                        version = "1.0.0";
                        format = "pyproject";
                        src = ./../.;

                        nativeBuildInputs = [ pkgs.gcc python.pkgs.setuptools ];
                        buildInputs = [ pkgs.fftw pkgs.fftwFloat pkgs.gsl ];
                        propagatedBuildInputs = [ python.pkgs.numpy python.pkgs.scipy ];

                        pythonImportsCheck = [ "pyexshalos" ];

                        meta = {
                            description = "Python interface to ExSHalos";
                            homepage = "https://github.com/Voivodic/ExSHalos";
                        };
                    };

                    # Define the Python environment with the package ONCE.
                    pythonWithExSHalos = python.withPackages (ps: [ pyexshalos ]);
                in
                    # Expose these common definitions for this system
                    {
                    inherit pkgs;
                    package = pyexshalos;
                    pythonEnv = pythonWithExSHalos;
                }
            );
        in
            {
            # --- CLEANER OUTPUTS ---

            # Expose ExSHalos as a package for all systems
            # You can build it with `nix build .`
            packages = forAllSystems (system: {
                default = perSystem.${system}.package;
            });

            # Create the dev shell using the centralized definitions.
            # Enter with `nix develop`
            devShells = forAllSystems (system: {
                default = perSystem.${system}.pkgs.mkShell {
                    buildInputs = [
                        # Use the pre-built Python environment
                        perSystem.${system}.pythonEnv
                    ];
                    shellHook = ''
                    '';
                };
            });

            # Create the app using the centralized definitions.
            # Run with `nix run . -- your_script.py`
            apps = forAllSystems (system: {
                default = {
                    type = "app";
                    program = "${perSystem.${system}.pythonEnv}/bin/python";
                };
            });
        };
}
