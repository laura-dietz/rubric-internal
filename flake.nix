{
  description = "ExamPP";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
  inputs.nixpkgs.follows  = "dspy-nix/nixpkgs";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.dspy-nix.url = "git+https://git.smart-cactus.org/ben/dspy-nix";

  outputs = inputs@{ self, nixpkgs, flake-utils, dspy-nix, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        pythonOverrides = self: super: {
          pylatex = self.callPackage ./nix/pylatex.nix {};
          trec-car-tools = self.callPackage ./nix/trec-car-tools.nix {};
          ir_datasets = self.callPackage ./nix/ir_datasets.nix {};
          unlzw3 = self.callPackage ./nix/unlzw3.nix {};
          pyautocorpus = self.callPackage ./nix/pyautocorpus.nix {};
          zlib-state = self.callPackage ./nix/zlib-state.nix {};
          warc3-wet = self.callPackage ./nix/warc3-wet.nix {};
          warc3-wet-clueweb09 = self.callPackage ./nix/warc3-wet-clueweb09.nix {};
          backports-asyncio-queues = self.buildPythonPackage rec {
            pname = "backports_asyncio_queues";
            version = "0.1.2";
            pyproject = true;
            build-system = [ self.hatchling ];
            src = self.fetchPypi {
              inherit pname version;
              hash = "sha256-CnQtfAKZ3im27lwqEe8IUak8r0roK8sas5icMPIOdiM=";
            };
          };
          exampp = self.buildPythonPackage {
            name = "exampp";
            src = ./.;
            format = "pyproject";
            propagatedBuildInputs = with self; [
              setuptools
              pydantic
              pylatex
              scipy
              openai
              torch
              transformers
              nltk
              ir_datasets
              fuzzywuzzy
              duckdb
              backports-asyncio-queues
            ];
            doCheck = false;
          };
        };

        mkShell = target: (dspy-nix.lib.${system}.mkShell {
          inherit target;
          pythonOverrides = [ pythonOverrides ];
          packages = ps: with ps; [ duckdb ];
          pythonPackages = ps: with ps; [
            backports-asyncio-queues
            pydantic
            fuzzywuzzy
            nltk
            mypy
            jedi
            pylatex
            trec-car-tools
            ir_datasets
            duckdb
          ];
        });

      in {
        lib.pythonOverrides = pythonOverrides;
        packages.exampp = (pkgs.python3.override {
          packageOverrides = pythonOverrides;
        }).pkgs.exampp;

        devShells.default = self.outputs.devShells.${system}.cuda;
        devShells.cpu = mkShell "cpu";
        devShells.rocm = mkShell "rocm";
        devShells.cuda = mkShell "cuda";
      }
    );

  nixConfig = {
    substituters = [ "https://cache.nixos.org" "https://dspy-nix.cachix.org" ];
    trusted-public-keys = [ "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY=" "dspy-nix.cachix.org-1:VJ553D0iJVoA8ov2+ly+dLnGHarfSQpemzVW6dY6CfE=" ];
  };
  }


        # checks.default = pkgs.stdenv.mkDerivation {
        #   name = "check-exampp";
        #   src = ./.;
        #   nativeBuildInputs =
        #     let py = dspy-nix.outputs.packages.${system}.python-cuda.withPackages (ps: [ (ps.callPackage ./exampp.nix {}) ]);
        #     in [py];
        #   buildPhase = '' python scripts/minimal_tests.py '';
        # };
